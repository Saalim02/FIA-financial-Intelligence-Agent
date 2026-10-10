from __future__ import annotations

import re
import math
from typing import Any

import pandas as pd

from core.data.schema_profiler import SchemaProfiler


class FinancialAnalyzer:
    """
    Dataset-agnostic financial/quantitative analyzer.

    The requested operation determines the evidence returned:
    - total / sum
    - average / mean
    - minimum / min
    - maximum / max
    - group-by
    - descriptive fallback

    The analyzer does not assume any fixed dataset schema.
    It uses the schema profiler and the actual columns present
    in the supplied dataframe.
    """

    def run(
        self,
        df: pd.DataFrame,
        metrics: dict[str, Any] | None = None,
        question: str = "",
    ) -> dict | list[str]:

        if df is None or df.empty:
            return {"error": "Empty dataframe."}

        profile = SchemaProfiler.profile(df)

        useful_numeric_columns = profile.get(
            "useful_numeric_columns",
            [],
        )

        if not useful_numeric_columns:
            return {
                "error": "No useful numeric columns were found in the dataset."
            }

        # Resolve the metric selected by the query-understanding layer.
        selected_columns = self._resolve_metric_columns(
            df,
            metrics,
            useful_numeric_columns,
        )

        # Safe fallback: use the first useful numeric column.
        if not selected_columns:
            selected_columns = useful_numeric_columns[:1]

        metric = selected_columns[0]

        # Detect the requested analytical operation.
        operation = self._detect_operation(question)

        # --------------------------------------------------------------
        # Scalar operations
        # --------------------------------------------------------------

        if operation == "total":
            return [self._total(df, metric)]

        if operation == "average":
            return [self._average(df, metric)]

        if operation == "minimum":
            return [self._minimum(df, metric)]

        if operation == "maximum":
            return [self._maximum(df, metric)]

        # --------------------------------------------------------------
        # Group-by operation
        # --------------------------------------------------------------

        if operation == "group_by":
            return self._group_by(
                df,
                metric,
                profile.get("categorical_columns", []),
                question,
            )

        # --------------------------------------------------------------
        # Safe generic fallback
        # --------------------------------------------------------------

        series = df[metric].dropna()

        if series.empty:
            return [f"No numeric values available for '{metric}'."]

        return [
            f"'{metric}': "
            f"total={self._round_value(series.sum())}, "
            f"average={self._round_value(series.mean())}, "
            f"minimum={self._round_value(series.min())}, "
            f"maximum={self._round_value(series.max())}."
        ]

    # ==================================================================
    # Operations
    # ==================================================================

    @staticmethod
    def _total(df: pd.DataFrame, metric: str) -> str:
        series = df[metric].dropna()

        if series.empty:
            return f"No numeric values available for '{metric}'."

        value = series.sum()

        return f"'{metric}': total={FinancialAnalyzer._round_value(value)}."

    @staticmethod
    def _average(df: pd.DataFrame, metric: str) -> str:
        series = df[metric].dropna()

        if series.empty:
            return f"No numeric values available for '{metric}'."

        value = series.mean()

        return f"'{metric}': average={FinancialAnalyzer._round_value(value)}."

    @staticmethod
    def _minimum(df: pd.DataFrame, metric: str) -> str:
        series = df[metric].dropna()

        if series.empty:
            return f"No numeric values available for '{metric}'."

        value = series.min()

        return f"'{metric}': minimum={FinancialAnalyzer._round_value(value)}."

    @staticmethod
    def _maximum(df: pd.DataFrame, metric: str) -> str:
        series = df[metric].dropna()

        if series.empty:
            return f"No numeric values available for '{metric}'."

        value = series.max()

        return f"'{metric}': maximum={FinancialAnalyzer._round_value(value)}."

    # ==================================================================
    # Group-by
    # ==================================================================

    def _group_by(
        self,
        df: pd.DataFrame,
        metric: str,
        categorical_columns: list[str],
        question: str,
    ) -> list[str]:

        dimension = self._resolve_group_dimension(
            question,
            categorical_columns,
            df,
        )

        if dimension is None:
            return [
                f"No categorical grouping dimension was identified for '{metric}'."
            ]

        # Ensure the metric and dimension exist.
        if metric not in df.columns:
            return [f"Metric '{metric}' was not found in the dataset."]

        if dimension not in df.columns:
            return [
                f"Grouping dimension '{dimension}' was not found in the dataset."
            ]

        # Group by the requested dimension and sum the selected metric.
        aggregation = self._group_aggregation(question)
        values = df.groupby(dimension, dropna=False)[metric]
        grouped = (values.mean() if aggregation == "mean" else
                   values.min() if aggregation == "min" else
                   values.max() if aggregation == "max" else
                   values.count() if aggregation == "count" else values.sum(min_count=1))
        grouped = grouped.sort_values(ascending=False)

        if grouped.empty:
            return [f"No group-by results available for '{metric}'."]

        findings: list[str] = []

        for value, amount in grouped.items():

            # Represent missing grouping values clearly.
            if pd.isna(value):
                display_value = "Unknown"
            else:
                display_value = str(value)

            findings.append(
                f"'{metric}' by {dimension}: "
                f"{display_value}={self._round_value(amount)}."
            )

        return findings

    def chart_data(
        self,
        df: pd.DataFrame,
        metrics: dict[str, Any] | None,
        question: str,
    ) -> dict[str, Any] | None:
        """Return a bar chart only for explicitly grouped questions."""
        if self._detect_operation(question) != "group_by" or df is None or df.empty:
            return None
        profile = SchemaProfiler.profile(df)
        selected = self._resolve_metric_columns(df, metrics, profile["useful_numeric_columns"])
        metric = selected[0] if selected else (profile["useful_numeric_columns"] or [None])[0]
        if metric is None:
            return None
        dimension = self._resolve_group_dimension(question, profile["categorical_columns"], df)
        if dimension is None:
            return None
        aggregation = self._group_aggregation(question)
        values = df.groupby(dimension, dropna=False)[metric]
        grouped = (values.mean() if aggregation == "mean" else
                   values.min() if aggregation == "min" else
                   values.max() if aggregation == "max" else
                   values.count() if aggregation == "count" else values.sum(min_count=1))
        grouped = grouped.sort_values(ascending=False)
        if grouped.empty:
            return None
        x = ["Unknown" if pd.isna(value) else str(value) for value in grouped.index]
        y = [self._finite_or_none(value) for value in grouped.tolist()]
        return {
            "chart_type": "bar",
            "title": f"{metric} by {dimension}",
            "x_label": dimension,
            "y_label": metric,
            "x": x,
            "y": y,
            "metric": metric,
            "dimension": dimension,
            "aggregation": aggregation,
            "value_format": "number",
        }

    @staticmethod
    def _group_aggregation(question: str) -> str:
        q = (question or "").lower()
        if re.search(r"\b(average|avg|mean)\b", q):
            return "mean"
        if re.search(r"\b(minimum|min|lowest|smallest)\b", q):
            return "min"
        if re.search(r"\b(maximum|max|highest|largest)\b", q):
            return "max"
        if re.search(r"\b(count|number of)\b", q):
            return "count"
        return "sum"

    # ==================================================================
    # Query understanding
    # ==================================================================

    @staticmethod
    def _detect_operation(question: str) -> str:
        """
        Detect the analytical operation requested by the user.

        IMPORTANT:
        Group-by is checked BEFORE scalar aggregations because queries
        such as:

            "total sales by Region"
            "average profit by category"
            "maximum sales per state"

        contain both an aggregation word and a grouping word.

        In those cases, group-by must take priority.
        """

        q = (question or "").lower().strip()

        # --------------------------------------------------------------
        # Group-by has highest priority.
        # --------------------------------------------------------------

        if re.search(
            r"\b(by|per|breakdown|break\s+down|group(?:ed)?|group\s+by)\b",
            q,
        ):
            return "group_by"

        # --------------------------------------------------------------
        # Scalar aggregations.
        # --------------------------------------------------------------

        if re.search(
            r"\b(total|sum|summed|overall\s+total)\b",
            q,
        ):
            return "total"

        if re.search(
            r"\b(average|avg|mean)\b",
            q,
        ):
            return "average"

        if re.search(
            r"\b(minimum|min|lowest|smallest)\b",
            q,
        ):
            return "minimum"

        if re.search(
            r"\b(maximum|max|highest|largest)\b",
            q,
        ):
            return "maximum"

        return "descriptive"

    # ==================================================================
    # Group dimension resolution
    # ==================================================================

    @staticmethod
    def _resolve_group_dimension(
        question: str,
        categorical_columns: list[str],
        df: pd.DataFrame,
    ) -> str | None:
        """
        Resolve the grouping dimension from the actual dataset schema.

        No dataset-specific columns are hardcoded.

        Examples:

            "sales by Region"
            "profit by Product Category"
            "sales per Customer Segment"

        The method first looks for an exact/normalized column mention.
        """

        if not question:
            return None

        q = question.lower().strip()

        if not categorical_columns:
            return None

        # --------------------------------------------------------------
        # 1. Exact case-insensitive column match.
        # --------------------------------------------------------------

        for column in categorical_columns:
            if column.lower() in q:
                return column

        # --------------------------------------------------------------
        # 2. Normalized column-name matching.
        #
        # Converts:
        #   "Product Category" -> "product category"
        #   "Product-Category" -> "product category"
        #   "product_category" -> "product category"
        # --------------------------------------------------------------

        normalized_question = FinancialAnalyzer._normalize_text(q)

        for column in categorical_columns:

            normalized_column = FinancialAnalyzer._normalize_text(
                column
            )

            if not normalized_column:
                continue

            if normalized_column in normalized_question:
                return column

        # --------------------------------------------------------------
        # 3. Explicit "by <dimension>" extraction.
        #
        # Example:
        #   "total sales by region"
        #
        # We extract "region" and compare it against the actual
        # categorical columns.
        # --------------------------------------------------------------

        match = re.search(
            r"\b(?:by|per)\s+([a-z0-9][a-z0-9 _-]*?)(?:\?|$)",
            q,
        )

        if match:
            requested = self._normalize_text(match.group(1))

            # Remove trailing analytical words that are not part
            # of the dimension name.
            requested = re.sub(
                r"\b(total|sum|average|avg|mean|minimum|min|maximum|max)\b",
                "",
                requested,
            ).strip()

            for column in categorical_columns:

                normalized_column = self._normalize_text(
                    column
                )

                if normalized_column == requested:
                    return column

        # --------------------------------------------------------------
        # 4. Token-level matching.
        #
        # This helps with natural queries where the user mentions
        # the column name along with additional words.
        # --------------------------------------------------------------

        question_tokens = set(normalized_question.split())

        best_column: str | None = None
        best_score = 0

        for column in categorical_columns:

            normalized_column = self._normalize_text(column)

            column_tokens = set(normalized_column.split())

            if not column_tokens:
                continue

            score = len(question_tokens.intersection(column_tokens))

            if score > best_score:
                best_score = score
                best_column = column

        if best_score > 0:
            return best_column

        return None

    # ==================================================================
    # Metric resolution
    # ==================================================================

    @staticmethod
    def _resolve_metric_columns(
        df: pd.DataFrame,
        metrics: dict[str, Any] | None,
        useful_numeric_columns: list[str],
    ) -> list[str]:
        """
        Resolve metric columns supplied by the query-understanding layer.

        Only columns that actually exist in the dataframe and are
        classified as useful numeric columns are accepted.
        """

        if not isinstance(metrics, dict):
            return []

        selected_columns: list[str] = []

        for value in metrics.values():

            # ----------------------------------------------------------
            # Single metric
            # ----------------------------------------------------------

            if isinstance(value, str):

                if (
                    value in df.columns
                    and value in useful_numeric_columns
                    and value not in selected_columns
                ):
                    selected_columns.append(value)

            # ----------------------------------------------------------
            # Multiple metrics
            # ----------------------------------------------------------

            elif isinstance(value, (list, tuple, set)):

                for column in value:

                    if (
                        isinstance(column, str)
                        and column in df.columns
                        and column in useful_numeric_columns
                        and column not in selected_columns
                    ):
                        selected_columns.append(column)

        return selected_columns

    # ==================================================================
    # Utility helpers
    # ==================================================================

    @staticmethod
    def _normalize_text(value: str) -> str:
        """
        Normalize text for dataset-agnostic column matching.

        Example:

            "Product_Category"
                ->
            "product category"

            "Product-Category"
                ->
            "product category"
        """

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            value.lower(),
        )

        return re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

    @staticmethod
    def _round_value(value: Any) -> float | None:
        try:
            numeric = float(value)
            return round(numeric, 2) if math.isfinite(numeric) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _finite_or_none(value: Any) -> float | None:
        try:
            numeric = float(value)
            return numeric if math.isfinite(numeric) else None
        except (TypeError, ValueError):
            return None
