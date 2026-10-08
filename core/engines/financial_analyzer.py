from __future__ import annotations

from typing import Any

import pandas as pd

from core.data.schema_profiler import SchemaProfiler
from core.utils.dataframe_utils import is_numeric_column


class FinancialAnalyzer:
    """
    Dataset-agnostic financial/quantitative analyzer.

    Responsibilities:
    - Inspect the actual dataset schema.
    - Analyze dynamically selected numeric columns.
    - Calculate descriptive statistics.
    - Perform generic categorical group analysis.
    - Avoid assuming columns such as Revenue, Sales, Profit, or Region.
    - Never treat identifier columns as financial metrics.
    """

    def run(
        self,
        df: pd.DataFrame,
        metrics: dict[str, Any] | None = None,
    ) -> dict | list[str]:

        # ---------------------------------------------------------
        # Validate dataframe
        # ---------------------------------------------------------

        if df is None or df.empty:
            return {
                "error": "Empty dataframe."
            }

        # ---------------------------------------------------------
        # Profile actual dataset
        # ---------------------------------------------------------

        profile = SchemaProfiler.profile(df)

        useful_numeric_columns = profile.get(
            "useful_numeric_columns",
            [],
        )

        if not useful_numeric_columns:
            return {
                "error": (
                    "No useful numeric columns were found "
                    "in the dataset."
                )
            }

        # ---------------------------------------------------------
        # Resolve requested metric columns
        # ---------------------------------------------------------

        selected_columns = self._resolve_metric_columns(
            df=df,
            metrics=metrics,
            useful_numeric_columns=useful_numeric_columns,
        )

        # ---------------------------------------------------------
        # Dataset-agnostic fallback
        # ---------------------------------------------------------

        if not selected_columns:
            selected_columns = useful_numeric_columns

        # ---------------------------------------------------------
        # Generate insights
        # ---------------------------------------------------------

        insights: list[str] = []

        # Dataset overview
        insights.append(
            f"Dataset contains {len(df)} records and "
            f"{len(df.columns)} columns."
        )

        # ---------------------------------------------------------
        # Numeric metric analysis
        # ---------------------------------------------------------

        for column in selected_columns:

            if column not in df.columns:
                continue

            if not is_numeric_column(df, column):
                continue

            series = df[column].dropna()

            if series.empty:
                continue

            total = series.sum()
            average = series.mean()
            minimum = series.min()
            maximum = series.max()

            insights.append(
                f"'{column}': "
                f"total={self._round_value(total)}, "
                f"average={self._round_value(average)}, "
                f"minimum={self._round_value(minimum)}, "
                f"maximum={self._round_value(maximum)}."
            )

        # ---------------------------------------------------------
        # Generic categorical group analysis
        # ---------------------------------------------------------

        categorical_columns = profile.get(
            "categorical_columns",
            [],
        )

        # Limit automatic dimensions so large datasets
        # do not generate excessive output.
        for dimension in categorical_columns[:3]:

            if dimension not in df.columns:
                continue

            unique_values = (
                df[dimension]
                .dropna()
                .unique()
            )

            if len(unique_values) < 2:
                continue

            # Analyze only a few metrics per dimension.
            for metric_column in selected_columns[:3]:

                if metric_column not in df.columns:
                    continue

                if not is_numeric_column(
                    df,
                    metric_column,
                ):
                    continue

                grouped = (
                    df.groupby(dimension)[metric_column]
                    .sum()
                    .dropna()
                )

                if grouped.empty:
                    continue

                highest_group = grouped.idxmax()
                lowest_group = grouped.idxmin()

                insights.append(
                    f"'{metric_column}' highest "
                    f"by {dimension}: "
                    f"{highest_group} "
                    f"({self._round_value(grouped.loc[highest_group])})."
                )

                insights.append(
                    f"'{metric_column}' lowest "
                    f"by {dimension}: "
                    f"{lowest_group} "
                    f"({self._round_value(grouped.loc[lowest_group])})."
                )

        # ---------------------------------------------------------
        # Final safety
        # ---------------------------------------------------------

        if not insights:
            return [
                "No financial insights could be generated "
                "from the available dataset."
            ]

        return insights

    # =============================================================
    # Metric resolution
    # =============================================================

    def _resolve_metric_columns(
        self,
        df: pd.DataFrame,
        metrics: dict[str, Any] | None,
        useful_numeric_columns: list[str],
    ) -> list[str]:
        """
        Resolve metric columns supplied by Query Understanding /
        ColumnResolver.

        Supports formats such as:

            {"primary": "Sales"}

            {"metric": "Profit"}

            {"metrics": ["Sales", "Profit"]}

        No dataset-specific column names are assumed.
        """

        if not isinstance(metrics, dict):
            return []

        selected_columns: list[str] = []

        for value in metrics.values():

            # -----------------------------------------------------
            # Single column
            # -----------------------------------------------------

            if isinstance(value, str):

                if (
                    value in df.columns
                    and value in useful_numeric_columns
                    and value not in selected_columns
                ):
                    selected_columns.append(value)

            # -----------------------------------------------------
            # Multiple columns
            # -----------------------------------------------------

            elif isinstance(value, (list, tuple, set)):

                for column in value:

                    if not isinstance(column, str):
                        continue

                    if (
                        column in df.columns
                        and column in useful_numeric_columns
                        and column not in selected_columns
                    ):
                        selected_columns.append(column)

        return selected_columns

    # =============================================================
    # Numeric formatting
    # =============================================================

    @staticmethod
    def _round_value(value: Any) -> float:
        """
        Safely round numeric values for output.
        """

        try:
            return round(float(value), 2)

        except (TypeError, ValueError):
            return value