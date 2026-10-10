from __future__ import annotations

import re
import math

import pandas as pd

from core.data.schema_profiler import SchemaProfiler


class TimeAnalyzer:
    """Dataset-agnostic time series analysis and chart data generation."""

    def analyze(self, df, entities=None, question=""):
        if df is None or df.empty:
            return {"error": "Empty dataframe.", "insights": [], "visualization": None}

        profile = SchemaProfiler.profile(df)
        numeric = profile["useful_numeric_columns"]
        requested = [column for column in (entities or []) if column in numeric]
        metrics = requested or numeric
        if not metrics:
            return {"error": "No useful numeric columns available for trend analysis.", "insights": [], "visualization": None}

        date_columns = profile["datetime_columns"]
        if not date_columns:
            return {"error": "No usable date/time column detected for trend analysis.", "insights": [], "visualization": None}

        normalized_question = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
        date_column = next(
            (column for column in date_columns if re.sub(r"[^a-z0-9]+", " ", column.lower()).strip() in normalized_question),
            date_columns[0],
        )
        frame = df[[date_column, *metrics]].copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
        frame = frame.dropna(subset=[date_column])
        if frame.empty:
            return {"error": "Date column could not be parsed.", "insights": [], "visualization": None}

        granularity = self._granularity(frame[date_column], question)
        frequency = {"year": "Y", "quarter": "Q", "month": "M", "week": "W", "day": "D"}[granularity]
        frame["_period"] = frame[date_column].dt.to_period(frequency)
        aggregation = self._aggregation(question)
        insights = []
        grouped_by_metric = {}
        for metric in metrics:
            values = frame.groupby("_period", sort=True)[metric]
            grouped = values.mean() if aggregation == "mean" else values.count() if aggregation == "count" else values.sum(min_count=1)
            grouped = grouped.map(lambda value: value if self._finite_or_none(value) is not None else float("nan"))
            if grouped.empty:
                continue
            grouped_by_metric[metric] = grouped
            period_label = granularity.title()
            insights.extend(self._build_trend_insights(grouped, metric, period_label))

        if not grouped_by_metric:
            return {"error": "Insufficient data for trend analysis.", "insights": [], "visualization": None}

        metric = metrics[0]
        grouped = grouped_by_metric[metric]
        x = [str(period) for period in grouped.index]
        y = [self._finite_or_none(value) for value in grouped.tolist()]
        valid_values = [value for value in y if value is not None]
        if not valid_values:
            return {"insights": ["No numeric values available for trend analysis."], "visualization": None}
        valid_grouped = grouped.dropna()
        start, end = float(valid_grouped.iloc[0]), float(valid_grouped.iloc[-1])
        change_percent = ((end - start) / abs(start) * 100) if start else None
        if change_percent is not None and not math.isfinite(change_percent):
            change_percent = None
        direction = "increasing" if end > start else "decreasing" if end < start else "stable"

        visualization = {
            "chart_type": "line",
            "title": f"{metric} over time",
            "x_label": granularity.title(),
            "y_label": metric,
            "x": x,
            "y": y,
            "metric": metric,
            "time_column": date_column,
            "granularity": granularity,
            "value_format": "number",
            "summary": {
                "direction": direction,
                "start": start,
                "end": end,
                "change_percent": change_percent,
            },
        }
        return {"insights": insights, "visualization": visualization}

    def run(self, df, entities=None):
        """Keep the historical text-only interface backward compatible."""
        analysis = self.analyze(df, entities)
        if analysis.get("error"):
            return {"error": analysis["error"]}
        return analysis["insights"]

    @staticmethod
    def _aggregation(question):
        lowered = question.lower()
        if re.search(r"\b(average|avg|mean)\b", lowered):
            return "mean"
        if re.search(r"\b(count|number of)\b", lowered):
            return "count"
        return "sum"

    @staticmethod
    def _granularity(dates, question):
        lowered = question.lower()
        if re.search(r"\b(month|monthly)\b", lowered):
            return "month"
        if re.search(r"\b(quarter|quarterly)\b", lowered):
            return "quarter"
        if re.search(r"\b(week|weekly)\b", lowered):
            return "week"
        if re.search(r"\b(day|daily)\b", lowered):
            return "day"
        if re.search(r"\b(year|yearly|annually|annual)\b", lowered):
            return "year"
        return "year" if dates.dt.year.nunique() > 1 else "month"

    def _build_trend_insights(self, grouped, metric_col, period_name):
        grouped = grouped.dropna()
        if grouped.empty:
            return ["Insufficient data for trend analysis."]

        best_period, worst_period = grouped.idxmax(), grouped.idxmin()
        insights = [
            f"Best {period_name.lower()}: {best_period} with {metric_col} of {round(float(grouped.max()), 2)}.",
            f"Worst {period_name.lower()}: {worst_period} with {metric_col} of {round(float(grouped.min()), 2)}.",
        ]
        first_value, last_value = float(grouped.iloc[0]), float(grouped.iloc[-1])
        if first_value != 0:
            growth = (last_value - first_value) / abs(first_value) * 100
            insights.append(
                f"{metric_col} growth from the first to the last {period_name.lower()} was {round(growth, 2)}%."
            )
        return insights

    @staticmethod
    def _finite_or_none(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None
