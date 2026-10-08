import pandas as pd

from core.data.schema_profiler import SchemaProfiler


class TimeAnalyzer:

    def run(self, df, entities=None):
        if df is None or df.empty:
            return {"error": "Empty dataframe."}

        profile = SchemaProfiler.profile(df)

        # Use requested entities only when they actually exist and are numeric.
        metric_columns = []

        if entities:
            for column in entities:
                if (
                    column in df.columns
                    and column in profile["useful_numeric_columns"]
                ):
                    metric_columns.append(column)

        # Dataset-agnostic fallback.
        if not metric_columns:
            metric_columns = profile["useful_numeric_columns"]

        if not metric_columns:
            return {
                "error": "No useful numeric columns available for trend analysis."
            }

        date_columns = profile["datetime_columns"]

        if not date_columns:
            return {
                "error": "No usable date/time column detected for trend analysis."
            }

        date_column = date_columns[0]

        temp_df = df.copy()

        temp_df[date_column] = pd.to_datetime(
            temp_df[date_column],
            errors="coerce",
            format="mixed",
        )

        temp_df = temp_df.dropna(subset=[date_column])

        if temp_df.empty:
            return {"error": "Date column could not be parsed."}

        insights = []

        for metric_col in metric_columns:
            series = temp_df[metric_col]

            if series.dropna().empty:
                continue

            years = temp_df[date_column].dt.year.nunique()

            if years > 1:
                temp_df["_analysis_period"] = temp_df[date_column].dt.year
                period_name = "Year"
            else:
                temp_df["_analysis_period"] = (
                    temp_df[date_column].dt.to_period("M")
                )
                period_name = "Month"

            grouped = temp_df.groupby("_analysis_period")[metric_col].sum()

            if grouped.empty:
                continue

            metric_insights = self._build_trend_insights(
                grouped,
                metric_col,
                period_name,
            )

            insights.extend(metric_insights)

        if not insights:
            return {"error": "Insufficient data for trend analysis."}

        return insights

    def _build_trend_insights(self, grouped, metric_col, period_name):
        insights = []

        if grouped.empty:
            return ["Insufficient data for trend analysis."]

        best_period = grouped.idxmax()
        worst_period = grouped.idxmin()

        best_value = grouped.max()
        worst_value = grouped.min()

        insights.append(
            f"Best {period_name.lower()}: "
            f"{best_period} with {metric_col} of {round(best_value, 2)}."
        )

        insights.append(
            f"Worst {period_name.lower()}: "
            f"{worst_period} with {metric_col} of {round(worst_value, 2)}."
        )

        first_value = grouped.iloc[0]
        last_value = grouped.iloc[-1]

        if first_value != 0:
            growth = ((last_value - first_value) / abs(first_value)) * 100

            insights.append(
                f"{metric_col} growth from the first to the last "
                f"{period_name.lower()} was {round(growth, 2)}%."
            )

        return insights