import pandas as pd

from core.utils.dataframe_utils import is_numeric_column


class TimeAnalyzer:

    def run(self, df, entities=None):

        if df.empty:
            return {"error": "Empty dataframe."}

        # Identify metric column from entities
        metric_col = None

        if entities:
            for col in entities:
                if is_numeric_column(df, col):
                    metric_col = col
                    break

        if not metric_col:
            return {"error": "No numeric metric provided for trend analysis."}

        # Detect date column
        date_column = self._detect_date_column(df)

        if not date_column:
            return {"error": "No date column detected."}

        temp_df = df.copy()

        temp_df[date_column] = pd.to_datetime(
            temp_df[date_column],
            errors="coerce",
        )

        # Remove rows where date could not be parsed
        temp_df = temp_df.dropna(subset=[date_column])

        if temp_df.empty:
            return {"error": "Date column could not be parsed."}

        # -------------------------------------------------
        # Multiple years -> yearly trend
        # -------------------------------------------------

        years = temp_df[date_column].dt.year.nunique()

        if years > 1:

            temp_df["Year"] = temp_df[date_column].dt.year

            grouped = temp_df.groupby("Year")[metric_col].sum()

            if grouped.empty:
                return {"error": "Insufficient data for trend analysis."}

            return self._build_trend_insights(
                grouped,
                metric_col,
                period_name="Year",
            )

        # -------------------------------------------------
        # Single year -> monthly trend
        # -------------------------------------------------

        temp_df["Month"] = temp_df[date_column].dt.to_period("M")

        grouped = temp_df.groupby("Month")[metric_col].sum()

        if grouped.empty:
            return {"error": "Insufficient data for trend analysis."}

        return self._build_trend_insights(
            grouped,
            metric_col,
            period_name="Month",
        )

    def _build_trend_insights(
        self,
        grouped,
        metric_col,
        period_name,
    ):
        insights = []

        # Best period
        best_period = grouped.idxmax()

        # Worst period
        worst_period = grouped.idxmin()

        insights.append(
            f"Best {metric_col} {period_name}: "
            f"{best_period} ({round(grouped.loc[best_period], 2)})"
        )

        insights.append(
            f"Worst {metric_col} {period_name}: "
            f"{worst_period} ({round(grouped.loc[worst_period], 2)})"
        )

        # Growth calculation
        if len(grouped) > 1:

            first_period = grouped.index.min()
            last_period = grouped.index.max()

            first_value = grouped.loc[first_period]
            last_value = grouped.loc[last_period]

            if first_value != 0:

                growth = (
                    (last_value - first_value)
                    / first_value
                    * 100
                )

                insights.append(
                    f"{metric_col} Growth from "
                    f"{first_period} to {last_period}: "
                    f"{round(growth, 2)}%"
                )

        return insights

    def _detect_date_column(self, df):

        for col in df.columns:

            if "date" in col.lower():
                return col

        return None
