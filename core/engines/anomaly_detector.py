import numpy as np
from core.utils.dataframe_utils import is_numeric_column
from core.data.schema_profiler import SchemaProfiler


class AnomalyDetector:

    def run(self, df, entities=None):
        if df.empty:
            return {"error": "Empty dataframe."}

        if not entities:
            return {"error": "No metric provided for anomaly detection."}

        insights = []
        valid_metric_found = False

        for column in entities:
            if column not in df.columns:
                continue

            if not is_numeric_column(df, column):
                continue

            series = df[column].dropna()

            if series.empty:
                continue

            valid_metric_found = True

            outliers = self._iqr_outliers(series)

            if not outliers.empty:
                insights.append(
                    f"{len(outliers)} abnormal records detected "
                    f"in '{column}' (IQR method)."
                )

            extreme_low = series[series < series.quantile(0.01)]

            if not extreme_low.empty:
                insights.append(
                    f"{len(extreme_low)} extreme low values detected "
                    f"in '{column}' (bottom 1%)."
                )

        if not valid_metric_found:
            return {"error": "No valid numeric metric found for anomaly detection."}

        if not insights:
            return ["No anomalies detected."]

        return insights
                    

    def _iqr_outliers(self, series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            return series.iloc[0:0]

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return series[
            (series < lower_bound) |
            (series > upper_bound)
        ]
