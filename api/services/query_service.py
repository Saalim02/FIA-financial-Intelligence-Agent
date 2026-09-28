from api.services.dataset_service import DatasetService


class QueryService:
    def __init__(self, dataset_service: DatasetService):
        self.dataset_service = dataset_service

    def answer_query(self, dataset_id: str, question: str) -> dict:
        df = self.dataset_service.get_dataset(dataset_id)

        question_lower = question.lower()

        result = {
            "total_revenue": None,
            "total_profit": None,
            "profit_margin": None,
        }

        if "revenue" in question_lower:
            if "revenue" not in df.columns:
                raise ValueError("Revenue column not found.")

            result["total_revenue"] = float(df["revenue"].sum())

        if "profit" in question_lower:
            if "profit" not in df.columns:
                raise ValueError("Profit column not found.")

            result["total_profit"] = float(df["profit"].sum())

        if "margin" in question_lower:
            if "revenue" not in df.columns:
                raise ValueError("Revenue column not found.")

            if "profit" not in df.columns:
                raise ValueError("Profit column not found.")

            total_revenue = df["revenue"].sum()
            total_profit = df["profit"].sum()

            if total_revenue == 0:
                raise ValueError(
                    "Cannot calculate profit margin because revenue is zero."
                )

            result["profit_margin"] = round(
                float((total_profit / total_revenue) * 100),
                2,
            )

        if all(value is None for value in result.values()):
            raise ValueError(
                "The query is not supported yet. "
                "Try asking about revenue, profit, or profit margin."
            )

        return result