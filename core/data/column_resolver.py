from __future__ import annotations

import re
from typing import Any

import pandas as pd

from core.data.schema_profiler import SchemaProfiler


class ColumnResolver:
    """
    Resolve user query terms to actual dataset columns.

    Dataset-agnostic:
    - Does not assume specific dataset column names.
    - Uses the schema profiler to identify usable columns.
    - Uses semantic concepts to understand user intent.
    - Prefers explicit column mentions over semantic matches.
    - Avoids identifier columns such as IDs, codes, keys, etc.
    """

    # Generic semantic concepts.
    # These represent business/analytical concepts,
    # not specific dataset column names.
    CONCEPT_TERMS = {
        "revenue": {
            "revenue",
            "sales",
            "income",
            "turnover",
            "earnings",
        },
        "profit": {
            "profit",
            "net profit",
            "operating profit",
        },
        "cost": {
            "cost",
            "expense",
            "expenses",
            "spending",
        },
        "quantity": {
            "quantity",
            "units",
            "volume",
            "count",
        },
        "discount": {
            "discount",
            "discount rate",
        },
        "price": {
            "price",
            "unit price",
            "selling price",
        },
        "margin": {
            "margin",
            "profit margin",
        },
    }

    # Generic words that usually indicate an identifier.
    IDENTIFIER_TERMS = {
        "id",
        "identifier",
        "code",
        "key",
        "uuid",
        "guid",
    }

    @staticmethod
    def normalize(text: Any) -> str:
        """
        Normalize text for reliable comparison.

        Examples:
            "Unit_Price" -> "unit price"
            "Order-ID"   -> "order id"
            " Sales "    -> "sales"
        """
        text = str(text).strip().lower()

        # Replace underscores and hyphens with spaces.
        text = re.sub(r"[_\-]+", " ", text)

        # Remove repeated whitespace.
        text = re.sub(r"\s+", " ", text)

        return text

    @classmethod
    def _is_identifier_column(cls, column: str) -> bool:
        """
        Determine whether a column looks like an identifier.

        This is intentionally generic and does not depend
        on a specific dataset.
        """
        normalized = cls.normalize(column)

        words = set(normalized.split())

        # Explicit identifier words.
        if words & cls.IDENTIFIER_TERMS:
            return True

        # Strong suffix/prefix signals.
        if normalized.endswith(" id"):
            return True

        if normalized.startswith("id "):
            return True

        # Common identifier naming patterns.
        if normalized.endswith(" code"):
            return True

        if normalized.endswith(" key"):
            return True

        return False

    @classmethod
    def _column_score(
        cls,
        column: str,
        question: str,
        concept_name: str,
        concept_terms: set[str],
    ) -> int:
        """
        Score how strongly a column matches a requested concept.

        Higher score = stronger match.
        """

        normalized_column = cls.normalize(column)
        normalized_question = cls.normalize(question)

        score = 0

        # ---------------------------------------------------------
        # 1. Exact column name mentioned in the question.
        # ---------------------------------------------------------
        if normalized_column in normalized_question:
            score += 100

        # ---------------------------------------------------------
        # 2. Exact concept match.
        # Example:
        # question = "what is profit?"
        # column   = "Profit"
        # ---------------------------------------------------------
        if normalized_column == concept_name:
            score += 90

        # ---------------------------------------------------------
        # 3. Exact semantic term match.
        # Example:
        # column = "Sales"
        # concept term = "sales"
        # ---------------------------------------------------------
        for term in concept_terms:
            normalized_term = cls.normalize(term)

            if normalized_column == normalized_term:
                score += 80

            # Multi-word semantic term contained in column.
            elif (
                len(normalized_term.split()) > 1
                and normalized_term in normalized_column
            ):
                score += 50

        # ---------------------------------------------------------
        # 4. Word overlap.
        # ---------------------------------------------------------
        column_words = set(normalized_column.split())
        question_words = set(normalized_question.split())

        overlap = column_words & question_words

        score += len(overlap) * 20

        # ---------------------------------------------------------
        # 5. Concept term explicitly mentioned in question.
        #
        # Important:
        # This score is only applied to columns that already
        # have some relationship with the concept.
        # It prevents unrelated numeric columns from winning.
        # ---------------------------------------------------------
        concept_mentioned = False

        for term in concept_terms:
            normalized_term = cls.normalize(term)

            if normalized_term in normalized_question:
                concept_mentioned = True
                break

        if concept_mentioned:
            for term in concept_terms:
                normalized_term = cls.normalize(term)

                if (
                    normalized_term == normalized_column
                    or normalized_term in normalized_column
                ):
                    score += 40

        return score

    @classmethod
    def resolve_numeric_column(
        cls,
        df: pd.DataFrame,
        question: str,
    ) -> str | None:
        """
        Resolve the most appropriate numeric column for a question.

        The resolver:
        1. Uses the schema profiler.
        2. Removes identifier columns.
        3. Detects the requested semantic concept.
        4. Scores only relevant numeric columns.
        5. Returns None when no meaningful match exists.
        """

        profile = SchemaProfiler.profile(df)

        numeric_columns = profile.get(
            "useful_numeric_columns",
            [],
        )

        if not numeric_columns:
            return None

        # Never use identifier-like columns as analytical metrics.
        candidate_columns = [
            column
            for column in numeric_columns
            if not cls._is_identifier_column(column)
        ]

        if not candidate_columns:
            return None

        normalized_question = cls.normalize(question)

        best_column: str | None = None
        best_score = 0

        # ---------------------------------------------------------
        # Evaluate each semantic concept independently.
        # ---------------------------------------------------------
        for concept_name, concept_terms in cls.CONCEPT_TERMS.items():

            # Determine whether this concept appears in the question.
            concept_requested = False

            for term in concept_terms:
                normalized_term = cls.normalize(term)

                if normalized_term in normalized_question:
                    concept_requested = True
                    break

            # The concept itself may be the exact requested word.
            if concept_name in normalized_question:
                concept_requested = True

            # Do not evaluate unrelated concepts.
            if not concept_requested:
                continue

            for column in candidate_columns:

                score = cls._column_score(
                    column=column,
                    question=question,
                    concept_name=concept_name,
                    concept_terms=concept_terms,
                )

                if score > best_score:
                    best_score = score
                    best_column = column

        # No meaningful semantic match.
        if best_score <= 0:
            return None

        return best_column

    @classmethod
    def resolve_explicit_columns(
        cls,
        df: pd.DataFrame,
        question: str,
    ) -> list[str]:
        """
        Find dataset columns explicitly mentioned in the question.

        Example:
            Question:
                "What is the total Sales and Profit?"

            Result:
                ["Sales", "Profit"]
        """

        normalized_question = cls.normalize(question)

        matches: list[str] = []

        for column in df.columns:
            normalized_column = cls.normalize(column)

            if normalized_column in normalized_question:
                matches.append(column)

        return matches

    @classmethod
    def resolve(
        cls,
        df: pd.DataFrame,
        question: str,
    ) -> dict:
        """
        Resolve all useful column information from a question.
        """

        explicit_columns = cls.resolve_explicit_columns(
            df,
            question,
        )

        numeric_column = cls.resolve_numeric_column(
            df,
            question,
        )

        return {
            "explicit_columns": explicit_columns,
            "numeric_column": numeric_column,
        }