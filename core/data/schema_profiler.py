from __future__ import annotations

import re
from typing import Any

import pandas as pd


class SchemaProfiler:
    """
    Dataset-agnostic schema profiler.

    Detects column characteristics from the actual dataframe.

    Roles:
        - measure
        - identifier
        - datetime
        - categorical
        - text
        - constant
        - unknown

    The profiler does NOT contain dataset-specific mappings such as
    Sales -> revenue or Profit -> profit.
    """

    # Generic identifier concepts.
    #
    # These are not dataset-specific mappings. They are generic
    # linguistic signals commonly associated with identifiers/codes.
    IDENTIFIER_TOKENS = {
        "id",
        "identifier",
        "code",
        "key",
        "uuid",
        "guid",
    }

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_name(value: Any) -> str:
        """
        Normalize names for generic semantic inspection.
        """
        value = str(value).strip().lower()

        value = re.sub(r"[_\-]+", " ", value)
        value = re.sub(r"\s+", " ", value)

        return value

    # ------------------------------------------------------------------
    # BASIC STATISTICS
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_ratio(series: pd.Series) -> float:
        """
        Ratio of unique non-null values.
        """
        non_null = series.dropna()

        if non_null.empty:
            return 0.0

        return non_null.nunique() / len(non_null)

    @staticmethod
    def _is_constant(series: pd.Series) -> bool:
        """
        Detect constant columns.
        """
        return series.nunique(dropna=True) <= 1

    # ------------------------------------------------------------------
    # GENERIC IDENTIFIER NAME SIGNAL
    # ------------------------------------------------------------------

    @classmethod
    def _has_identifier_token(cls, column: str) -> bool:
        """
        Detect generic identifier semantics from a column name.

        Examples that may be detected:

            customer id
            transaction id
            product code
            account key
            uuid

        This intentionally does NOT contain dataset-specific names.
        """

        normalized = cls.normalize_name(column)

        tokens = set(normalized.split())

        return bool(tokens & cls.IDENTIFIER_TOKENS)

    # ------------------------------------------------------------------
    # IDENTIFIER DETECTION
    # ------------------------------------------------------------------

    @classmethod
    def _is_id_like(
        cls,
        series: pd.Series,
        column: str | None = None,
    ) -> bool:
        """
        Detect identifier/code-like columns using multiple signals.

        Signals include:
            - generic identifier terminology
            - sequential integer patterns
            - extremely high cardinality
            - large numeric ranges
            - high-cardinality strings

        The function intentionally avoids dataset-specific mappings.
        """

        if series.empty:
            return False

        non_null = series.dropna()

        if non_null.empty:
            return False

        row_count = len(non_null)

        unique_count = non_null.nunique()

        unique_ratio = unique_count / row_count

        # --------------------------------------------------------------
        # GENERIC NAME SIGNAL
        # --------------------------------------------------------------

        if column is not None and cls._has_identifier_token(column):

            # A generic ID/code column with reasonable cardinality
            # is very likely an identifier.
            if unique_ratio >= 0.20:
                return True

        # --------------------------------------------------------------
        # SMALL DATASETS
        # --------------------------------------------------------------

        # Statistical ID detection becomes unreliable on tiny datasets.
        if row_count < 20:
            return False

        # --------------------------------------------------------------
        # INTEGER COLUMNS
        # --------------------------------------------------------------

        if pd.api.types.is_integer_dtype(series):

            # Sequential integer identifier.
            if (
                unique_ratio >= 0.98
                and series.is_monotonic_increasing
            ):
                return True

            # Almost completely unique integer column.
            if unique_ratio >= 0.995:
                return True

            # High-cardinality integer code.
            if unique_ratio >= 0.80:

                value_range = non_null.max() - non_null.min()

                if value_range > row_count * 2:
                    return True

        # --------------------------------------------------------------
        # STRING / OBJECT COLUMNS
        # --------------------------------------------------------------

        if (
            pd.api.types.is_string_dtype(series)
            or series.dtype == object
        ):

            # Extremely high-cardinality text is often an ID/code.
            if unique_ratio >= 0.98:
                return True

        # --------------------------------------------------------------
        # FLOAT COLUMNS
        # --------------------------------------------------------------

        # Never classify floating-point financial measurements as IDs
        # solely because they have many unique values.
        if pd.api.types.is_float_dtype(series):
            return False

        return False

    # ------------------------------------------------------------------
    # DATETIME DETECTION
    # ------------------------------------------------------------------

    @classmethod
    def _detect_datetime_columns(
        cls,
        df: pd.DataFrame,
    ) -> list[str]:

        date_columns: list[str] = []

        for column in df.columns:

            series = df[column]

            # Already datetime.
            if pd.api.types.is_datetime64_any_dtype(series):
                date_columns.append(column)
                continue

            # Numeric values are not automatically dates.
            if pd.api.types.is_numeric_dtype(series):
                continue

            try:
                converted = pd.to_datetime(
                    series,
                    errors="coerce",
                    format="mixed",
                )
            except (TypeError, ValueError):
                continue

            valid_ratio = converted.notna().mean()

            if (
                valid_ratio >= 0.80
                and converted.nunique() > 1
            ):
                date_columns.append(column)

        return date_columns

    # ------------------------------------------------------------------
    # COLUMN ROLE
    # ------------------------------------------------------------------

    @classmethod
    def _detect_column_role(
        cls,
        df: pd.DataFrame,
        column: str,
        datetime_columns: set[str],
    ) -> str:

        series = df[column]

        # --------------------------------------------------------------
        # DATETIME
        # --------------------------------------------------------------

        if column in datetime_columns:
            return "datetime"

        # --------------------------------------------------------------
        # NUMERIC
        # --------------------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            if cls._is_constant(series):
                return "constant"

            if cls._is_id_like(series, column):
                return "identifier"

            return "measure"

        # --------------------------------------------------------------
        # BOOLEAN
        # --------------------------------------------------------------

        if pd.api.types.is_bool_dtype(series):
            return "categorical"

        # --------------------------------------------------------------
        # STRING / OBJECT
        # --------------------------------------------------------------

        if (
            pd.api.types.is_string_dtype(series)
            or series.dtype == object
        ):

            non_null = series.dropna()

            if non_null.empty:
                return "unknown"

            unique_count = non_null.nunique()

            unique_ratio = unique_count / len(non_null)

            # Generic ID/code semantics should take priority.
            if cls._has_identifier_token(column):

                if unique_ratio >= 0.20:
                    return "identifier"

            # Extremely high cardinality.
            if unique_ratio >= 0.98:
                return "identifier"

            # Repeated values -> category.
            if unique_count <= min(
                100,
                max(
                    10,
                    int(len(non_null) * 0.20),
                ),
            ):
                return "categorical"

            # Otherwise descriptive/free-form text.
            return "text"

        return "unknown"

    # ------------------------------------------------------------------
    # MAIN PROFILE
    # ------------------------------------------------------------------

    @classmethod
    def profile(
        cls,
        df: pd.DataFrame,
    ) -> dict[str, Any]:

        # --------------------------------------------------------------
        # NONE
        # --------------------------------------------------------------

        if df is None:

            return {
                "columns": [],
                "numeric_columns": [],
                "useful_numeric_columns": [],
                "measure_columns": [],
                "datetime_columns": [],
                "categorical_columns": [],
                "text_columns": [],
                "identifier_columns": [],
                "constant_columns": [],
                "column_roles": {},
                "column_profiles": {},
                "row_count": 0,
                "column_count": 0,
            }

        # --------------------------------------------------------------
        # EMPTY
        # --------------------------------------------------------------

        if df.empty:

            return {
                "columns": list(df.columns),
                "numeric_columns": [],
                "useful_numeric_columns": [],
                "measure_columns": [],
                "datetime_columns": [],
                "categorical_columns": [],
                "text_columns": [],
                "identifier_columns": [],
                "constant_columns": [],
                "column_roles": {},
                "column_profiles": {},
                "row_count": 0,
                "column_count": len(df.columns),
            }

        # --------------------------------------------------------------
        # DATETIME DETECTION
        # --------------------------------------------------------------

        datetime_columns = cls._detect_datetime_columns(df)

        datetime_set = set(datetime_columns)

        # --------------------------------------------------------------
        # CONTAINERS
        # --------------------------------------------------------------

        numeric_columns: list[str] = []

        measure_columns: list[str] = []

        useful_numeric_columns: list[str] = []

        identifier_columns: list[str] = []

        categorical_columns: list[str] = []

        text_columns: list[str] = []

        constant_columns: list[str] = []

        column_roles: dict[str, str] = {}

        column_profiles: dict[str, dict[str, Any]] = {}

        # --------------------------------------------------------------
        # PROFILE EACH COLUMN
        # --------------------------------------------------------------

        for column in df.columns:

            series = df[column]

            role = cls._detect_column_role(
                df,
                column,
                datetime_set,
            )

            column_roles[column] = role

            non_null = series.dropna()

            if non_null.empty:

                unique_count = 0
                unique_ratio = 0.0

            else:

                unique_count = non_null.nunique()

                unique_ratio = (
                    unique_count / len(non_null)
                )

            # ----------------------------------------------------------
            # METADATA
            # ----------------------------------------------------------

            column_profiles[column] = {
                "dtype": str(series.dtype),
                "role": role,
                "row_count": len(series),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "unique_count": int(unique_count),
                "unique_ratio": round(
                    unique_ratio,
                    6,
                ),
            }

            # ----------------------------------------------------------
            # NUMERIC
            # ----------------------------------------------------------

            if pd.api.types.is_numeric_dtype(series):

                numeric_columns.append(column)

            # ----------------------------------------------------------
            # MEASURE
            # ----------------------------------------------------------

            if role == "measure":

                measure_columns.append(column)

                useful_numeric_columns.append(column)

            # ----------------------------------------------------------
            # IDENTIFIER
            # ----------------------------------------------------------

            if role == "identifier":

                identifier_columns.append(column)

            # ----------------------------------------------------------
            # DATETIME
            # ----------------------------------------------------------

            if role == "datetime":

                # Already stored in datetime_columns.
                pass

            # ----------------------------------------------------------
            # CATEGORICAL
            # ----------------------------------------------------------

            if role == "categorical":

                categorical_columns.append(column)

            # ----------------------------------------------------------
            # TEXT
            # ----------------------------------------------------------

            if role == "text":

                text_columns.append(column)

            # ----------------------------------------------------------
            # CONSTANT
            # ----------------------------------------------------------

            if role == "constant":

                constant_columns.append(column)

        # --------------------------------------------------------------
        # RETURN PROFILE
        # --------------------------------------------------------------

        return {
            "columns": list(df.columns),

            "numeric_columns": numeric_columns,

            "useful_numeric_columns": useful_numeric_columns,

            "measure_columns": measure_columns,

            "datetime_columns": datetime_columns,

            "categorical_columns": categorical_columns,

            "text_columns": text_columns,

            "identifier_columns": identifier_columns,

            "constant_columns": constant_columns,

            "column_roles": column_roles,

            "column_profiles": column_profiles,

            "row_count": len(df),

            "column_count": len(df.columns),
        }

    # ------------------------------------------------------------------
    # CONVENIENCE METHODS
    # ------------------------------------------------------------------

    @classmethod
    def useful_numeric_columns(
        cls,
        df: pd.DataFrame,
    ) -> list[str]:

        return cls.profile(df)[
            "useful_numeric_columns"
        ]

    @classmethod
    def measure_columns(
        cls,
        df: pd.DataFrame,
    ) -> list[str]:

        return cls.profile(df)[
            "measure_columns"
        ]

    @classmethod
    def datetime_columns(
        cls,
        df: pd.DataFrame,
    ) -> list[str]:

        return cls.profile(df)[
            "datetime_columns"
        ]

    @classmethod
    def categorical_columns(
        cls,
        df: pd.DataFrame,
    ) -> list[str]:

        return cls.profile(df)[
            "categorical_columns"
        ]

    @classmethod
    def identifier_columns(
        cls,
        df: pd.DataFrame,
    ) -> list[str]:

        return cls.profile(df)[
            "identifier_columns"
        ]

    @classmethod
    def column_roles(
        cls,
        df: pd.DataFrame,
    ) -> dict[str, str]:

        return cls.profile(df)[
            "column_roles"
        ]

    @classmethod
    def column_profiles(
        cls,
        df: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:

        return cls.profile(df)[
            "column_profiles"
        ]