import pandas as pd
import re


# ==================================================
# 1. CLEAN COLUMN NAMES
# ==================================================

def clean_column_name(column):
    """
    Convert column names into a consistent format.

    Examples:
        Customer Name -> customer_name
        Order-Date    -> order_date
        Revenue ($)   -> revenue
    """

    column = str(column).strip().lower()

    column = re.sub(
        r"[^a-z0-9]+",
        "_",
        column
    )

    column = column.strip("_")

    return column


# ==================================================
# 2. CLEAN TEXT VALUES
# ==================================================

def clean_text_values(df, cleaning_log):
    """
    Clean text columns by:
    - removing leading/trailing spaces
    - replacing multiple spaces with one
    - removing obvious empty strings
    """

    df = df.copy()

    total_changed = 0

    for column in df.columns:

        if not (
            df[column].dtype == "object"
            or pd.api.types.is_string_dtype(df[column])
        ):
            continue

        original = df[column].copy()

        cleaned = (
            df[column]
            .astype("string")
            .str.strip()
            .str.replace(
                r"\s+",
                " ",
                regex=True
            )
        )

        # Convert empty strings to missing values
        cleaned = cleaned.replace(
            "",
            pd.NA
        )

        changed = (
            original.fillna("__MISSING__").astype(str)
            != cleaned.fillna("__MISSING__").astype(str)
        ).sum()

        if changed > 0:

            df[column] = cleaned

            total_changed += changed

            cleaning_log.append(
                f"Cleaned {changed:,} text value(s) "
                f"in '{column}'."
            )

    return df


# ==================================================
# 3. GENERIC CATEGORICAL STANDARDIZATION
# ==================================================

def standardize_categorical_values(df, cleaning_log):
    """
    Automatically merge categorical values that differ
    only because of capitalization or whitespace.

    Examples:

        Furniture
        furniture
        FURNITURE
        Furniture

    become:

        Furniture
        Furniture
        Furniture
        Furniture

    The most frequently occurring spelling is used as
    the canonical representation.
    """

    df = df.copy()

    for column in df.columns:

        # Only process text/categorical columns
        if not (
            df[column].dtype == "object"
            or pd.api.types.is_string_dtype(df[column])
        ):
            continue

        non_null = df[column].dropna()

        if non_null.empty:
            continue

        values = non_null.astype(str)

        # Create normalized keys
        normalized_keys = (
            values
            .str.strip()
            .str.lower()
            .str.replace(
                r"\s+",
                " ",
                regex=True
            )
        )

        temp = pd.DataFrame(
            {
                "original": values,
                "normalized": normalized_keys
            }
        )

        # Find groups where multiple spellings
        # represent the same normalized value
        grouped = (
            temp.groupby("normalized")["original"]
            .agg(
                [
                    "count",
                    lambda x: x.nunique()
                ]
            )
        )

        grouped.columns = [
            "count",
            "unique_spellings"
        ]

        changed_groups = grouped[
            grouped["unique_spellings"] > 1
        ]

        if changed_groups.empty:
            continue

        # ------------------------------------------
        # Build canonical mapping
        # ------------------------------------------

        canonical_mapping = {}

        for normalized_value in changed_groups.index:

            matching_rows = temp[
                temp["normalized"] == normalized_value
            ]

            # Choose the most common spelling
            canonical_value = (
                matching_rows["original"]
                .value_counts()
                .idxmax()
            )

            canonical_mapping[
                normalized_value
            ] = canonical_value

        # ------------------------------------------
        # Apply mapping
        # ------------------------------------------

        normalized_column = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(
                r"\s+",
                " ",
                regex=True
            )
        )

        changed_count = 0

        for normalized_value, canonical_value in canonical_mapping.items():

            mask = (
                normalized_column
                == normalized_value
            )

            original_values = (
                df.loc[mask, column]
                .astype("string")
            )

            changed_count += (
                original_values
                != canonical_value
            ).sum()

            df.loc[
                mask,
                column
            ] = canonical_value

        if changed_count > 0:

            cleaning_log.append(
                f"Standardized {changed_count:,} "
                f"inconsistent categorical value(s) "
                f"in '{column}' based on capitalization "
                f"and whitespace."
            )

    return df


# ==================================================
# 4. BUSINESS ALIAS STANDARDIZATION
# ==================================================

def standardize_business_aliases(df, cleaning_log):
    """
    Apply known business aliases.

    These are intentionally explicit so that the system
    does not incorrectly merge unrelated categories.
    """

    df = df.copy()

    # ------------------------------------------
    # REGION
    # ------------------------------------------

    region_aliases = {

        "blr": "Bangalore",
        "bengaluru": "Bangalore",

        "bombay": "Mumbai",

        "madras": "Chennai",

        "calcutta": "Kolkata"
    }

    # ------------------------------------------
    # CATEGORY
    # ------------------------------------------

    category_aliases = {

        "electronic": "Electronics",

        "clothes": "Clothing",
        "apparel": "Clothing",

        "home and kitchen": "Home & Kitchen",
        "home kitchen": "Home & Kitchen",

        "beauty and personal care":
            "Beauty",

        "sports and fitness":
            "Sports",

        "groceries":
            "Grocery"
    }

    # ------------------------------------------
    # PAYMENT METHOD
    # ------------------------------------------

    payment_aliases = {

        "cc": "Credit Card",
        "creditcard": "Credit Card",
        "credit_card": "Credit Card",

        "dc": "Debit Card",
        "debitcard": "Debit Card",
        "debit_card": "Debit Card",

        "upi payment": "UPI",

        "cod": "Cash on Delivery",

        "netbanking": "Net Banking",
        "net_banking": "Net Banking"
    }

    # ------------------------------------------
    # CUSTOMER SEGMENT
    # ------------------------------------------

    segment_aliases = {

        "consumers": "Consumer",

        "corporates": "Corporate",

        "smallbusiness": "Small Business",
        "small_business": "Small Business"
    }

    mappings = {

        "region": region_aliases,

        "category": category_aliases,

        "payment_method":
            payment_aliases,

        "customer_segment":
            segment_aliases,

        "segment":
            segment_aliases
    }

    # ------------------------------------------
    # APPLY ALIASES
    # ------------------------------------------

    for column in df.columns:

        if column not in mappings:
            continue

        mapping = mappings[column]

        normalized_column = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(
                r"\s+",
                " ",
                regex=True
            )
        )

        changed_count = 0

        for alias, canonical in mapping.items():

            mask = (
                normalized_column
                == alias
            )

            if mask.any():

                changed_count += mask.sum()

                df.loc[
                    mask,
                    column
                ] = canonical

        if changed_count > 0:

            cleaning_log.append(
                f"Mapped {changed_count:,} "
                f"business alias value(s) "
                f"in '{column}'."
            )

    return df


# ==================================================
# 5. DATE CONVERSION
# ==================================================

def try_convert_dates(df, cleaning_log):

    df = df.copy()

    for column in df.columns:

        column_name = column.lower()

        if not any(
            keyword in column_name
            for keyword in [
                "date",
                "time",
                "timestamp"
            ]
        ):
            continue

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            continue

        if not (
            df[column].dtype == "object"
            or pd.api.types.is_string_dtype(df[column])
        ):
            continue

        original_non_null = (
            df[column].notna().sum()
        )

        if original_non_null == 0:
            continue

        try:

            converted = pd.to_datetime(
                df[column],
                errors="coerce",
                format="mixed"
            )

            converted_non_null = (
                converted.notna().sum()
            )

            conversion_rate = (
                converted_non_null
                / original_non_null
            )

            if conversion_rate >= 0.80:

                df[column] = converted

                cleaning_log.append(
                    f"Converted '{column}' "
                    f"to datetime."
                )

        except Exception:
            pass

    return df


# ==================================================
# 6. NUMERIC CONVERSION
# ==================================================

def try_convert_numeric(df, cleaning_log):

    df = df.copy()

    numeric_keywords = [

        "price",
        "revenue",
        "sales",
        "profit",
        "cost",
        "amount",
        "quantity",
        "discount",
        "rating"
    ]

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        column_name = column.lower()

        if not any(
            keyword in column_name
            for keyword in numeric_keywords
        ):
            continue

        original_non_null = (
            df[column].notna().sum()
        )

        if original_non_null == 0:
            continue

        try:

            cleaned_values = (
                df[column]
                .astype(str)
                .str.replace(
                    ",",
                    "",
                    regex=False
                )
                .str.replace(
                    "₹",
                    "",
                    regex=False
                )
                .str.replace(
                    "$",
                    "",
                    regex=False
                )
                .str.replace(
                    "€",
                    "",
                    regex=False
                )
                .str.replace(
                    "£",
                    "",
                    regex=False
                )
                .str.strip()
            )

            converted = pd.to_numeric(
                cleaned_values,
                errors="coerce"
            )

            converted_non_null = (
                converted.notna().sum()
            )

            conversion_rate = (
                converted_non_null
                / original_non_null
            )

            if conversion_rate >= 0.80:

                df[column] = converted

                cleaning_log.append(
                    f"Converted '{column}' "
                    f"to numeric."
                )

        except Exception:
            pass

    return df


# ==================================================
# 7. REMOVE DUPLICATES
# ==================================================

def remove_duplicates(df, cleaning_log):

    df = df.copy()

    duplicate_count = (
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        df = (
            df
            .drop_duplicates()
            .reset_index(drop=True)
        )

        cleaning_log.append(
            f"Removed {duplicate_count:,} "
            f"duplicate row(s)."
        )

    return df, duplicate_count


# ==================================================
# 8. MAIN CLEANING PIPELINE
# ==================================================

def clean_dataframe(df):
    """
    Complete automatic data-cleaning pipeline.

    Order:

    1. Clean column names
    2. Clean text
    3. Convert dates
    4. Convert numeric columns
    5. Standardize categorical values
    6. Apply business aliases
    7. Remove duplicates AFTER standardization

    Returns:

        cleaned_df
        cleaning_log
        rows_removed
    """

    cleaned_df = df.copy()

    cleaning_log = []

    original_rows = len(cleaned_df)

    # ------------------------------------------
    # STEP 1
    # COLUMN NAMES
    # ------------------------------------------

    original_columns = list(
        cleaned_df.columns
    )

    cleaned_columns = [
        clean_column_name(column)
        for column in cleaned_df.columns
    ]

    if original_columns != cleaned_columns:

        cleaned_df.columns = cleaned_columns

        cleaning_log.append(
            "Standardized column names."
        )

    # ------------------------------------------
    # STEP 2
    # TEXT CLEANING
    # ------------------------------------------

    cleaned_df = clean_text_values(
        cleaned_df,
        cleaning_log
    )

    # ------------------------------------------
    # STEP 3
    # DATE CONVERSION
    # ------------------------------------------

    cleaned_df = try_convert_dates(
        cleaned_df,
        cleaning_log
    )

    # ------------------------------------------
    # STEP 4
    # NUMERIC CONVERSION
    # ------------------------------------------

    cleaned_df = try_convert_numeric(
        cleaned_df,
        cleaning_log
    )

    # ------------------------------------------
    # STEP 5
    # GENERIC CATEGORY STANDARDIZATION
    # ------------------------------------------

    cleaned_df = standardize_categorical_values(
        cleaned_df,
        cleaning_log
    )

    # ------------------------------------------
    # STEP 6
    # BUSINESS ALIASES
    # ------------------------------------------

    cleaned_df = standardize_business_aliases(
        cleaned_df,
        cleaning_log
    )

    # ------------------------------------------
    # STEP 7
    # REMOVE DUPLICATES
    #
    # IMPORTANT:
    # This happens AFTER standardization.
    # Cleaning can turn previously different rows
    # into identical rows.
    # ------------------------------------------

    cleaned_df, duplicate_count = (
        remove_duplicates(
            cleaned_df,
            cleaning_log
        )
    )

    # ------------------------------------------
    # FINAL ROW COUNT
    # ------------------------------------------

    rows_removed = (
        original_rows
        - len(cleaned_df)
    )

    return (
        cleaned_df,
        cleaning_log,
        rows_removed
    )