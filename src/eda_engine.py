import pandas as pd


def get_numeric_summary(df):
    """
    Generate descriptive statistics for numeric columns.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T

    summary = summary.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std Dev",
            "min": "Minimum",
            "25%": "25th Percentile",
            "50%": "Median",
            "75%": "75th Percentile",
            "max": "Maximum"
        }
    )

    return summary.round(2)


def get_categorical_summary(df):
    """
    Generate summary for categorical columns.

    Datetime columns are excluded automatically.
    """

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    results = []

    for column in categorical_columns:

        mode = df[column].mode(dropna=True)

        most_common_value = (
            mode.iloc[0]
            if not mode.empty
            else "N/A"
        )

        results.append({
            "Column": column,
            "Unique Values": df[column].nunique(
                dropna=True
            ),
            "Missing Values": int(
                df[column].isna().sum()
            ),
            "Most Common Value": most_common_value
        })

    return pd.DataFrame(results)


def get_top_categories(df, column, n=10):
    """
    Return the most frequent values in a categorical column.
    """

    if column not in df.columns:
        return pd.Series(dtype="int64")

    if pd.api.types.is_datetime64_any_dtype(
        df[column]
    ):
        return pd.Series(dtype="int64")

    return (
        df[column]
        .value_counts(dropna=False)
        .head(n)
    )


def get_correlation_matrix(df):
    """
    Generate correlation matrix for numeric columns.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr().round(2)


def get_dataset_profile(df):
    """
    Create a high-level EDA profile.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    datetime_columns = [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        )
    ]

    categorical_columns = [
        column
        for column in df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
        if column not in datetime_columns
    ]

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns
    }