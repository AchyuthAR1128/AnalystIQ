import pandas as pd


def analyze_data_quality(df):
    """
    Analyze the quality of a pandas DataFrame.

    Returns a dictionary containing
    important data-quality metrics.
    """

    total_rows = len(df)
    total_columns = len(df.columns)

    # -------------------------------
    # Missing values
    # -------------------------------

    missing_values = df.isna().sum()

    total_missing = int(missing_values.sum())

    if total_rows > 0 and total_columns > 0:
        missing_percentage = (
            total_missing / (total_rows * total_columns)
        ) * 100
    else:
        missing_percentage = 0

    # -------------------------------
    # Duplicate rows
    # -------------------------------

    duplicate_rows = int(df.duplicated().sum())

    # -------------------------------
    # Data types
    # -------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include="datetime"
    ).columns.tolist()

    # -------------------------------
    # Columns containing missing values
    # -------------------------------

    columns_with_missing = (
        missing_values[missing_values > 0]
        .sort_values(ascending=False)
    )

    # -------------------------------
    # Constant columns
    # -------------------------------

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    # -------------------------------
    # Potential outliers
    # -------------------------------

    outlier_counts = {}

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = series[
            (series < lower_bound)
            | (series > upper_bound)
        ]

        if len(outliers) > 0:
            outlier_counts[column] = int(len(outliers))

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_missing": total_missing,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "columns_with_missing": columns_with_missing,
        "constant_columns": constant_columns,
        "outlier_counts": outlier_counts,
    }