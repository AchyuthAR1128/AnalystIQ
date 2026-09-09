import pandas as pd


def calculate_kpis(df):
    """
    Calculate core business KPIs from cleaned data.
    """

    kpis = {}

    # ------------------------------------------
    # Total Revenue
    # ------------------------------------------

    if "revenue" in df.columns:

        revenue = pd.to_numeric(
            df["revenue"],
            errors="coerce"
        )

        kpis["total_revenue"] = revenue.sum()

    else:

        kpis["total_revenue"] = 0

    # ------------------------------------------
    # Total Profit
    # ------------------------------------------

    if "profit" in df.columns:

        profit = pd.to_numeric(
            df["profit"],
            errors="coerce"
        )

        kpis["total_profit"] = profit.sum()

    else:

        kpis["total_profit"] = 0

    # ------------------------------------------
    # Total Orders
    # ------------------------------------------

    if "order_id" in df.columns:

        kpis["total_orders"] = df[
            "order_id"
        ].nunique()

    else:

        kpis["total_orders"] = len(df)

    # ------------------------------------------
    # Units Sold
    # ------------------------------------------

    if "quantity" in df.columns:

        quantity = pd.to_numeric(
            df["quantity"],
            errors="coerce"
        )

        kpis["units_sold"] = quantity.sum()

    else:

        kpis["units_sold"] = 0

    # ------------------------------------------
    # Unique Customers
    # ------------------------------------------

    if "customer_id" in df.columns:

        kpis["unique_customers"] = df[
            "customer_id"
        ].nunique()

    else:

        kpis["unique_customers"] = 0

    # ------------------------------------------
    # Average Order Value
    # ------------------------------------------

    if (
        kpis["total_orders"] > 0
        and kpis["total_revenue"] is not None
    ):

        kpis["average_order_value"] = (
            kpis["total_revenue"]
            / kpis["total_orders"]
        )

    else:

        kpis["average_order_value"] = 0

    # ------------------------------------------
    # Profit Margin
    # ------------------------------------------

    if kpis["total_revenue"] > 0:

        kpis["profit_margin"] = (
            kpis["total_profit"]
            / kpis["total_revenue"]
        ) * 100

    else:

        kpis["profit_margin"] = 0

    return kpis


def get_top_category(df):

    if "category" not in df.columns:
        return None

    if "revenue" not in df.columns:
        return None

    temp = df.copy()

    temp["revenue"] = pd.to_numeric(
        temp["revenue"],
        errors="coerce"
    )

    result = (
        temp.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    if len(result) == 0:
        return None

    return result.index[0]


def get_top_region(df):

    if "region" not in df.columns:
        return None

    if "revenue" not in df.columns:
        return None

    temp = df.copy()

    temp["revenue"] = pd.to_numeric(
        temp["revenue"],
        errors="coerce"
    )

    result = (
        temp.groupby("region")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    if len(result) == 0:
        return None

    return result.index[0]


def get_top_product(df):

    if "product" not in df.columns:
        return None

    if "revenue" not in df.columns:
        return None

    temp = df.copy()

    temp["revenue"] = pd.to_numeric(
        temp["revenue"],
        errors="coerce"
    )

    result = (
        temp.groupby("product")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    if len(result) == 0:
        return None

    return result.index[0]