import pandas as pd
import plotly.express as px


def revenue_by_category(df):
    if "category" not in df.columns or "revenue" not in df.columns:
        return None

    temp = df.copy()
    temp["revenue"] = pd.to_numeric(
        temp["revenue"],
        errors="coerce"
    )

    data = (
        temp.groupby("category", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )

    if data.empty:
        return None

    return px.bar(
        data,
        x="category",
        y="revenue",
        title="Revenue by Category",
        labels={
            "category": "Category",
            "revenue": "Revenue"
        }
    )


def revenue_by_region(df):
    if "region" not in df.columns or "revenue" not in df.columns:
        return None

    temp = df.copy()
    temp["revenue"] = pd.to_numeric(
        temp["revenue"],
        errors="coerce"
    )

    data = (
        temp.groupby("region", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )

    if data.empty:
        return None

    return px.bar(
        data,
        x="region",
        y="revenue",
        title="Revenue by Region",
        labels={
            "region": "Region",
            "revenue": "Revenue"
        }
    )


def profit_by_category(df):
    if "category" not in df.columns or "profit" not in df.columns:
        return None

    temp = df.copy()
    temp["profit"] = pd.to_numeric(
        temp["profit"],
        errors="coerce"
    )

    data = (
        temp.groupby("category", as_index=False)["profit"]
        .sum()
        .sort_values("profit", ascending=False)
    )

    if data.empty:
        return None

    return px.bar(
        data,
        x="category",
        y="profit",
        title="Profit by Category",
        labels={
            "category": "Category",
            "profit": "Profit"
        }
    )


def revenue_over_time(df):
    if "order_date" not in df.columns or "revenue" not in df.columns:
        return None

    temp = df.copy()

    temp["order_date"] = pd.to_datetime(
        temp["order_date"],
        errors="coerce"
    )

    temp["revenue"] = pd.to_numeric(
        temp["revenue"],
        errors="coerce"
    )

    temp = temp.dropna(
        subset=["order_date", "revenue"]
    )

    if temp.empty:
        return None

    data = (
        temp.groupby("order_date", as_index=False)["revenue"]
        .sum()
        .sort_values("order_date")
    )

    return px.line(
        data,
        x="order_date",
        y="revenue",
        title="Revenue Over Time",
        labels={
            "order_date": "Date",
            "revenue": "Revenue"
        }
    )


def orders_by_segment(df):
    column = next(
        (candidate for candidate in ("fulfillment_type", "fulfillment", "customer_segment") if candidate in df.columns),
        None,
    )
    if column is None:
        return None

    data = (
        df[column]
        .value_counts()
        .reset_index()
    )

    data.columns = [
        column,
        "orders"
    ]

    title = "Orders by Fulfillment Type" if "fulfillment" in column else "Orders by Customer Segment"

    return px.pie(
        data,
        names=column,
        values="orders",
        title=title
    )
