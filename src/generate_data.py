import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# Reproducible data
random.seed(42)
np.random.seed(42)


ROWS = 10000


regions = [
    "Bangalore",
    "Bengaluru",
    "bangalore",
    "BLR",
    "Mumbai",
    "mumbai",
    "Delhi",
    "New Delhi",
    "Chennai",
    "Hyderabad",
    "Pune"
]

categories = [
    "Electronics",
    "electronics",
    "Electronics ",
    "Furniture",
    "furniture",
    "Clothing",
    "clothing",
    "Home & Kitchen"
]

products = {
    "Electronics": [
        "Laptop",
        "Smartphone",
        "Monitor",
        "Keyboard",
        "Mouse",
        "Tablet",
        "Headphones"
    ],
    "Furniture": [
        "Office Chair",
        "Desk",
        "Bookshelf",
        "Sofa",
        "Coffee Table"
    ],
    "Clothing": [
        "Jacket",
        "T-Shirt",
        "Jeans",
        "Shoes",
        "Backpack"
    ],
    "Home & Kitchen": [
        "Mixer",
        "Cookware Set",
        "Water Bottle",
        "Air Fryer",
        "Coffee Maker"
    ]
}

customer_segments = [
    "Consumer",
    "Corporate",
    "Home Office"
]

payment_methods = [
    "Credit Card",
    "Debit Card",
    "UPI",
    "Cash on Delivery",
    "COD",
    "upi",
    "credit card"
]

customers = [
    "Rahul Sharma",
    "Priya Kumar",
    "Arjun Reddy",
    "Sneha Rao",
    "Vikram Singh",
    "Ananya Das",
    "Rohit Verma",
    "Neha Patel",
    "Kiran Kumar",
    "Amit Joshi"
]


# --------------------------------------------------
# Generate base data
# --------------------------------------------------

rows = []

start_date = datetime(2025, 1, 1)

for i in range(ROWS):

    category = random.choice([
        "Electronics",
        "Furniture",
        "Clothing",
        "Home & Kitchen"
    ])

    product = random.choice(products[category])

    quantity = random.randint(1, 10)

    unit_price = random.choice([
        499,
        799,
        999,
        1499,
        2499,
        4999,
        9999,
        19999,
        49999
    ])

    discount = round(
        random.uniform(0, 30),
        2
    )

    revenue = round(
        quantity * unit_price * (1 - discount / 100),
        2
    )

    profit = round(
        revenue * random.uniform(0.05, 0.30),
        2
    )

    order_date = start_date + timedelta(
        days=random.randint(0, 364)
    )

    rows.append({
        "Order ID": f"ORD-{100000 + i}",
        "Order Date": order_date.strftime("%Y-%m-%d"),
        "Customer ID": f"CUST-{random.randint(1000, 3000)}",
        "Customer Name": random.choice(customers),
        "Region": random.choice(regions),
        "Category": random.choice(categories),
        "Product": product,
        "Quantity": quantity,
        "Unit Price": unit_price,
        "Discount": discount,
        "Revenue": revenue,
        "Profit": profit,
        "Payment Method": random.choice(payment_methods),
        "Customer Segment": random.choice(customer_segments),
        "Customer Rating": random.choice([1, 2, 3, 4, 5])
    })


df = pd.DataFrame(rows)


# --------------------------------------------------
# Introduce missing values
# --------------------------------------------------

for column in [
    "Customer Name",
    "Region",
    "Discount",
    "Customer Rating"
]:

    indexes = np.random.choice(
        df.index,
        size=150,
        replace=False
    )

    df.loc[indexes, column] = np.nan


# --------------------------------------------------
# Introduce duplicate rows
# --------------------------------------------------

duplicate_rows = df.sample(
    120,
    random_state=42
)

df = pd.concat(
    [df, duplicate_rows],
    ignore_index=True
)


# --------------------------------------------------
# Mess up date formats
# --------------------------------------------------

date_indexes = np.random.choice(
    df.index,
    size=250,
    replace=False
)

for index in date_indexes:

    date = pd.to_datetime(
        df.loc[index, "Order Date"],
        errors="coerce"
    )

    if pd.notna(date):

        formats = [
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%b %d %Y"
        ]

        selected_format = random.choice(formats)

        df.loc[index, "Order Date"] = date.strftime(
            selected_format
        )


# --------------------------------------------------
# Mess up numeric formatting
# --------------------------------------------------
df["Revenue"] = df["Revenue"].astype(object)
numeric_indexes = np.random.choice(
    df.index,
    size=250,
    replace=False
)

for index in numeric_indexes:

    revenue = df.loc[index, "Revenue"]

    if pd.notna(revenue):

        df.loc[index, "Revenue"] = f"₹{revenue:,.2f}"


# --------------------------------------------------
# Add whitespace
# --------------------------------------------------

whitespace_indexes = np.random.choice(
    df.index,
    size=200,
    replace=False
)

for index in whitespace_indexes:

    if pd.notna(df.loc[index, "Customer Name"]):

        df.loc[index, "Customer Name"] = (
            "  "
            + str(df.loc[index, "Customer Name"])
            + "  "
        )


# --------------------------------------------------
# Create extreme outliers
# --------------------------------------------------

outlier_indexes = np.random.choice(
    df.index,
    size=20,
    replace=False
)

df.loc[outlier_indexes, "Quantity"] = np.random.randint(
    100,
    500,
    size=len(outlier_indexes)
)


# --------------------------------------------------
# Shuffle rows
# --------------------------------------------------

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = "data/messy_ecommerce.csv"

df.to_csv(
    output_path,
    index=False
)

print("Messy dataset created successfully.")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_path}")