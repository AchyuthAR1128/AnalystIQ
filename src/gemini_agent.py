import os
import re

from dotenv import load_dotenv
from google import genai


load_dotenv()


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to the .env file."
        )

    return genai.Client(api_key=api_key)


def clean_generated_code(code):
    """Remove Markdown formatting from Gemini's code response."""

    if not code:
        return ""

    code = code.strip()

    match = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        code,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        code = match.group(1).strip()

    code = code.replace("```python", "")
    code = code.replace("```Python", "")
    code = code.replace("```PYTHON", "")
    code = code.replace("```", "")

    return code.strip()


def _find_column(df, candidates):
    """
    Find the first matching column from a list of possible names.
    Matching is case-insensitive.
    """
    column_map = {str(column).lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in column_map:
            return column_map[candidate.lower()]

    return None


def check_question_relevance(question, df):
    """
    Fast local relevance check.

    This function deliberately does NOT call Gemini.
    It rejects only clearly unrelated questions.
    """

    normalized_question = " ".join(
        re.findall(r"[a-z0-9]+", str(question).lower())
    )

    question_words = set(normalized_question.split())

    business_terms = {
        "revenue",
        "sales",
        "profit",
        "order",
        "orders",
        "customer",
        "customers",
        "product",
        "products",
        "category",
        "categories",
        "region",
        "regions",
        "date",
        "dates",
        "trend",
        "trends",
        "comparison",
        "compare",
        "highest",
        "lowest",
        "top",
        "bottom",
        "average",
        "mean",
        "median",
        "total",
        "sum",
        "count",
        "percentage",
        "percent",
        "distribution",
        "correlation",
        "relationship",
        "pattern",
        "statistics",
        "growth",
        "performance",
        "data",
        "dataset",
        "column",
        "columns",
        "value",
        "values",
        "minimum",
        "maximum",
        "max",
        "min",
    }

    column_terms = set()
    normalized_columns = []

    for column in df.columns:
        normalized_column = " ".join(
            re.findall(r"[a-z0-9]+", str(column).lower())
        )

        normalized_columns.append(normalized_column)
        column_terms.update(normalized_column.split())

    # Direct business/data terminology match.
    if question_words & business_terms:
        return True

    # Match words from dataset column names.
    if question_words & column_terms:
        return True

    # Match complete column names.
    for column in normalized_columns:
        if column and column in normalized_question:
            return True

    # Clearly unrelated questions.
    obvious_unrelated_phrases = (
        "world cup",
        "fifa",
        "capital of",
        "tell me a joke",
        "president",
        "what is python",
        "sports news",
        "politics",
    )

    if any(
        phrase in normalized_question
        for phrase in obvious_unrelated_phrases
    ):
        return False

    # When uncertain, allow the question through.
    # Gemini can attempt to answer it from the dataset.
    return True


def generate_analysis_code(question, df):
    """
    Generate Pandas + Plotly analysis code.

    Common questions use a local fast path.
    Other questions are handled by Gemini.
    """

    question_lower = str(question).lower()

    # ==========================================================
    # COLUMN DISCOVERY FOR FAST PATHS
    # ==========================================================

    product_column = _find_column(
        df,
        ["product", "product_name", "item", "item_name"],
    )

    revenue_column = _find_column(
        df,
        ["revenue", "sales", "amount", "total_sales"],
    )

    profit_column = _find_column(
        df,
        ["profit", "net_profit"],
    )

    quantity_column = _find_column(
        df,
        ["quantity", "units_sold", "units", "qty"],
    )

    price_column = _find_column(
        df,
        ["unit_price", "selling_price", "price", "sale_price"],
    )

    category_column = _find_column(
        df,
        ["category", "product_category"],
    )

    region_column = _find_column(
        df,
        ["region", "sales_region", "location"],
    )

    # ==========================================================
    # FAST PATH
    # Common analyst questions are answered locally.
    # No Gemini API call is made.
    # ==========================================================

    # ----------------------------------------------------------
    # Top 5 products by average selling price
    # ----------------------------------------------------------

    if (
        "top 5" in question_lower
        and product_column
        and price_column
        and "product" in question_lower
        and "average" in question_lower
        and (
            "selling price" in question_lower
            or "price" in question_lower
        )
    ):
        return f"""
result = (
    df.groupby("{product_column}")["{price_column}"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

data = result.reset_index()
fig = px.bar(
    data,
    x="{product_column}",
    y="{price_column}",
    title="Top 5 Products by Average Selling Price",
)
""".strip()

    # ----------------------------------------------------------
    # Top 5 products by revenue
    # ----------------------------------------------------------

    if (
        "top 5" in question_lower
        and product_column
        and revenue_column
        and "product" in question_lower
        and (
            "revenue" in question_lower
            or "sales" in question_lower
            or "amount" in question_lower
        )
    ):
        return f"""
result = (
    df.groupby("{product_column}")["{revenue_column}"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

data = result.reset_index()
fig = px.bar(
    data,
    x="{product_column}",
    y="{revenue_column}",
    title="Top 5 Products by Revenue",
)
""".strip()

    # ----------------------------------------------------------
    # Top 5 products by units sold
    # ----------------------------------------------------------

    if (
        "top 5" in question_lower
        and product_column
        and quantity_column
        and "product" in question_lower
        and (
            "units" in question_lower
            or "quantity" in question_lower
            or "qty" in question_lower
        )
    ):
        return f"""
result = (
    df.groupby("{product_column}")["{quantity_column}"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

data = result.reset_index()
fig = px.bar(
    data,
    x="{product_column}",
    y="{quantity_column}",
    title="Top 5 Products by Units Sold",
)
""".strip()

    # ----------------------------------------------------------
    # Total revenue
    # ----------------------------------------------------------

    if (
        "total revenue" in question_lower
        and revenue_column
    ):
        return f"""
result = df["{revenue_column}"].sum()
""".strip()

    # ----------------------------------------------------------
    # Total profit
    # ----------------------------------------------------------

    if (
        "total profit" in question_lower
        and profit_column
    ):
        return f"""
result = df["{profit_column}"].sum()
""".strip()

    # ----------------------------------------------------------
    # Revenue by category
    # ----------------------------------------------------------

    if (
        category_column
        and revenue_column
        and "category" in question_lower
        and (
            "revenue" in question_lower
            or "sales" in question_lower
            or "amount" in question_lower
        )
    ):
        return f"""
result = (
    df.groupby("{category_column}")["{revenue_column}"]
    .sum()
    .sort_values(ascending=False)
)

data = result.reset_index()
fig = px.bar(
    data,
    x="{category_column}",
    y="{revenue_column}",
    title="Revenue by Category",
)
""".strip()

    # ----------------------------------------------------------
    # Revenue by region
    # ----------------------------------------------------------

    if (
        region_column
        and revenue_column
        and "region" in question_lower
        and (
            "revenue" in question_lower
            or "sales" in question_lower
            or "amount" in question_lower
        )
    ):
        return f"""
result = (
    df.groupby("{region_column}")["{revenue_column}"]
    .sum()
    .sort_values(ascending=False)
)

data = result.reset_index()
fig = px.bar(
    data,
    x="{region_column}",
    y="{revenue_column}",
    title="Revenue by Region",
)
""".strip()

    # ==========================================================
    # GEMINI PATH
    # ==========================================================

    client = get_gemini_client()

    columns = list(df.columns)
    dtypes = df.dtypes.astype(str).to_dict()

    prompt = f"""
You are an expert Data Analyst.

A pandas DataFrame named df contains business data.

Dataset columns:
{columns}

Column data types:
{dtypes}

User question:
{question}

Generate Python code using pandas and Plotly that answers the question.

IMPORTANT:

The answer must be stored in a variable named:
result

When a visualization meaningfully improves the answer, also create:
fig

using the pre-provided Plotly namespaces:
px
or
go

Do NOT import any libraries.

VISUALIZATION GUIDANCE:

Prefer a chart when the question involves:
- Top-N or ranking results
- comparisons between categories
- comparisons between regions
- comparisons between products
- revenue, sales, or profit breakdowns
- trends over time
- distributions
- relationships between numeric variables

Choose an appropriate chart:

- Bar chart for rankings and categorical comparisons.
- Horizontal bar chart for Top-N results with several categories.
- Line chart for trends over time.
- Scatter plot for relationships between numeric variables.
- Pie chart only for simple part-to-whole comparisons with a small
  number of categories.

Do NOT create a chart for a simple single-number answer when a chart
would not add meaningful information.

STRICT RULES:

1. Use the existing DataFrame named df.
2. Use the EXACT column names provided above.
3. Use pandas plus the pre-provided Plotly namespaces px and go.
4. Do not import any libraries.
5. Do not read or write files.
6. Do not access the internet.
7. Do not use os, sys, subprocess, pathlib, requests, socket,
   open, eval, exec, or __import__.
8. Store the final answer in a variable named result.
9. If a visualization is useful, store the Plotly figure in a variable
   named fig.
10. Return ONLY executable Python code.
11. Do NOT return Markdown.
12. Do NOT use ``` or ```python.
13. Do not explain the code.
14. If a numeric column contains currency symbols, commas, or text
    formatting, convert it to numeric before calculations.
15. Do not invent columns that do not exist in the dataset.
16. Do not invent data values.
17. Keep the analysis directly focused on the user's question.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    generated_code = interaction.output_text

    return clean_generated_code(generated_code)


def explain_analysis_result(question, result):
    """
    Generate a concise business explanation for a completed analysis.
    """

    client = get_gemini_client()

    prompt = f"""
You are a professional Business Data Analyst.

User question:
{question}

Analysis result:
{result}

Provide a concise business explanation.

Rules:
1. Answer the question directly.
2. Explain what the result means.
3. Mention a business implication only when supported by the result.
4. Do not invent facts, numbers, causes, or assumptions.
5. Keep the answer concise and useful to a business stakeholder.
6. Do not mention AI or Gemini.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    return interaction.output_text.strip()


def explain_cluster_profile(profile):
    """
    Turn cluster-level descriptive statistics into a concise
    business narrative.
    """

    client = get_gemini_client()

    prompt = f"""
You are a business analyst.

Summarize these K-means cluster statistics in 2-4 short bullets.

Identify what distinguishes each cluster using ONLY the shown numbers.

Do not claim:
- causation
- targeting suitability
- production accuracy

Cluster statistics:
{profile}
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    return interaction.output_text.strip()


def generate_business_summary(
    kpis,
    top_category,
    top_region,
    top_product,
):
    """
    Generate a tightly bounded executive summary from
    already computed dashboard values.
    """

    client = get_gemini_client()

    prompt = f"""
You are a business analyst.

Write a concise executive summary of these already computed metrics
in no more than three sentences.

Do not invent data.
Do not make causal claims.
Mention a useful commercial implication only when the metrics support it.

KPIs:
{kpis}

Top category:
{top_category or 'not available'}

Top region:
{top_region or 'not available'}

Top product:
{top_product or 'not available'}
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    return interaction.output_text.strip()