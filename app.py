"""Streamlit entry point for AnalystIQ — AI-Powered Data Analytics Copilot."""

from hashlib import sha256

import pandas as pd
import plotly.express as px
import streamlit as st

from src.code_executor import execute_code
from src.data_cleaner import clean_dataframe
from src.data_loader import load_csv
from src.data_quality import analyze_data_quality
from src.eda_engine import (
    get_categorical_summary,
    get_correlation_matrix,
    get_dataset_profile,
    get_numeric_summary,
)
from src.gemini_agent import (
    check_question_relevance,
    explain_cluster_profile,
    explain_analysis_result,
    generate_business_summary,
    generate_analysis_code,
)
from src.kpi_engine import (
    calculate_kpis,
    get_top_category,
    get_top_product,
    get_top_region,
)
from src.segmentation import get_clusterable_columns, run_segmentation
from src.report_export import build_pdf_report
from src.visualization import (
    orders_by_segment,
    profit_by_category,
    revenue_by_category,
    revenue_by_region,
    revenue_over_time,
)


PALETTE = ["#2563EB", "#14B8A6", "#F59E0B"]
st.set_page_config(page_title="AnalystIQ — AI-Powered Data Analytics Copilot ", page_icon="📊", layout="wide")


def apply_theme() -> None:
    st.markdown(
        """<style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
        [data-testid="stMetric"] { background: #F8FAFC; border: 1px solid #E2E8F0;
          border-radius: 0.6rem; padding: 0.7rem; }
        </style>""",
        unsafe_allow_html=True,
    )


def create_data_quality_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """Compare the original and cleaned datasets."""
    type_changes = []
    for column in sorted(set(original_df.columns) & set(cleaned_df.columns)):
        before, after = str(original_df[column].dtype), str(cleaned_df[column].dtype)
        if before != after:
            type_changes.append({"Column": column, "Before": before, "After": after})
    return {
        "original_rows": len(original_df), "cleaned_rows": len(cleaned_df),
        "rows_removed": len(original_df) - len(cleaned_df),
        "original_columns": len(original_df.columns), "cleaned_columns": len(cleaned_df.columns),
        "original_missing": int(original_df.isna().sum().sum()),
        "cleaned_missing": int(cleaned_df.isna().sum().sum()),
        "original_duplicates": int(original_df.duplicated().sum()),
        "cleaned_duplicates": int(cleaned_df.duplicated().sum()),
        "type_changes": pd.DataFrame(type_changes),
    }


def ensure_correct_data_types(df: pd.DataFrame, cleaning_log: list[str]) -> pd.DataFrame:
    """Final conservative type pass retained from the original workflow."""
    df = df.copy()
    for column in df.columns:
        name = column.lower()
        is_date_name = any(word in name for word in ("date", "time", "timestamp"))
        if is_date_name and not pd.api.types.is_datetime64_any_dtype(df[column]):
            source_count = df[column].notna().sum()
            if source_count and (df[column].dtype == "object" or pd.api.types.is_string_dtype(df[column])):
                converted = pd.to_datetime(df[column], errors="coerce", format="mixed")
                if converted.notna().sum() / source_count >= 0.8:
                    df[column] = converted
                    cleaning_log.append(f"Converted '{column}' to datetime.")
    numeric_words = ("price", "revenue", "sales", "profit", "cost", "amount", "quantity", "discount")
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) or not any(word in column.lower() for word in numeric_words):
            continue
        source_count = df[column].notna().sum()
        if not source_count:
            continue
        values = df[column].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False)
        values = values.str.replace("$", "", regex=False).str.replace("€", "", regex=False).str.replace("£", "", regex=False).str.strip()
        converted = pd.to_numeric(values, errors="coerce")
        if converted.notna().sum() / source_count >= 0.8:
            df[column] = converted
            cleaning_log.append(f"Converted '{column}' to numeric.")
    return df


def prepare_dataset(uploaded_file) -> None:
    """Process an upload once; all tabs reuse the session-held analysis."""
    fingerprint = sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("dataset_fingerprint") == fingerprint:
        return
    with st.spinner("Processing and cleaning your dataset…"):
        original_df, error = load_csv(uploaded_file)
        if error:
            st.session_state.pop("analysis", None)
            st.session_state["dataset_error"] = error
            return
        cleaned_df, cleaning_log, rows_removed = clean_dataframe(original_df)
        cleaned_df = ensure_correct_data_types(cleaned_df, cleaning_log)
        st.session_state["analysis"] = {
            "name": uploaded_file.name, "original_df": original_df, "cleaned_df": cleaned_df,
            "quality": analyze_data_quality(original_df),
            "quality_report": create_data_quality_report(original_df, cleaned_df),
            "cleaning_log": cleaning_log, "rows_removed": rows_removed,
            "kpis": calculate_kpis(cleaned_df),
            "chat_history": [],
        }
        st.session_state["dataset_fingerprint"] = fingerprint
        st.session_state["chat_history"] = []
        st.session_state.pop("business_ai_summary", None)
        st.session_state.pop("segmentation_result", None)
        st.session_state.pop("cluster_narrative", None)
        st.session_state.pop("dataset_error", None)


def render_overview(analysis: dict) -> None:
    df, quality, report = analysis["original_df"], analysis["quality"], analysis["quality_report"]
    st.header("📁 Overview")
    st.caption(f"Loaded file: {analysis['name']}. Analysis persists while you move between tabs.")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{df.shape[0]:,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Memory usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    st.subheader("🧹 Data quality")
    a, b, c, d = st.columns(4)
    a.metric("Missing values", f"{quality['total_missing']:,}")
    b.metric("Missing %", f"{quality['missing_percentage']:.2f}%")
    c.metric("Duplicate rows", f"{quality['duplicate_rows']:,}")
    d.metric("Potential outliers", f"{sum(quality['outlier_counts'].values()):,}")
    if not quality["columns_with_missing"].empty:
        # ``names=`` was introduced after the pandas version used by some
        # supported deployments, so label the resulting columns explicitly.
        missing = quality["columns_with_missing"].rename("Missing values").reset_index()
        missing.columns = ["Column", "Missing values"]
        missing["Missing %"] = (missing["Missing values"] / len(df) * 100).round(2)
        st.dataframe(missing, use_container_width=True, hide_index=True)
    if quality["outlier_counts"]:
        st.dataframe(pd.DataFrame(quality["outlier_counts"].items(), columns=["Column", "Potential outliers"]), use_container_width=True, hide_index=True)
    st.subheader("✨ Cleaning actions")
    a, b, c, d = st.columns(4)
    a.metric("Original rows", f"{report['original_rows']:,}")
    b.metric("Cleaned rows", f"{report['cleaned_rows']:,}")
    c.metric("Duplicates removed", f"{report['original_duplicates'] - report['cleaned_duplicates']:,}")
    d.metric("Missing values remaining", f"{report['cleaned_missing']:,}")
    if analysis["cleaning_log"]:
        st.success(f"Completed {len(analysis['cleaning_log'])} cleaning action(s).")
        for action in analysis["cleaning_log"]:
            st.write(f"• {action}")
    else:
        st.info("No automatic cleaning was required.")
    if not report["type_changes"].empty:
        st.dataframe(report["type_changes"], use_container_width=True, hide_index=True)
    st.subheader("Dataset preview")
    st.dataframe(analysis["cleaned_df"].head(10), use_container_width=True)


def build_business_summary(analysis: dict) -> str:
    kpis, df = analysis["kpis"], analysis["cleaned_df"]
    summary = f"The cleaned dataset contains {len(df):,} rows and {kpis['total_orders']:,} orders."
    if kpis["total_revenue"]:
        summary += f" Revenue is ₹{kpis['total_revenue']:,.0f} with a {kpis['profit_margin']:.1f}% profit margin."
    leaders = [("category", get_top_category(df)), ("region", get_top_region(df)), ("product", get_top_product(df))]
    leader_text = [f"top {label}: {value}" for label, value in leaders if value]
    return summary + (" Key performers — " + "; ".join(leader_text) + "." if leader_text else "")


def render_kpis(analysis: dict) -> None:
    kpis, df = analysis["kpis"], analysis["cleaned_df"]
    st.header("💼 KPIs & Insights")
    a, b, c, d = st.columns(4)
    a.metric("Total revenue", f"₹{kpis['total_revenue']:,.0f}")
    b.metric("Total profit", f"₹{kpis['total_profit']:,.0f}")
    c.metric("Total orders", f"{kpis['total_orders']:,}")
    d.metric("Units sold", f"{kpis['units_sold']:,.0f}")
    a, b, c = st.columns(3)
    a.metric("Customers", f"{kpis['unique_customers']:,}")
    b.metric("Average order value", f"₹{kpis['average_order_value']:,.0f}")
    c.metric("Profit margin", f"{kpis['profit_margin']:.2f}%")
    st.subheader("🏆 Top performers")
    a, b, c = st.columns(3)
    a.info(f"**Top category**\n\n{get_top_category(df) or 'N/A'}")
    b.info(f"**Top region**\n\n{get_top_region(df) or 'N/A'}")
    c.info(f"**Top product**\n\n{get_top_product(df) or 'N/A'}")
    st.subheader("Key business summary")
    if st.button("Generate AI business summary"):
        with st.spinner("Generating an executive summary…"):
            try:
                st.session_state["business_ai_summary"] = generate_business_summary(
                    kpis, get_top_category(df), get_top_region(df), get_top_product(df)
                )
            except Exception as error:
                st.error(f"Could not generate the summary: {error}")
    st.info(st.session_state.get("business_ai_summary") or build_business_summary(analysis))


def render_eda(analysis: dict) -> None:
    df, profile = analysis["cleaned_df"], get_dataset_profile(analysis["cleaned_df"])
    st.header("🔎 Exploratory Data Analysis")
    a, b, c = st.columns(3)
    a.metric("Numeric columns", len(profile["numeric_columns"]))
    b.metric("Categorical columns", len(profile["categorical_columns"]))
    c.metric("Date columns", len(profile["datetime_columns"]))
    st.subheader("Numeric analysis")
    numeric = get_numeric_summary(df)
    if numeric.empty: st.info("No numeric columns available.")
    else: st.dataframe(numeric, use_container_width=True)
    st.subheader("Categorical analysis")
    categorical = get_categorical_summary(df)
    if categorical.empty: st.info("No categorical columns available.")
    else: st.dataframe(categorical, use_container_width=True, hide_index=True)
    st.subheader("Correlation analysis")
    correlation = get_correlation_matrix(df)
    if correlation.empty: st.info("At least two numeric columns are needed for correlation analysis.")
    else: st.dataframe(correlation, use_container_width=True)
    st.subheader("Column information")
    info = pd.DataFrame({"Column": df.columns, "Data type": df.dtypes.astype(str).values, "Non-null values": df.notna().sum().values, "Missing values": df.isna().sum().values, "Unique values": df.nunique().values})
    st.dataframe(info, use_container_width=True, hide_index=True)


def get_charts(df: pd.DataFrame) -> list:
    charts = [revenue_by_category(df), revenue_by_region(df), profit_by_category(df), revenue_over_time(df), orders_by_segment(df)]
    for fig in charts:
        if fig is not None:
            fig.update_layout(template="plotly_white", colorway=PALETTE, margin=dict(l=20, r=20, t=55, b=20))
    return charts


def render_charts(analysis: dict) -> None:
    st.header("📈 Interactive charts")
    available = [fig for fig in get_charts(analysis["cleaned_df"]) if fig is not None]
    if not available:
        st.info("No standard business charts are available for these columns.")
        return
    for index in range(0, len(available), 2):
        left, right = st.columns(2)
        with left: st.plotly_chart(available[index], use_container_width=True)
        if index + 1 < len(available):
            with right: st.plotly_chart(available[index + 1], use_container_width=True)


def create_question_chart(question: str, df: pd.DataFrame):
    """Original Plotly fallback used when a pandas-only AI response has no figure."""
    question = question.lower()
    category = next((col for col in df if "category" in col.lower()), None)
    region = next((col for col in df if "region" in col.lower()), None)
    revenue = next((col for col in df if any(word in col.lower() for word in ("revenue", "sales", "amount"))), None)
    profit = next((col for col in df if "profit" in col.lower()), None)
    date = next((col for col in df if any(word in col.lower() for word in ("date", "time", "timestamp")) and pd.api.types.is_datetime64_any_dtype(df[col])), None)
    if category and revenue and "category" in question and any(word in question for word in ("revenue", "sales", "amount")):
        data = df.groupby(category, as_index=False)[revenue].sum().sort_values(revenue, ascending=False)
        return px.bar(data, x=category, y=revenue, title="Revenue by Category", color_discrete_sequence=PALETTE)
    if region and revenue and "region" in question and any(word in question for word in ("revenue", "sales", "amount")):
        data = df.groupby(region, as_index=False)[revenue].sum().sort_values(revenue, ascending=False)
        return px.bar(data, x=region, y=revenue, title="Revenue by Region", color_discrete_sequence=PALETTE)
    if category and profit and "category" in question and "profit" in question:
        data = df.groupby(category, as_index=False)[profit].sum().sort_values(profit, ascending=False)
        return px.bar(data, x=category, y=profit, title="Profit by Category", color_discrete_sequence=PALETTE)
    if date and revenue and any(word in question for word in ("trend", "over time", "monthly", "daily", "date", "growth")):
        data = df.groupby(date, as_index=False)[revenue].sum().sort_values(date)
        return px.line(data, x=date, y=revenue, title="Revenue Trend Over Time", markers=True, color_discrete_sequence=PALETTE)
    return None


def render_ai_result(entry: dict) -> None:
    with st.chat_message("user"): st.write(entry["question"])
    with st.chat_message("assistant"):
        if entry.get("error"):
            st.warning(entry["error"])
            return
        with st.expander("AI-generated analysis code"):
            st.code(entry["code"], language="python")
        result = entry["result"]
        if isinstance(result, pd.DataFrame): st.dataframe(result, use_container_width=True)
        elif isinstance(result, pd.Series): st.dataframe(result.to_frame(), use_container_width=True)
        else: st.write(result)
        if entry.get("chart") is not None: st.plotly_chart(entry["chart"], use_container_width=True)
        st.info(entry["insight"])


def render_ask_ai(analysis: dict) -> None:
    st.header("🤖 Ask AI")
    st.caption("Ask a data question. Your conversation is retained for this uploaded dataset.")
    history = st.session_state.setdefault("chat_history", [])
    with st.container(height=520):
        for entry in history: render_ai_result(entry)
    question = st.chat_input("e.g. Which category generated the highest revenue?")
    if not question: return
    entry = {"question": question}
    with st.spinner("Checking the question and generating analysis…"):
        try:
            if not check_question_relevance(question, analysis["cleaned_df"]):
                entry["error"] = "This question does not appear related to the uploaded dataset. Try asking about its columns, revenue, profit, orders, customers, or trends."
            else:
                code = generate_analysis_code(question, analysis["cleaned_df"])
                result, error, generated_fig = execute_code(code, analysis["cleaned_df"], return_figure=True)
                if error: entry["error"] = f"Analysis failed: {error}"
                else:
                    chart = generated_fig if generated_fig is not None else create_question_chart(question, analysis["cleaned_df"])
                    entry.update({"code": code, "result": result, "chart": chart})
                    entry["insight"] = explain_analysis_result(question, result)
        except Exception as error:
            entry["error"] = f"AI Analyst error: {error}"
    history.append(entry)
    analysis["chat_history"] = history
    st.rerun()


def render_segmentation(analysis: dict) -> None:
    """Provide an opt-in, dataset-adaptive clustering workflow."""
    df = analysis["cleaned_df"]
    available = get_clusterable_columns(df)
    st.header("🔮 Predictive analysis: segmentation")
    st.caption("Group similar rows using numeric patterns. This runs locally and is illustrative, not a production scoring model.")
    if len(available) < 2:
        st.info("Segmentation needs at least two numeric columns with more than one distinct value.")
        return
    default_features = available[: min(4, len(available))]
    selection_mode = st.selectbox("Feature selection", ["Automatic (recommended)", "Manual override"])
    if selection_mode == "Automatic (recommended)":
        features = default_features
        st.caption("Automatically selected: " + ", ".join(features))
    else:
        features = st.multiselect("Features used for clustering", available, default=default_features)
    complete_rows = len(df[features].dropna()) if features else 0
    maximum = min(8, complete_rows)
    if len(features) < 2 or maximum < 2:
        st.warning("Select at least two features that have two or more complete rows.")
        return
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=maximum, value=min(3, maximum))
    if st.button("Run segmentation", type="primary"):
        with st.spinner("Training the K-means segmentation model…"):
            try:
                st.session_state["segmentation_result"] = run_segmentation(df, features, n_clusters)
                st.session_state.pop("cluster_narrative", None)
            except ValueError as error:
                st.error(str(error))
    result = st.session_state.get("segmentation_result")
    if not result:
        return
    st.plotly_chart(result["figure"], use_container_width=True)
    st.subheader("Cluster profile")
    st.dataframe(result["profile"], use_container_width=True)
    with st.expander("View assigned rows"):
        st.dataframe(result["assignments"], use_container_width=True)
    if st.button("Generate AI cluster narrative"):
        with st.spinner("Generating a concise cluster narrative…"):
            try:
                st.session_state["cluster_narrative"] = explain_cluster_profile(result["profile"].to_string())
            except Exception as error:
                st.error(f"Could not generate the narrative: {error}")
    if st.session_state.get("cluster_narrative"):
        st.info(st.session_state["cluster_narrative"])
    st.markdown("**Model notes:** K-means clusters rows by distance after standardizing the selected features; PCA only projects those patterns into two dimensions for display. Results are sensitive to the selected columns and cluster count, so they are useful for exploration—not production decisions or causal conclusions.")


def main() -> None:
    apply_theme()
    st.title("📊 AnalystIQ — AI-Powered Data Analytics Copilot ")
    st.markdown("Turn messy business data into actionable insights.")
    with st.sidebar:
        st.header("Dataset")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="dataset_uploader")
        st.caption("Your processed data remains available while you navigate the app.")
    if uploaded_file is not None: prepare_dataset(uploaded_file)
    if st.session_state.get("dataset_error"):
        st.error(f"Unable to read the dataset: {st.session_state['dataset_error']}")
        return
    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info("👈 Upload a CSV or Excel file from the sidebar to begin.")
        return
    overview, kpi_tab, eda_tab, charts_tab, ai_tab, segmentation_tab, report_tab = st.tabs(["Overview", "KPIs & Insights", "EDA", "Charts", "Ask AI", "Predictive", "Report"])
    with overview: render_overview(analysis)
    with kpi_tab: render_kpis(analysis)
    with eda_tab: render_eda(analysis)
    with charts_tab: render_charts(analysis)
    with ai_tab: render_ask_ai(analysis)
    with segmentation_tab: render_segmentation(analysis)
    with report_tab:
        st.header("📄 Report")
        st.caption("Create a concise PDF that combines data quality, KPIs, selected charts, insights, and segmentation when available.")
        if st.button("Prepare PDF report", type="primary"):
            with st.spinner("Preparing your PDF report…"):
                try:
                    st.session_state["pdf_report"] = build_pdf_report(
                        analysis,
                        get_charts(analysis["cleaned_df"]),
                        st.session_state.get("business_ai_summary") or build_business_summary(analysis),
                        st.session_state.get("segmentation_result"),
                    )
                except Exception as error:
                    st.error(f"Could not prepare the report: {error}")
        if st.session_state.get("pdf_report"):
            st.download_button(
                "Download report (PDF)",
                data=st.session_state["pdf_report"],
                file_name="ai_data_analyst_report.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
