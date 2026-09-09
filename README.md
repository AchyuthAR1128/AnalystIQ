# AnalystIQ — AI-Powered Data Analytics Copilot

AnalystIQ is an AI-powered data analytics application that helps transform messy business data into meaningful, actionable insights.

It simulates the workflow of a real-world data analyst by automating data quality analysis, data cleaning, KPI generation, exploratory data analysis, interactive visualization, AI-assisted analysis, segmentation, and report generation.

## 🚀 Key Features

- 📂 Upload CSV or Excel datasets
- 🧹 Automatic data cleaning and preprocessing
- 🔍 Automated data quality analysis
- 📊 Automatic KPI generation
- 📈 Exploratory Data Analysis (EDA)
- 📉 Interactive Plotly visualizations
- 🤖 Natural-language data analysis using Google Gemini
- 💻 AI-generated Pandas analysis code
- 🔐 Constrained execution of AI-generated analysis code
- 💡 Business-focused insights from analytical results
- 👥 K-Means based data segmentation
- 📄 Automated PDF analytics reports

## 🧠 How AnalystIQ Works

```text
Upload Dataset
      ↓
Data Loading
      ↓
Data Quality Analysis
      ↓
Automatic Data Cleaning
      ↓
KPI Generation
      ↓
Exploratory Data Analysis
      ↓
Interactive Visualizations
      ↓
Ask AI
      ↓
Gemini Generates Analysis Code
      ↓
Constrained Code Execution
      ↓
Business Insights
      ↓
Segmentation
      ↓
PDF Report
💬 Natural-Language AI Analyst

Users can ask business questions in plain English, such as:

Which category generated the highest revenue?
Which region has the highest revenue?
What is the revenue trend over time?
Compare revenue across regions.
Which category has the highest profit?

AnalystIQ can generate the required analytical code, execute the analysis, display the result, create a visualization when appropriate, and explain the business meaning of the result.

📊 Data Quality & Cleaning

AnalystIQ is designed to work with messy, real-world-style datasets containing issues such as:

Missing values
Duplicate records
Inconsistent categorical values
Inconsistent region names
Mixed data types
Date formatting issues
Potential outliers

The application analyzes the dataset before applying automated cleaning and presents the changes made to the user.

🛠️ Technology Stack

Language

Python

Data Analysis

Pandas
NumPy
Scikit-learn

Visualization

Plotly
Matplotlib

Generative AI

Google Gemini API
AI-generated Pandas analysis

Application

Streamlit

Reporting

ReportLab
Kaleido

Development

Git
GitHub
python-dotenv
📁 Project Structure
AnalystIQ/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── messy_ecommerce.csv
│   └── test_sales.csv
│
└── src/
    ├── code_executor.py
    ├── data_cleaner.py
    ├── data_loader.py
    ├── data_quality.py
    ├── eda_engine.py
    ├── gemini_agent.py
    ├── generate_data.py
    ├── kpi_engine.py
    ├── report_export.py
    ├── segmentation.py
    └── visualization.py
⚙️ Local Installation
1. Clone the repository
git clone https://github.com/AchyuthAR1128/AnalystIQ.git
cd AnalystIQ
2. Create a virtual environment
python -m venv .venv
3. Activate the virtual environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
5. Configure Gemini API

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here

Never commit the .env file or expose your API key publicly.

6. Run AnalystIQ
python -m streamlit run app.py
🔐 AI Code Execution

AnalystIQ uses a constrained execution layer for AI-generated analytical code.

The execution layer restricts operations such as:

File access
Network access
System commands
Arbitrary imports
File writing

Only approved analytical functionality is exposed to generated analysis code.

The execution layer is a constrained application-level mechanism and should not be considered a complete security sandbox for hostile code.

🎯 Project Objective

AnalystIQ demonstrates how Generative AI can augment traditional data-analysis workflows.

The project combines:

Data Analysis + Data Cleaning + EDA + Data Visualization + Machine Learning + Generative AI

The goal is to demonstrate an end-to-end analytics workflow that moves from messy raw data to business-oriented insights.

📌 Portfolio Value

This project demonstrates practical skills in:

Python
Pandas
SQL-oriented analytical thinking
Data cleaning
Exploratory Data Analysis
Data visualization
KPI and business analysis
Machine learning
Generative AI
AI-assisted code generation
Streamlit application development
Git and GitHub
End-to-end project development

👨‍💻 Author

A R Achyuth

Computer Science & Engineering Graduate

⭐ Built as an end-to-end AI-powered data analytics portfolio project.
