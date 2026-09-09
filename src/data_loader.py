import pandas as pd


def load_csv(uploaded_file):
    """
    Load an uploaded CSV file into a pandas DataFrame.
    """

    try:
        filename = getattr(uploaded_file, "name", "").lower()

        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        return df, None

    except Exception as error:
        return None, str(error)
