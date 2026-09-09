"""Illustrative, self-contained customer/row segmentation utilities."""

import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_clusterable_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that can contribute meaningful variation."""
    return [
        column for column in df.select_dtypes(include="number").columns
        if df[column].nunique(dropna=True) > 1
    ]


def run_segmentation(df: pd.DataFrame, features: list[str], n_clusters: int) -> dict:
    """Fit standardized K-means and return a two-dimensional PCA visualization."""
    if len(features) < 2:
        raise ValueError("Choose at least two numeric features for segmentation.")

    feature_data = df.loc[:, features].apply(pd.to_numeric, errors="coerce").dropna()
    if len(feature_data) < n_clusters:
        raise ValueError("The selected features leave too few complete rows for this cluster count.")

    scaled = StandardScaler().fit_transform(feature_data)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(scaled)
    projection = PCA(n_components=2, random_state=42).fit_transform(scaled)

    points = pd.DataFrame(
        {"PCA 1": projection[:, 0], "PCA 2": projection[:, 1], "Cluster": [f"Cluster {label + 1}" for label in labels]},
        index=feature_data.index,
    )
    assignments = df.loc[feature_data.index].copy()
    assignments["Cluster"] = points["Cluster"]
    profile = assignments.groupby("Cluster")[features].mean().round(2)
    profile.insert(0, "Rows", assignments.groupby("Cluster").size())
    figure = px.scatter(
        points,
        x="PCA 1",
        y="PCA 2",
        color="Cluster",
        title="Row Segments (PCA projection)",
        hover_data={"Cluster": True},
        color_discrete_sequence=["#2563EB", "#14B8A6", "#F59E0B", "#8B5CF6", "#EF4444", "#64748B"],
    )
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=55, b=20))
    return {"figure": figure, "assignments": assignments, "profile": profile, "features": features, "n_clusters": n_clusters}
