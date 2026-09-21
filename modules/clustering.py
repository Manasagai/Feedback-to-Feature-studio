"""KMeans clustering and data-derived theme labels for feedback insights."""

from collections import Counter
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

STOP_WORDS = {"the", "and", "for", "with", "this", "that", "have", "from", "would", "should", "they", "when", "more", "very", "want", "need", "user", "users", "please"}


def cluster_feedback(embeddings: np.ndarray, n_clusters: int | None = None):
    """
    Apply KMeans clustering to feedback embeddings.

    Args:
        embeddings (np.ndarray): 2D array of feedback embeddings.
        n_clusters (int): Number of clusters to create.

    Returns:
        np.ndarray: Array of cluster label integers (one per feedback record).
    """
    if len(embeddings) < 3:
        return np.zeros(len(embeddings), dtype=int), 1

    cluster_count = n_clusters or find_optimal_clusters(embeddings)
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    return model.fit_predict(embeddings), cluster_count


def group_by_cluster(df, labels: np.ndarray) -> Dict[int, List[str]]:
    """
    Group feedback text by their assigned cluster label.

    Args:
        df: Preprocessed feedback DataFrame.
        labels (np.ndarray): Cluster label for each feedback record.

    Returns:
        Dict[int, List[str]]: Mapping of cluster_id → list of feedback strings.
    """
    return {
        int(cluster): group["feedback"].astype(str).tolist()
        for cluster, group in df.assign(cluster=labels).groupby("cluster")
    }


def find_optimal_clusters(embeddings: np.ndarray, max_k: int = 8) -> int:
    """
    Use silhouette score to suggest the optimal number of clusters.

    Args:
        embeddings (np.ndarray): Feedback embeddings.
        max_k (int): Maximum number of clusters to evaluate.

    Returns:
        int: Suggested optimal cluster count.
    """
    sample_count = len(embeddings)
    max_valid_k = min(max_k, sample_count - 1)
    if max_valid_k < 2:
        return 1

    best_k = 2
    best_score = -1.0
    for candidate_k in range(2, max_valid_k + 1):
        labels = KMeans(n_clusters=candidate_k, random_state=42, n_init=10).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k = candidate_k
            best_score = score
    return best_k


def generate_theme_names(df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    """Create readable names from dominant product areas and feedback terms."""
    names = {}
    used_names = set()
    stop_words = STOP_WORDS | {"app", "feature", "product", "screen", "also", "every", "time"}
    labeled = df.assign(cluster=labels)
    for cluster, group in labeled.groupby("cluster"):
        areas = group["product_area"].dropna().astype(str)
        if not areas.empty:
            area, count = Counter(areas).most_common(1)[0]
            if count >= max(2, len(group) // 2) and area not in used_names:
                names[int(cluster)] = area
                used_names.add(area)
                continue
        words = " ".join(group["cleaned_feedback"].astype(str)).split()
        terms = [word for word in words if len(word) > 2 and word not in stop_words]
        common_terms = [term.title() for term, _ in Counter(terms).most_common(2)]
        candidate = " / ".join(common_terms) or f"Theme {int(cluster) + 1}"
        if candidate in used_names:
            candidate = f"{candidate} / {int(cluster) + 1}"
        names[int(cluster)] = candidate
        used_names.add(candidate)
    return names
