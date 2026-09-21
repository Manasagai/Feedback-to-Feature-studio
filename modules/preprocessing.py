"""Small, conservative text-cleaning helpers for customer feedback."""

import re
import pandas as pd


def load_feedback(filepath: str) -> pd.DataFrame:
    """
    Load the feedback CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Raw feedback data.
    """
    return pd.read_csv(filepath)


def clean_text(text: str) -> str:
    """
    Clean and normalize a single feedback string.

    Args:
        text (str): Raw feedback text.

    Returns:
        str: Cleaned feedback text.
    """
    if pd.isna(text):
        return ""
    cleaned = str(text).lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def preprocess_feedback(text) -> str:
    """
    Clean one feedback value while preserving its semantic words.

    Args:
        text: Raw feedback text or a missing value.

    Returns:
        str: Lowercase, whitespace-normalized, lightly de-punctuated text.
    """
    return clean_text(text)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the feedback frame with cleaned text."""
    result = df.copy()
    result["cleaned_feedback"] = result["feedback"].map(preprocess_feedback)
    return result
