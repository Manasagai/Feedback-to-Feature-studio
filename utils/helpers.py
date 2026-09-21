"""
utils/helpers.py
================
Common helper / utility functions shared across all modules.

Add reusable utilities here to keep module files clean and focused.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List


def load_env_variable(key: str, default: Any = None) -> Any:
    """
    Safely read an environment variable with an optional default value.

    Args:
        key (str): Environment variable name.
        default (Any): Value to return if the variable is not set.

    Returns:
        Any: The environment variable value, or the default.
    """
    return os.environ.get(key, default)


def validate_dataframe_columns(df, required_columns: List[str]) -> bool:
    """
    Check that a DataFrame contains all required columns.

    Args:
        df: pandas DataFrame to validate.
        required_columns (List[str]): List of expected column names.

    Returns:
        bool: True if all columns are present, False otherwise.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"[helpers] Missing columns: {missing}")
        return False
    return True


def save_json(data: Dict, filepath: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data (Dict): Data to serialize.
        filepath (str): Destination file path.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[helpers] Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        Dict: Parsed JSON data.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_timestamp() -> str:
    """
    Return the current date and time as a formatted string.

    Returns:
        str: Timestamp in 'YYYY-MM-DD HH:MM:SS' format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 120) -> str:
    """
    Truncate a string to a maximum length for display purposes.

    Args:
        text (str): Input text.
        max_length (int): Maximum allowed character count.

    Returns:
        str: Truncated text with ellipsis if needed.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."
