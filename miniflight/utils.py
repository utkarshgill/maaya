"""
Firmware utilities: configuration loader
"""
import json


def load_config(path: str = "config.json") -> dict:
    """
    Load JSON configuration and return a dict, or empty dict if not found.
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} 