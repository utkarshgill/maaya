"""
Utility functions: configuration parsing, math, and logging.
"""

import json
import math

def load_config(path="config.json"):
    """
    Load JSON configuration and return a dict.
    """
    with open(path, "r") as f:
        return json.load(f)

def log(message):
    """
    Simple logger stub.
    """
    print(f"[LOG] {message}")

def vector_add(a, b):
    """
    Add two vectors element-wise.
    """
    return [x + y for x, y in zip(a, b)]

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    """
    # TODO: implement quaternion multiplication
    return [0.0, 0.0, 0.0, 1.0] 