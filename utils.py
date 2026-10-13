"""
helper functions
"""

import os
import numpy as np
from datetime import datetime


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def set_random_seed(seed=42):
    np.random.seed(seed)


def print_data_summary(data):
    print("\nData summary")

    print(f"Training samples:   {len(data['X_train'])}")
    print(f"Validation samples: {len(data['X_val'])}")
    print(f"Test samples:       {len(data['X_test'])}")
    print(f"Features:           {data['n_features']}")
    print(f"Classes:            {data['n_classes']}")

    print("\nClass names:")
    print(list(data["class_names"]))

    print("\nFeatures used:")
    print(data["feature_names"])


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes = seconds // 60
    secs = seconds % 60
    return f"{int(minutes)}m {secs:.1f}s"


class Timer:
    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        end = datetime.now()
        elapsed = (end - self.start).total_seconds()
        print(f"{self.name} finished in {format_time(elapsed)}")


if __name__ == "__main__":
    print("Utils module")