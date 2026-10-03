import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    test_size = params["preprocess"]["test_size"]
    seed = params["preprocess"]["seed"]

    # load raw arrays
    x_train = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    # normalize using standardization (zero mean, unit variance)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    mean, std = x_train.mean(), x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # split training data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=test_size, random_state=seed
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"), x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print("Saved processed data to", PROCESSED_DIR)
    print("train:", x_train.shape, "val:", x_val.shape, "test:", x_test.shape)


if __name__ == "__main__":
    main()