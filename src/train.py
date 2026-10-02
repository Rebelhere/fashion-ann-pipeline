import csv
import os

import numpy as np
import tensorflow as tf
import yaml

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    # load processed data
    x_train = np.load(os.path.join(PROCESSED_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    x_val = np.load(os.path.join(PROCESSED_DIR, "x_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    # ANN: Flatten -> Dense(ReLU) -> Dropout -> Dense(10, softmax)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(params["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(params["dropout_rate"]),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "model.h5"))

    # save training history as CSV
    keys = list(history.history.keys())
    with open(os.path.join(MODEL_DIR, "history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(len(history.history[keys[0]])):
            writer.writerow([i + 1] + [history.history[k][i] for k in keys])

    print("Saved model and history to", MODEL_DIR)


if __name__ == "__main__":
    main()