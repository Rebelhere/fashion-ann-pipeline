import json
import os

import matplotlib

matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "model.h5")

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    x_test = np.load(os.path.join(PROCESSED_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    # test loss and accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # confusion matrix
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (test set)")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=100)

    # write metrics to the project root
    metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Test loss:", metrics["test_loss"])
    print("Test accuracy:", metrics["test_accuracy"])


if __name__ == "__main__":
    main()