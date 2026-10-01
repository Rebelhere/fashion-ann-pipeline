# Fashion-MNIST ANN Pipeline

End-to-end ML versioning project using Git, DVC and Google Drive.

A fully connected neural network (ANN) classifies Fashion-MNIST images
into 10 clothing categories. The pipeline is versioned with Git (code)
and DVC (data, models, metrics).

## Pipeline stages
1. prepare - download Fashion-MNIST and save raw arrays
2. preprocess - normalize pixels and split train/validation
3. train - train the ANN using hyperparameters from params.yaml
4. evaluate - compute test metrics and a confusion matrix

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install tensorflow pyyaml scikit-learn matplotlib "dvc[gdrive]"
```

## Status
Work in progress. This project uses Fashoin-MNIST from Keras.