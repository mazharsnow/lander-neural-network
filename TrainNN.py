# TrainNN.py (using pre-scaled CSVs)
import csv
import math
import random
from NeuralNetwork import MLP

TRAIN_FILE = "lander_train.csv"
VAL_FILE = "lander_val.csv"
MODEL_FILE = "trained_model.json"


def load_scaled_dataset(filename):
    dataset = []
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_t = float(row["x_target"])
            y_t = float(row["y_target"])
            vel_y = float(row["vel_y"])
            vel_x = float(row["vel_x"])

            # Inputs: x_target, y_target (already 0–1)
            x = [x_t, y_t]
            # Targets: vel_x, vel_y (already 0–1)
            y = [vel_x, vel_y]

            dataset.append((x, y))
    return dataset


def rmse(model, dataset):
    if not dataset:
        return 0.0
    sse = 0.0
    for x, y in dataset:
        pred = model.predict(x)
        for p, t in zip(pred, y):
            sse += (t - p) ** 2
    return math.sqrt(sse / (len(dataset) * 2))


def main():
    random.seed(0)

    train_set = load_scaled_dataset(TRAIN_FILE)
    val_set = load_scaled_dataset(VAL_FILE)
    print("Train samples:", len(train_set), "| Val samples:", len(val_set))

    mlp = MLP(n_inputs=2, n_hidden=10, n_outputs=2)
    mlp.train(train_set, n_epochs=200, learning_rate=0.05, verbose=True)

    val_rmse = rmse(mlp, val_set)
    print(f"Validation RMSE (scaled): {val_rmse:.6f}")

    mlp.save(MODEL_FILE)
    print("Saved model to", MODEL_FILE)


if __name__ == "__main__":
    main()
