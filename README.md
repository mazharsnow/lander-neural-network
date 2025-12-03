# Lander Neural Network Controller

This project is a coursework assignment implementing a neural network controller
for a 2D lander game. The game randomly generates a landing pad and unsafe
terrain; the goal is to safely land the ship on the pad.

## Project Structure

- `Main.py` / `GameLoop.py` – game entry point and main loop (Pygame).
- `DataCollection.py` – logs gameplay data (x_target, y_target, vel_y, vel_x).
- `PrepareData.py` – cleans, normalises, and splits data into train/validation.
- `NeuralNetwork.py` – custom feed-forward MLP (Neuron + MLP classes).
- `TrainNN.py` – offline training (~100 epochs) and RMSE reporting.
- `NeuralNetHolder.py` – loads trained model and connects it to the game.
- `Matlab/` – MATLAB scripts for analysing training:
  - `learning_rate.m` – RMSE & learning parameter (mu) graphs
  - `momentum.m` – effect of momentum on RMSE
  - `lambda_graph.m` – effect of regularisation (lambda) on RMSE

## How to Run

1. Create and activate a virtual environment (optional).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
