# Lander Neural Network Controller

This project is a coursework assignment implementing a neural network controller
for a 2D lander game. The game randomly generates a landing pad and unsafe
terrain; the goal is to safely land the ship on the pad without crashing.

The workflow is:

1. **Collect gameplay data** while controlling the lander manually.
2. **Preprocess and normalise** the data in Python.
3. **Train a custom feed-forward MLP** (implemented from scratch) for ~100 epochs.
4. **Load the trained network inside the game** and let it control the lander.
5. Optionally, **analyse training behaviour in MATLAB** (RMSE, learning parameter, momentum, lambda).

---

## Project Structure

Core Python/game files:

- `Main.py`  
  Entry point: launches the game and main menu (Pygame window).

- `GameLoop.py`  
  Main game loop: updates physics, reads controller input, draws graphics, and
  calls the neural network in “Neural Net” mode.

- `Controller.py`  
  Encapsulates the lander controls (thrust, left/right turning).

- `Lander.py`, `Surface.py`, `Vector.py`, `GameLogic.py`, `CollisionUtility.py`, `EventHandler.py`,  
  `MainMenu.py`, `ResultMenu.py`  
  Core game logic, physics, UI and rendering.

Data and neural network files:

- `DataCollection.py`  
  Logs gameplay data in **Data Collection** mode. For each frame it records:
  - `x_target`, `y_target` (distance from lander to target pad)
  - `vel_y`, `vel_x` (current vertical/horizontal velocities)  
  Data is appended to `ce889_dataCollection.csv` when the game closes.

- `PrepareData.py`  
  Offline preprocessing script:
  - Loads `ce889_dataCollection.csv`
  - Cleans invalid values
  - Scales all columns to **[0, 1]** using min–max normalisation
  - Splits into training/validation sets:
    - `lander_train.csv` (80% train)
    - `lander_val.csv` (20% validation)
  - Saves original min/max values to `normalization.json`

- `NeuralNetwork.py`  
  **Custom feed-forward backpropagation MLP** (no external NN libraries):
  - `Neuron` class (weights, bias, sigmoid activation, delta)
  - `MLP` class with:
    - 2 input neurons (`x_target`, `y_target`)
    - 1 hidden layer (configurable, e.g. 6 neurons)
    - 2 output neurons (desired `vel_x`, `vel_y`)
    - Methods for feedforward, backprop, weight updates
    - `save()` / `load()` using JSON (`trained_model.json`)

- `TrainNN.py`  
  Offline training script:
  - Loads `lander_train.csv` / `lander_val.csv`
  - Builds training pairs:
    - Inputs: `[x_target, y_target]` (scaled)
    - Targets: `[vel_x, vel_y]` (scaled)
  - Trains a **2–6–2** MLP for ~100 epochs (as per assignment spec)
  - Prints training RMSE per epoch and final validation RMSE
  - Saves the trained model to `trained_model.json`

- `NeuralNetHolder.py`  
  Runtime wrapper that:
  - Loads `trained_model.json` and `normalization.json`
  - Scales incoming inputs `[x_target, y_target]` to [0, 1]
  - Runs `MLP.predict()` to obtain `[vel_x, vel_y]` in scaled space
  - Unscales these back to real velocities
  - Returns the predicted velocities to `GameLoop`  
  `GameLoop` then converts these desired velocities into **thruster** and
  **turning** commands.

Other files:

- `Sprites/`  
  Contains background and lander images used by Pygame.

- `requirements.txt`  
  Python dependencies for this project.

- `Matlab/`  
  MATLAB analysis scripts:
  - `learning_rate.m` – trains a 2–6–2 network using `trainlm` and plots:
    - RMSE over epochs (train/val/test)
    - Learning parameter μ (Levenberg–Marquardt)
  - `momentum.m` – uses `traingdm` to compare different momentum values and
    their effect on RMSE.
  - `lambda_graph.m` – uses `trainlm` to compare different regularisation
    strengths (`net.performParam.regularization`) and their effect on RMSE.
  - `.fig` / `.jpg` – saved figures for inclusion in the report/presentation.

---

## Requirements

- **Python 3.11** (or compatible 3.x)
- **pip** (Python package manager)
- **Virtual environment** support (recommended)
- **MATLAB** with Neural Network / Deep Learning Toolbox (only for the `Matlab/` analysis; not required to run the game)

Python packages (see `requirements.txt`):

- `pygame`
- `pandas`
- `scikit-learn`

---

## Setup & Installation (Python)

The steps below assume Windows + PowerShell, but the commands are easy to adapt for macOS/Linux.

### 1. Clone the repository
```bash
git clone https://github.com/mazharsnow/lander-neural-network.git
cd lander-neural-network
```

### **2. Create a virtual environment**
```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate
```

**On Windows (cmd):**
```cmd
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### 4. Install required libraries
```bash
pip install -r requirements.txt
```

This installs:
* `pygame`
* `pandas`
* `scikit-learn`

## Workflow: Data → Training → Game

### Step 1 – Collect gameplay data

1. Make sure your virtual environment is activated.
2. Run the game:
```bash
   python3 Main.py
```
3. In the game menu, select **Data Collection mode**.
4. Play several episodes manually:
   * Use **Up** to thrust, **Left/Right** to rotate/steer.
   * Try to land on the randomly positioned landing pad from various directions.
5. Each time you close the game, your gameplay data is appended to:
```
   ce889_dataCollection.csv
```

Collect enough data (several complete landings) so that the network has meaningful examples to learn from.

### Step 2 – Prepare and normalise the data

After collecting data:
```bash
python PrepareData.py
```

This script will:
* Load `ce889_dataCollection.csv`
* Clean invalid rows (NaN/inf)
* Scale all columns to [0, 1] with min–max normalisation
* Split into training/validation sets

**Generated files:**
* `lander_data_scaled.csv` – full scaled dataset
* `lander_train.csv` – 80% of the data for training
* `lander_val.csv` – 20% for validation
* `normalization.json` – original min/max values for each feature

You should see console output similar to:
```
Loaded XXXXX rows from ce889_dataCollection.csv
Removed 0 rows with NaN/inf
Saved scaled data to lander_data_scaled.csv
Train rows: NNNNN | Val rows: MMMM
Saved train data to lander_train.csv
Saved validation data to lander_val.csv
Saved normalisation parameters to normalization.json
```

### Step 3 – Train the neural network (offline)

Train the custom MLP:
```bash
python TrainNN.py
```

**What this does:**
* Loads `lander_train.csv` and `lander_val.csv`
* Constructs:
  * Inputs: `[x_target, y_target]` (scaled)
  * Targets: `[vel_x, vel_y]` (scaled)
* Creates a 2–6–2 MLP (2 inputs, 6 hidden, 2 outputs)
* Trains for ~100 epochs using backpropagation
* Prints per-epoch RMSE and final validation RMSE

**Example output (truncated):**
```
Train samples: 82585 | Val samples: 20647
Epoch 1/100, RMSE: 0.195162
...
Epoch 100/100, RMSE: 0.146419
Validation RMSE (scaled): 0.146148
Saved model to trained_model.json
```

`TrainNN.py` saves the trained network to:
* `trained_model.json` – used later by the game in Neural Net mode.

### Step 4 – Run the game with the trained NN

With `trained_model.json` and `normalization.json` present in the project root, run:
```bash
python Main.py
```

**In the game:**
1. Select **Neural Net mode** from the main menu.
2. The game will:
   * Create a `NeuralNetHolder` instance.
   * Load `trained_model.json` and `normalization.json`.
   * For each frame:
     * Compute `x_target` and `y_target`.
     * Scale them and pass them to the MLP.
     * Receive predicted velocities `[vel_x, vel_y]`.
     * Convert these into thruster and turning commands.
3. You should see the lander attempt to land automatically on the pad, using the policy learned from your gameplay data.

## Summary

1. **Collect data** by playing manually in Data Collection mode
2. **Prepare data** using `PrepareData.py`
3. **Train the network** using `TrainNN.py`
4. **Run the trained agent** in Neural Net mode

Enjoy watching your neural network learn to land! 🚀
