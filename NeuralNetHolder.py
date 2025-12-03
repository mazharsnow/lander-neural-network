# NeuralNetHolder.py
import json
from NeuralNetwork import MLP


class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.model = None
        self.in_min = None
        self.in_max = None
        self.out_min = None
        self.out_max = None
        self.loaded = False

    def _load_model_and_norm(self):
        if self.loaded:
            return

        # 1) Load trained MLP
        self.model = MLP.load("trained_model.json")

        # 2) Load original min/max from PrepareData.py
        with open("normalization.json", "r") as f:
            norm = json.load(f)

        # norm has keys: "columns", "min", "max"
        mins = norm["min"]   # [x_target, y_target, vel_y, vel_x]
        maxs = norm["max"]

        # Inputs are [x_target, y_target]
        self.in_min = [mins[0], mins[1]]
        self.in_max = [maxs[0], maxs[1]]

        # Outputs are [vel_x, vel_y]
        # Original order: index 2 = vel_y, index 3 = vel_x
        self.out_min = [mins[3], mins[2]]  # [vel_x_min, vel_y_min]
        self.out_max = [maxs[3], maxs[2]]  # [vel_x_max, vel_y_max]

        self.loaded = True

    def _scale_inputs(self, inputs):
        scaled = []
        for v, mn, mx in zip(inputs, self.in_min, self.in_max):
            if mx == mn:
                scaled.append(0.0)
            else:
                scaled.append((v - mn) / (mx - mn))
        return scaled

    def _unscale_outputs(self, outputs):
        unscaled = []
        for v, mn, mx in zip(outputs, self.out_min, self.out_max):
            unscaled.append(v * (mx - mn) + mn)
        return unscaled

    def predict(self, input_row):
        """
        input_row: 'x_target,y_target' from DataCollection.get_input_row
        returns: [pred_vel_x, pred_vel_y] in real game units
        """
        self._load_model_and_norm()

        try:
            parts = [float(p) for p in input_row.strip().split(",")]
        except ValueError:
            return [0.0, 0.0]

        if len(parts) < 2:
            return [0.0, 0.0]

        x_target, y_target = parts[:2]
        inputs = [x_target, y_target]

        # 1) scale inputs to [0,1]
        x_scaled = self._scale_inputs(inputs)

        # 2) NN prediction in scaled space
        out_scaled = self.model.predict(x_scaled)

        # 3) unscale to real velocities
        vel_x, vel_y = self._unscale_outputs(out_scaled)

        return [vel_x, vel_y]
