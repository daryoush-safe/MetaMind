import numpy as np
from typing import Literal
from pydantic import Field
from langchain_core.tools import tool


class FuzzyController:
    def __init__(self, n_membership_functions=3, membership_type="triangular", 
                 defuzzification="centroid", rule_generation="wang_mendel"):
        self.n_mf = n_membership_functions
        self.mf_type = membership_type
        self.defuz_method = defuzzification
        self.rule_gen_method = rule_generation
        self.rules = []
        self.var_ranges = []  # Min/Max for each feature

    def _membership(self, x, center, width):
        """Calculates membership degree mu(x)."""
        if self.mf_type == "triangular":
            # trimf: max(min((x-a)/(b-a), (c-x)/(c-b)), 0)
            a, b, c = center - width, center, center + width
            if width == 0: return 1.0 if x == center else 0.0
            return max(min((x - a) / (b - a), (c - x) / (c - b)), 0)
        elif self.mf_type == "gaussian":
            # gaussmf: exp(-(x - c)^2 / (2*sigma^2))
            sigma = width / 2.0  # approximate width mapping
            return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        elif self.mf_type == "trapezoidal":
            # trapmf: simplified symmetric trapezoid
            a, b, c, d = center - width, center - width/2, center + width/2, center + width
            if width == 0: return 1.0 if x == center else 0.0
            return max(min((x - a) / (b - a), 1, (d - x) / (d - c)), 0)
        return 0.0

    def fit(self, X, y):
        """Generates rules using Wang-Mendel method."""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        data = np.hstack((X, y))
        n_features = data.shape[1]
        
        # 1. Determine ranges and grid for inputs/outputs
        self.var_ranges = []
        self.mf_params = [] # List of list of (center, width)
        
        for i in range(n_features):
            d_min, d_max = np.min(data[:, i]), np.max(data[:, i])
            self.var_ranges.append((d_min, d_max))
            
            step = (d_max - d_min) / (self.n_mf - 1)
            params = []
            for m in range(self.n_mf):
                center = d_min + m * step
                params.append((center, step))
            self.mf_params.append(params)

        # 2. Generate Rules from Data
        if self.rule_gen_method == "wang_mendel":
            raw_rules = []
            for row in data:
                # Find fuzzy region with max membership for each variable
                rule_indices = []
                degree = 1.0
                for i, val in enumerate(row):
                    mu_vals = [self._membership(val, p[0], p[1]) for p in self.mf_params[i]]
                    idx = np.argmax(mu_vals)
                    rule_indices.append(idx)
                    degree *= mu_vals[idx] # Rule strength
                
                # Format: (input_indices_tuple, output_index, degree)
                raw_rules.append((tuple(rule_indices[:-1]), rule_indices[-1], degree))

            # 3. Handle Conflicting Rules (keep max degree)
            rule_dict = {}
            for inputs, output, deg in raw_rules:
                if inputs in rule_dict:
                    if deg > rule_dict[inputs]['degree']:
                        rule_dict[inputs] = {'out': output, 'degree': deg}
                else:
                    rule_dict[inputs] = {'out': output, 'degree': deg}
            
            self.rules = rule_dict

    def predict(self, X):
        predictions = []
        for row in X:
            # 1. Fuzzify Inputs & Evaluate Rules
            # Aggregation: Maximize firing strength for output regions
            # Initialize aggregate output fuzzy set (discretized for centroid calc)
            out_min, out_max = self.var_ranges[-1]
            y_points = np.linspace(out_min, out_max, 100)
            agg_output = np.zeros_like(y_points)
            
            for rule_in, rule_meta in self.rules.items():
                out_idx = rule_meta['out']
                
                # Calculate Firing Strength (Min operator for AND)
                mu_vals = []
                for i, r_idx in enumerate(rule_in):
                    c, w = self.mf_params[i][r_idx]
                    mu_vals.append(self._membership(row[i], c, w))
                firing_strength = np.min(mu_vals)
                
                if firing_strength > 0:
                    # Implication (Min operator): Clip output MF
                    c_out, w_out = self.mf_params[-1][out_idx]
                    rule_output = np.array([min(firing_strength, self._membership(y, c_out, w_out)) for y in y_points])
                    
                    # Aggregation (Max operator)
                    agg_output = np.maximum(agg_output, rule_output)

            # 2. Defuzzification
            if np.sum(agg_output) == 0:
                predictions.append((out_min + out_max) / 2) # Default fallback
                continue

            if self.defuz_method == "centroid":
                numerator = np.sum(y_points * agg_output)
                denominator = np.sum(agg_output)
                predictions.append(numerator / denominator)
            elif self.defuz_method == "mom": # Mean of Maxima
                max_val = np.max(agg_output)
                indices = np.where(agg_output == max_val)[0]
                predictions.append(np.mean(y_points[indices]))
            else:
                # Fallback to centroid
                numerator = np.sum(y_points * agg_output)
                denominator = np.sum(agg_output)
                predictions.append(numerator / denominator)
                
        return np.array(predictions)


@tool
def fuzzy_tool(
    n_membership_functions: Literal[3, 5, 7] = Field(default=3),
    membership_type: Literal["triangular", "gaussian", "trapezoidal"] = Field(default="triangular"),
    defuzzification: Literal["centroid", "bisector", "mom", "som", "lom"] = Field(default="centroid"),
    rule_generation: Literal["wang_mendel", "manual"] = Field(default="wang_mendel"),
):
    """Creates a Fuzzy Logic Controller tool with specified hyperparameters."""
    pass