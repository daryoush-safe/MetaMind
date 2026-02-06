from .perceptron import train_perceptron_tool
from .perceptron import inference_perceptron_tool
from .mlp import train_mlp_tool
from .mlp import inference_mlp_tool
from .som import train_som_tool
from .som import inference_som_tool
from .hopfield import train_hopfield_tool
from .hopfield import inference_hopfield_tool
from .fuzzy import train_fuzzy_tool
from .fuzzy import inference_fuzzy_tool
from .ga import ga_tool
from .gp import train_gp_tool
from .gp import inference_gp_tool
from .aco import aco_tool
from .pso import pso_tool
from .tsp import read_tsp_file
from .csv import read_and_preprocess_csv


ALL_TOOLS = [
    train_perceptron_tool,
    inference_perceptron_tool,
    train_mlp_tool,
    inference_mlp_tool,
    train_som_tool,
    inference_som_tool,
    train_hopfield_tool,
    inference_hopfield_tool,
    train_fuzzy_tool,
    inference_fuzzy_tool,
    ga_tool,
    train_gp_tool,
    inference_gp_tool,
    aco_tool,
    pso_tool,
    read_tsp_file,
    read_and_preprocess_csv,
]