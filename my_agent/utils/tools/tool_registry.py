from .perceptron import perceptron_tool
from .mlp import mlp_tool
from .som import som_tool
from .hopfield import hopfield_recall
from .fuzzy import fuzzy_tool
from .ga import ga_tool
from .gp import gp_tool
from .aco import aco_tool
from .pso import pso_tool


ALL_TOOLS = [
    perceptron_tool,
    mlp_tool,
    som_tool,
    hopfield_recall,
    fuzzy_tool,
    ga_tool,
    gp_tool,
    aco_tool,
    pso_tool,
]
