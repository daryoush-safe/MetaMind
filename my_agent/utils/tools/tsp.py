import math
from typing import List, Optional, Any, Dict
from pydantic import Field, BaseModel
from langchain_core.tools import tool


class TSPDatasetInput(BaseModel):
    file_path: str = Field(description="Path to the dataset file (TSPLIB-format .tsp)")


@tool(args_schema=TSPDatasetInput)
def read_tsp_file(file_path: str) -> Dict[str, Any]:
    """Read a TSPLIB-format .tsp file and return city coordinates + distance matrix.

    Parameters
    ----------
    file_path : str
        Path to the .tsp file (e.g. "data/berlin52.tsp").

    Returns
    -------
    dict with keys:
        name, dimension, coordinates (list of [x, y]),
        distance_matrix (2-D list), comment, edge_weight_type.
    """
    name = ""
    comment = ""
    dimension = 0
    edge_weight_type = ""
    coordinates: List[List[float]] = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    reading_coords = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "EOF":
            if reading_coords:
                break
            continue

        if stripped == "NODE_COORD_SECTION":
            reading_coords = True
            continue

        if reading_coords:
            parts = stripped.split()
            if len(parts) >= 3:
                coordinates.append([float(parts[1]), float(parts[2])])
        else:
            if stripped.startswith("NAME"):
                name = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("COMMENT"):
                comment = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("DIMENSION"):
                dimension = int(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = stripped.split(":", 1)[1].strip()

    if not dimension:
        dimension = len(coordinates)

    # Build Euclidean distance matrix
    n = len(coordinates)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist_matrix[i][j] = round(d, 4)
            dist_matrix[j][i] = round(d, 4)

    return {
        "name": name,
        "comment": comment,
        "dimension": dimension,
        "edge_weight_type": edge_weight_type,
        "coordinates": coordinates,
        "distance_matrix": dist_matrix,
    }