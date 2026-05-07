import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os


def load_data(path: os.PathLike) -> pd.DataFrame:
    data = pd.DataFrame(
    np.fromfile(path,
    dtype=np.dtype([
        ("shape", np.int16),
        ("decomposing function name", "U25"),
        ("solving function name", "U20"),
        ("accuracy of computation", np.float64),
        ("contextualized accuracy of computation", np.float64),
        ("decomposing time", np.float64),
        ("solving time", np.float64),
        ("total time", np.float64)]
    )))
    data["decomposing function name"] = data["decomposing function name"].astype("str")
    data["solving function name"] = data["solving function name"].astype("str")
    return data

def color(x: str) -> str:
    match x:
        case "scipy.linalg.solve":
            return "#4800ff"
        case "np.linalg.solve":
            return "#0026ff"
        case "ldu_decomposition_plus + solve":
            return "#ff0000"
        case "ldu_decomposition + solve":
            return "#ff6A00"

if __name__ == "__main__":
    shape = 1000
    epsilon = 1e-300
    project_root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(project_root, "logs", "logs.bin")
    
    cols = ["accuracy of computation", "contextualized accuracy of computation", "total time"]
    df = load_data(path)
    df.insert(0, "name", df["decomposing function name"].combine(df["solving function name"], lambda x, y: x + " + " + y if x != y else x))
    del df["decomposing function name"], df["solving function name"]

    val = df.loc[df["shape"] == shape, cols].map(lambda x: -np.log(x + epsilon))
    points = ((val - val.min(axis=0))/(val.max(axis=0) - val.min(axis=0))).to_numpy()
    colors = df.loc[df["shape"] == shape, "name"].apply(color)

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],c=colors.to_numpy())

    plt.style.use("_mpl-gallery")
    ax.set_xlabel("accuracy of computation")
    ax.set_ylabel("contextualized accuracy of computation")
    ax.set_zlabel("total time")
    ax.legend(handles=[mpatches.Patch(color=color(name), label=name) for name in df["name"].unique()])

    plt.show()