import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_logfile(path: os.PathLike) -> pd.DataFrame:
    data = pd.DataFrame(
    np.fromfile(path,
    dtype=np.dtype([
        ("ts", "S32"),
        ("salg", np.int32),
        ("miarowind", np.int32),
        ("miacolind", np.int32),
        ("pivot_ratio", np.float64),
        ("miaresccl", np.float64),
        ("amresccl", np.float64),
        ("miaresup", np.float64),
        ("amresup", np.float64),
        ("miaresn", np.float64),
        ("amresn", np.float64)
    ])))
    data["ts"] = data["ts"].astype("str")
    return data


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(project_root, "logs", "logs.bin")

    shape, runs = 1_000, 10

    df = read_logfile(path)[6499:16499]
    c = shape - df.iloc[:-2, 1].unique() - 1
    comdis = (df["amresccl"].to_numpy().reshape(runs, shape) + df["amresup"].to_numpy().reshape(runs, shape) * c + df["amresn"].to_numpy().reshape(runs, shape)) * c
    comdis /= comdis.sum(axis=1)[:, np.newaxis]

    x = np.repeat(np.arange(1, runs + 1), shape)
    y = np.tile(df["salg"].unique(), runs)
    z = np.zeros_like(x)

    dx = np.ones_like(x) * 0.5
    dy = np.ones_like(x) * 0.5
    dz = comdis.flatten()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.bar3d(x, y, z, dx, dy, dz)

    plt.style.use('_mpl-gallery')
    ax.set_xlabel('Run number')
    ax.set_ylabel('Iteration')
    ax.set_zlabel('Residual share')

    plt.show()