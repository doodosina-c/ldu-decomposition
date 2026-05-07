from ldu_decomposition import solve, ldu_decomposition_plus
import numpy as np
import scipy
import time
import os

def generateMatrix(shape:int):
    return scipy.linalg.hilbert(shape)

def getResults(shape, n, path: os.PathLike):
    for i in range(n):
        A = generateMatrix(shape)

        time.sleep(0)

        P, Q, L, D, U = ldu_decomposition_plus(A, path)


if __name__ == "__main__":
    n = 10
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "logs", "logs.bin")

    for i in [50, 100, 500, 1000, 2000]:
        getResults(i, n, path)