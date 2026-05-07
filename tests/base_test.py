from ldu_decomposition import solve, ldu_decomposition, ldu_decomposition_plus
import scipy
import numpy as np
import time
import os

def generateMatrix(shape:int):
    A = scipy.linalg.hilbert(shape)
    x = np.random.rand(shape)
    return A, A @ x

def checkResults(A: np.ndarray, b: np.ndarray, path: os.PathLike):
    epsilon = 1e-300
    arr = np.empty(4, dtype=[
        ("shape", "i2"),
        ("decomposing function name", "U25"),
        ("solving function name", "U20"),
        ("accuracy of computation", "f8"),
        ("contextualized accuracy of computation", "f8"),
        ("decomposing time", "f8"),
        ("solving time", "f8"),
        ("total time", "f8")])
    start_time1 = time.perf_counter()
    x_scipy = scipy.linalg.solve(A, b)
    end_time1 = time.perf_counter() - start_time1
    arr[0] = (A.shape[0], "scipy.linalg.solve", "scipy.linalg.solve", np.linalg.norm(A @ x_scipy - b), np.linalg.norm((A @ x_scipy - b)/(b + epsilon)), 0, end_time1, end_time1)

    time.sleep(0)

    start_time2 = time.perf_counter()
    x_numpy = np.linalg.solve(A, b)
    end_time2 = time.perf_counter() - start_time2
    arr[1] = (A.shape[0], "np.linalg.solve", "np.linalg.solve", np.linalg.norm(A @ x_numpy - b), np.linalg.norm((A @ x_numpy - b)/(b + epsilon)), 0, end_time2, end_time2)

    time.sleep(0)

    funcs = [(ldu_decomposition_plus, solve), (ldu_decomposition, solve)]
    for i in range(len(funcs)):
        func1, func2 = funcs[i]

        if func1.__name__ == "ldu_decomposition_plus":
            start_time3 = time.perf_counter()
            P, Q, L, D, U = func1(A, path)
            end_time3 = time.perf_counter() - start_time3
        else:
            start_time3 = time.perf_counter()
            P, Q, L, D, U = func1(A)
            end_time3 = time.perf_counter() - start_time3

        time.sleep(0)

        start_time4 = time.perf_counter()
        x_eval = func2(P, Q, L, D, U, b)
        end_time4 = time.perf_counter() - start_time4

        arr[2 + i] = (A.shape[0], func1.__name__, func2.__name__, np.linalg.norm(A @ x_eval - b), np.linalg.norm((A @ x_eval - b)/(b + epsilon)), end_time3, end_time4, end_time3 + end_time4)

        time.sleep(0) 
    return arr


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_logs = os.path.join(project_root, "logs", "logs.bin")
    path_to_benchmarks = os.path.join(project_root, "logs", "benchmarks.bin")
    n = 10

    matrices = [generateMatrix(i) for i in [50, 100, 500, 1000, 2000]]


    result = None
    for i in range(len(matrices)):
        for k in range(n):
            a = checkResults(*matrices[i], path_to_logs)
            if result is not None:
                result = np.concatenate((result, a))
            else:
                result = a
    result.tofile(path_to_benchmarks)