cimport cython
cimport numpy as cnp
import numpy as np
from datetime import datetime
import time
import random
from libc.string cimport strncpy
from libc.math cimport fabs, fma
from libc.stdio cimport FILE, fopen, fwrite, fclose


cdef double _dot_product(double[::1] A, double[::1] B, int start, int stop):
        cdef:
            double dot_product = 0.0
            double c = 0.0
            int i
            double y, t

        for i in range(start, stop):
             y = fma(A[i], B[i], -c)

             t = dot_product + y
             c = (t - dot_product) - y
             dot_product = t

        return dot_product

cdef double _divide(double A, double B):
    cdef:
        double r, C
        int i

    C = A / B
    r = fma(-B, C, A)
    C = C + r / B

    return C


cdef double[::1] _source_solve(char[:, ::1] P, char[:, ::1] Q, double[:, ::1] L, double[::1] D, double[:, ::1] U, double[::1] b):
    cdef:
        int n = U.shape[0]
        double[::1] W = np.zeros(n)
        double[::1] Y = np.zeros(n)
        double[::1] Z = np.zeros(n)
        double[::1] x = np.zeros(n)
        double[::1] tilde_b = np.zeros(n)
        int i, j

    for i in range(n):
        for j in range(n):
            if P[i, j] == 1:
                   tilde_b[i] = b[j]
                   break

    for i in range(n):
        if i > 0:
           Y[i] = tilde_b[i] - _dot_product(L[i], Y, 0, i)
        else:
           Y[i] = tilde_b[i]

    for i in range(0, n):
        Z[i] = _divide(Y[i], D[i])

    for i in range(n - 1, -1, -1):
        if i < n - 1:
            W[i] = Z[i] - _dot_product(U[i], W, i + 1, n)
        else:
            W[i] = Z[i]

    for i in range(n):
        for j in range(n):
            if Q[i, j] == 1:
                   x[i] = W[j]
                   break

    return x

cdef struct Maxval:
    int rowind
    int colind
    double val

cdef Maxval _fmax(double[:, ::1] A, int salg):
    cdef:
         Maxval result
         int rowind = salg
         int colind = salg
         int n = A.shape[0]
         int i, j
         double maxval = fabs(A[salg, salg])

    for i in range(salg, n):
        for j in range(salg, n):
             if maxval < fabs(A[i, j]):
                 maxval = fabs(A[i, j])
                 rowind = i
                 colind = j

    result.rowind = rowind
    result.colind = colind
    result.val = maxval
    return result

cdef void _ldu_decomposition(char[:, ::1] P, char[:, ::1] Q, double[:, ::1] L, double[::1] D, double[:, ::1] U):
    cdef:
        int i, k, j
        int U_shape = U.shape[0]
        double f
        char b
        Maxval maxval

    for i in range(U_shape):
            if i != U_shape - 1:
                maxval = _fmax(U, i)

                if maxval.val == 0:
                    raise ValueError("The matrix is degenerate")
                if maxval.val < 1e-300:
                    raise ValueError("The matrix is close to degenerate")

                if maxval.val != fabs(U[i, i]):
                    if maxval.rowind != i:
                        for k in range(U_shape):
                            f = U[i, k]
                            U[i, k] = U[maxval.rowind, k]
                            U[maxval.rowind, k] = f

                        for k in range(U_shape):
                            b = P[i, k]
                            P[i, k] = P[maxval.rowind, k]
                            P[maxval.rowind, k] = b

                        for k in range(i):
                            f = L[i, k]
                            L[i, k] = L[maxval.rowind, k]
                            L[maxval.rowind, k] = f

                    if maxval.colind != i:
                        for k in range(U_shape):
                            f = U[k, i]
                            U[k, i] = U[k, maxval.colind]
                            U[k, maxval.colind] = f

                        for k in range(U_shape):
                            b = Q[k, i]
                            Q[k, i] = Q[k, maxval.colind]
                            Q[k, maxval.colind] = b

            D[i] = U[i, i]

            for k in range(i + 1, U_shape):
                L[k, i] = U[k, i] / D[i]

            U[i + 1:, i] = 0.0
            for k in range(i + 1, U_shape):
                for j in range(i + 1, U_shape):
                    U[k, j] = U[k, j] - L[k, i] * U[i, j]

            U[i, i] = 1.0
            for k in range(i + 1, U_shape):
                U[i, k] = U[i, k] / D[i]

cdef packed struct LogMSG:
    char ts[32]
    int salg
    int miarowind
    int miacolind
    double pivot_ratio

    double miaresccl
    double amresccl

    double miaresup
    double amresup

    double miaresn
    double amresn

cdef void _ldu_decomposition_plus(char[:, ::1] P, char[:, ::1] Q, double[:, ::1] L, double[::1] D, double[:, ::1] U, const char* base_path, const char* timestamp):
    cdef:
        int i, k, j
        int ptr = 0
        int U_shape = U.shape[0]
        double f, v, c1
        double c2 = 0.0
        char b
        Maxval maxval
        FILE* cfile = fopen(base_path, "a+b")
        LogMSG logmsg
        LogMSG buffer[1024]

    if cfile == NULL:
        raise IOError("Cannot open specified file")
    try:
        for i in range(U_shape):
                strncpy(logmsg.ts, timestamp, 31)
                logmsg.ts[31] = b'\0'
                logmsg.salg = i
                logmsg.pivot_ratio = 1
                logmsg.miarowind = 0
                logmsg.miacolind = 0

                logmsg.amresccl = 0.0
                logmsg.amresup = 0.0
                logmsg.amresn = 0.0

                logmsg.miaresccl = 0.0
                logmsg.miaresup = 0.0
                logmsg.miaresn = 0.0

                if i != U_shape - 1:
                    maxval = _fmax(U, i)

                    if maxval.val == 0:
                        raise ValueError("The matrix is degenerate")
                    if maxval.val < 1e-300:
                        raise ValueError("The matrix is close to degenerate")

                    if maxval.val != fabs(U[i, i]):
                        logmsg.pivot_ratio = fabs(maxval.val / U[i, i])
                        if maxval.rowind != i:
                            logmsg.miarowind = maxval.rowind - i
                            for k in range(U_shape):
                                f = U[i, k]
                                U[i, k] = U[maxval.rowind, k]
                                U[maxval.rowind, k] = f

                            for k in range(U_shape):
                                b = P[i, k]
                                P[i, k] = P[maxval.rowind, k]
                                P[maxval.rowind, k] = b

                            for k in range(i):
                                f = L[i, k]
                                L[i, k] = L[maxval.rowind, k]
                                L[maxval.rowind, k] = f

                        if maxval.colind != i:
                            logmsg.miacolind = maxval.colind - i
                            for k in range(U_shape):
                                f = U[k, i]
                                U[k, i] = U[k, maxval.colind]
                                U[k, maxval.colind] = f

                            for k in range(U_shape):
                                b = Q[k, i]
                                Q[k, i] = Q[k, maxval.colind]
                                Q[k, maxval.colind] = b

                D[i] = U[i, i]
                U[i, i] = 1.0

                for k in range(i + 1, U_shape):
                    L[k, i] = U[k, i] / D[i]
                    c1 = U[k, i] - L[k, i] * D[i]
                    logmsg.amresccl += fabs(c1)
                    if c1 > c2:
                        c2 = c1

                logmsg.miaresccl = c2

                U[i + 1:, i] = 0.0
                c2 = 0.0

                for k in range(i + 1, U_shape):
                    for j in range(i + 1, U_shape):
                        v = L[k, i] * U[i, j]
                        f = U[k, j]
                        U[k, j] = f - v
                        c1 = (f - U[k, j]) - v
                        logmsg.amresup += fabs(c1)
                        if c1 > c2:
                            c2 = c1

                logmsg.miaresup = c2
                c2 = 0.0

                for k in range(i + 1, U_shape):
                    f = U[i, k]
                    U[i, k] = f / D[i]
                    c1 = f - U[i, k] * D[i]
                    logmsg.amresn += fabs(c1)
                    if c1 > c2:
                        c2 = c1

                logmsg.miaresn = c2
                c2 = 0.0
                if U_shape > i + 1:
                    logmsg.amresccl = logmsg.amresccl / (U_shape - i - 1)
                    logmsg.amresup = logmsg.amresup / (U_shape - i - 1)**2
                    logmsg.amresn = logmsg.amresn / (U_shape - i - 1)
                else:
                    logmsg.amresccl = 0.0
                    logmsg.amresup = 0.0
                    logmsg.amresn = 0.0

                buffer[ptr] = logmsg
                ptr += 1
                if ptr >= 1024:
                    fwrite(buffer, sizeof(LogMSG), ptr, cfile)
                    ptr = 0

        if ptr > 0:
            fwrite(buffer, sizeof(LogMSG), ptr, cfile)
    finally:
        fclose(cfile)

def ldu_decomposition(cnp.ndarray[cnp.double_t, ndim=2] A, bint copy = True, bint for_solve = True):
    cdef double[:, ::1] U
    if copy:
        U = np.copy(A)
    else:
        U = A

    cdef:
        char[:, ::1] P = np.eye(A.shape[0], A.shape[1], dtype=np.int8)
        char[:, ::1] Q = np.eye(A.shape[0], A.shape[1], dtype=np.int8)
        double[:, ::1] L = np.eye(A.shape[0], A.shape[1], dtype=np.float64)
        double[::1] D = np.zeros(A.shape[0],  dtype=np.float64)

    _ldu_decomposition(P, Q, L, D, U)

    if for_solve:
        return np.asarray(P), np.asarray(Q), np.asarray(L), np.asarray(D), np.asarray(U)
    else:
        return np.asarray(P), np.asarray(Q), np.asarray(L), np.asarray(np.diag(D)), np.asarray(U)

def ldu_decomposition_plus(cnp.ndarray[cnp.double_t, ndim=2] A, str path, bint copy = True, bint for_solve = True):
    cdef double[:, ::1] U
    if copy:
        U = np.copy(A)
    else:
        U = A

    cdef:
        char[:, ::1] P = np.eye(A.shape[0], A.shape[1], dtype=np.int8)
        char[:, ::1] Q = np.eye(A.shape[0], A.shape[1], dtype=np.int8)
        double[:, ::1] L = np.eye(A.shape[0], A.shape[1], dtype=np.float64)
        double[::1] D = np.zeros(A.shape[0],  dtype=np.float64)
        bytes log_path_bytes = path.encode("utf-8")
        bytes timestamp_bytes = (datetime.strftime(datetime.now(),"%d.%m.%Y %H:%M:%S:%f") + ":" + f"{random.randint(0, 9999):04d}").encode("utf-8")
        char* timestamp = timestamp_bytes
        char* log_path = log_path_bytes

    _ldu_decomposition_plus(P, Q, L, D, U, log_path, timestamp)

    if for_solve:
        return np.asarray(P), np.asarray(Q), np.asarray(L), np.asarray(D), np.asarray(U)
    else:
        return np.asarray(P), np.asarray(Q), np.asarray(L), np.asarray(np.diag(D)), np.asarray(U)

def solve(cnp.ndarray[cnp.int8_t, ndim=2] P,
            cnp.ndarray[cnp.int8_t, ndim=2] Q,
            cnp.ndarray[cnp.double_t, ndim=2] L,
            cnp.ndarray[cnp.double_t, ndim=1] D,
            cnp.ndarray[cnp.double_t, ndim=2] U,
            cnp.ndarray[cnp.double_t, ndim=1] b):
    cdef:
        char[:, ::1] P_view = np.ascontiguousarray(P, dtype=np.int8)
        char[:, ::1] Q_view = np.ascontiguousarray(Q, dtype=np.int8)
        double[:, ::1] L_view = np.ascontiguousarray(L, dtype=np.float64)
        double[::1] D_view = np.ascontiguousarray(D, dtype=np.float64)
        double[:, ::1] U_view = np.ascontiguousarray(U, dtype=np.float64)
        double[::1] b_view = np.ascontiguousarray(b, dtype=np.float64)

    result = _source_solve(P_view, Q_view, L_view, D_view, U_view, b_view)

    return np.asarray(result)