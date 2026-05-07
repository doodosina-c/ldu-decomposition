ldu-decomposition provides the implementation of LDU-decomposition with full pivoting and allows one to solve a system of linear equations using the obtained LDU-decomposition of given coefficient matrix.

---

### `ldu_decomposition(A, copy = True, for_solve = True)`
Decomposes a coefficient matrix into matrices `P`, `Q`, `L`, `D`, `U`.
#### Parameters:
##### `A: numpy.ndarray`
A coefficient matrix.
##### `copy: bool, optional`
If `True`, then the array `A` is copied. If `copy = False`, make sure that `A` is C-contiguous.
##### `for_solve: bool, optional`
If `True`, then the matrix `D` is represented as 1-D array. To represent the matrix `D` as diagonal matrix `numpy.diag` can be used.
#### Returns
##### `P: ndarray of np.int8`
A permutation matrix that reorders the rows of `A`.
##### `Q: ndarray of np.int8`
A permutation matrix that reorders the columns of `A`.
##### `L: ndarray of float`
A lower unitriangular matrix. 
##### `D: ndarray of float`
A diagonal matrix.
##### `U: ndarray of float`
A upper unitriangular matrix.
#### Raises 
`ValueError`
	If the absolute maximal pivot element equals $0$ in the $i$-th step.
	If the absolute maximal pivot element is less than $10^{-300}$ in the $i$-th step.
#### Notes
The data type of array `A` must be `numpy.float64`. 

--- 

### `ldu_decomposition_plus(A, path, copy = True, for_solve = True)`
Decomposes a coefficient matrix into matrices `P`, `Q`, `L`, `D`, `U` and writes logs into the specified `.bin` file.

#### Parameters:
##### `A: numpy.ndarray`
A coefficient matrix.
##### `path: str`
An absolute path to the `.bin` file.
##### `copy: bool, optional`
If `True`, then the array `A` is copied. If `copy = False`, make sure that `A` is C-contiguous.
##### `for_solve: bool, optional`
If `True`, then the matrix `D` is represented as 1-D array. To represent `D` as a diagonal matrix, `numpy.diag` can be used.
#### Returns
##### `P: ndarray of np.int8`
A permutation matrix that reorders the rows of `A`.
##### `Q: ndarray of np.int8`
A permutation matrix that reorders the columns of `A`.
##### `L: ndarray of float`
A lower unitriangular matrix.
##### `D: ndarray of float`
A diagonal matrix.
##### `U: ndarray of float`
A upper unitriangular matrix.
#### Raises 
##### `ValueError`
If the absolute maximal pivot element equals $0$ in the $i$-th step.
If the absolute maximal pivot element is less than $10^{-300}$ in the $i$-th step.
#### Notes
The data type of array `A` must be `numpy.float64`. 

---

### `solve(P, Q, L, D, U, b)`
Solves a system of linear equations using the obtained decomposition of coefficient matrix.

#### Parameters:
##### `P: ndarray of np.int8`
A permutation matrix that reorders the rows of `A`.
##### `Q: ndarray of np.int8`
A permutation matrix that reorders the columns of `A`.
##### `L: ndarray of float`
A lower unitriangular matrix.
##### `D: ndarray of float`
A diagonal matrix.
##### `U: ndarray of float`
A upper unitriangular matrix.
##### `b: ndarray of float`
A right-hand side vector.
#### Returns
##### `x: ndarray of float`
A vector of unknowns.
#### Notes
The data types of arrays `L`, `D`, `U` must be `numpy.float64`, `P`, `Q` must be `numpy.int8`.