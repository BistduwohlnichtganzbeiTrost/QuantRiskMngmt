import numpy as np

def is_positive_semidefinite(matrix: np.array) -> bool:
    """Check whether a matrix is positive semi-definite or not

    Attempt to compute the Cholesky decomposition of the matrix, if this fails
    then the matrix is not positive semidefinite.

    Parameters
    ----------
    matrix : numpy.array
        A matrix

    Returns
    -------
    bool
        True if the matrix is positive semidefinite, else False

    References
    ----------
    .. https://stackoverflow.com/questions/16266720
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False