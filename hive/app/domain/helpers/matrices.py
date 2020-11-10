"""Module used by :py:class:`domain.cluster_groups.Cluster` to create transition
matrices for the simulation.

You should implement your own metropolis-hastings or alternative algorithms
as well as any steady-state or transition matrix optimization algorithms in
this module.
"""

import random
from typing import Tuple, Optional

import cvxpy as cvx
import numpy as np
from matlab.engine import EngineError
from mosek import MosekException
from scipy.sparse.csgraph import connected_components

from domain.helpers.exceptions import *
from domain.helpers.matlab_utils import MatlabEngineContainer
from utils.randoms import random_index

OPTIMAL_STATUS = {cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE}


# region Markov Matrix Constructors
# noinspection PyIncorrectDocstring
def new_mh_transition_matrix(
        a: np.ndarray, v_: np.ndarray) -> Tuple[np.ndarray, float]:
    """ Constructs a transition matrix using metropolis-hastings.

    Constructs a transition matrix using metropolis-hastings algorithm  for
    the specified steady state ``v``.

    Note:
        The input Matrix hould have no transient states or absorbent nodes,
        but this is not enforced or verified.

    Args:
        a:
            A symmetric adjency matrix.
        `v_`:
            A stochastic steady state distribution vector.

    Returns:
        Markov Matrix with ``v_`` as steady state distribution and the
        respective mixing rate or ``None, float('inf')`` if the problem is
        infeasible.
    """
    t = _metropolis_hastings(a, v_)
    return t, get_mixing_rate(t)


# noinspection PyIncorrectDocstring
def new_sdp_mh_transition_matrix(
        a: np.ndarray, v_: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Constructs a transition matrix using semi-definite programming techniques.

    Constructs a transition matrix using metropolis-hastings algorithm  for
    the specified steady state ``v``. The provided adjacency matrix A is first
    optimized with semi-definite programming techniques for the uniform
    distribution vector.

    Args:
        a:
            A non-optimized symmetric adjency matrix.
        `v_`:
            A stochastic steady state distribution vector.

    Returns:
        Markov Matrix with ``v_`` as steady state distribution and the
        respective mixing rate or ``None, float('inf')`` if the problem is
        infeasible.
    """
    try:
        problem, a = _adjency_matrix_sdp_optimization(a)
        if problem.status in OPTIMAL_STATUS:
            t = _metropolis_hastings(a.value, v_)
            return t, get_mixing_rate(t)
        else:
            return None, float('inf')
    except (cvx.SolverError, cvx.DCPError):
        return None, float('inf')


# noinspection PyIncorrectDocstring
def new_go_transition_matrix(
        a: np.ndarray, v_: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Constructs a transition matrix using global optimization techniques.

    Constructs an optimized markov matrix using linear programming relaxations
    and convex envelope approximations for the specified steady state ``v``.
    Result is only trully optimal if :math:`normal(Mopt - (1 / len(v)), 2)`
    is equal to the highest Markov Matrix eigenvalue that is smaller than one.

    Args:
        a:
            A non-optimized symmetric adjency matrix.
        `v_`:
            A stochastic steady state distribution vector.

    Returns:
        Markov Matrix with ``v_`` as steady state distribution and the
        respective mixing rate.
    """
    # Allocate python variables
    n: int = a.shape[0]
    ones_vector: np.ndarray = np.ones(
        n)  # np.ones((3,1)) shape is (3, 1)... whereas np.ones(n) shape is (3,), the latter is closer to cvxpy representation of vector
    ones_matrix: np.ndarray = np.ones((n, n))
    zeros_matrix: np.ndarray = np.zeros((n, n))
    u: np.ndarray = np.ones((n, n)) / n

    # Specificy problem variables
    t: cvx.Variable = cvx.Variable((n, n))

    # Create constraints - Python @ is Matrix Multiplication (MatLab equivalent is *), # Python * is Element-Wise Multiplication (MatLab equivalent is .*)
    constraints = [
        t >= 0,  # Entries must be non-negative
        (t @ ones_vector) == ones_vector,  # Row vector sum equals one
        cvx.multiply(t, ones_matrix - a) == zeros_matrix,   # any zero entry in topology is also a zero in the new matrix
        (v_ @ t) == v_,  # The resulting markov matrix must converge to equilibrium.
    ]

    # Formulate and Solve Problem
    try:
        objective = cvx.Minimize(cvx.norm(t - u, 2))
        problem = cvx.Problem(objective, constraints)
        problem.solve()

        if problem.status in OPTIMAL_STATUS:
            return t.value.transpose(), get_mixing_rate(t.value)
        else:
            return None, float('inf')
    except (cvx.SolverError, cvx.DCPError):
        return None, float('inf')


# noinspection PyIncorrectDocstring
def new_mgo_transition_matrix(
        a: np.ndarray, v_: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Constructs an optimized transition matrix using the matlab engine.

    Constructs an optimized transition matrix using linear programming
    relaxations and convex envelope approximations for the specified steady
    state ``v``.
    Result is only trully optimal if :math:`normal(Mopt - (1 / len(v)), 2)`
    is equal to the highest Markov Matrix eigenvalue that is smaller than one.

    Note:
        This function's code runs inside a matlab engine because it provides
        a non-convex SDP solver BMIBNB. If you do not have valid matlab
        license the output of this function is always ``(None, float('inf')``.

    Args:
        a:
            A non-optimized symmetric adjency matrix.
        `v_`:
            A stochastic steady state distribution vector.

    Returns:
        Markov Matrix with ``v_`` as steady state distribution and the
        respective mixing rate.
    """
    matlab_container = MatlabEngineContainer.get_instance()
    try:
        result = matlab_container.matrix_global_opt(a, v_)
        if result:
            t = np.array(result._data).reshape(result.size, order='F').T
            return t, get_mixing_rate(t)
        else:
            return None, float('inf')
    except (EngineError, AttributeError):
        # EngineError deals with invalid license or unfeasible problems,
        # AttributeError deals MatlabEngineContainer.eng with None value.
        return None, float('inf')
# endregion


# region SDP Optimization
def _adjency_matrix_sdp_optimization(
        a: np.ndarray) -> Optional[Tuple[cvx.Problem, cvx.Variable]]:
    """Optimizes a symmetric adjacency matrix using Semidefinite Programming.

    The optimization is done with respect to the uniform stochastic vector
    with the the same length as the inputed symmetric matrix.

    Note:
        This function tries to use
        `Mosek Solver <https://docs.mosek.com/9.2/pythonapi/index.html>`_,
        if a valid license is not found, it uses
        `SCS Solver <https://github.com/cvxgrp/scs>`_ instead.

    Args:
        a:
            Any symmetric adjacency matrix.

    Returns:
        The optimal matrix or None if the problem is unfeasible.
    """

    # Allocate python variables
    n: int = a.shape[0]
    ones_vector: np.ndarray = np.ones(
        n)  # np.ones((3,1)) shape is (3, 1)... whereas np.ones(n) shape is (3,), the latter is closer to cvxpy representation of vector
    ones_matrix: np.ndarray = np.ones((n, n))
    zeros_matrix: np.ndarray = np.zeros((n, n))
    u: np.ndarray = np.ones((n, n)) / n

    # Specificy problem variables
    a_opt: cvx.Variable = cvx.Variable((n, n), symmetric=True)
    t: cvx.Variable = cvx.Variable()
    i: np.ndarray = np.identity(n)

    # Create constraints - Python @ is Matrix Multiplication (MatLab equivalent is *), # Python * is Element-Wise Multiplication (MatLab equivalent is .*)
    constraints = [
        a_opt >= 0,  # Entries must be non-negative
        (a_opt @ ones_vector) == ones_vector,  # Row vector sum equals one
        cvx.multiply(a_opt, ones_matrix - a) == zeros_matrix,  # any zero entry in topology is also a zero in the new matrix
        (a_opt - u) >> (-t * i),  # eigenvalue lower bound,
        (a_opt - u) << (t * i)  # eigenvalue upper bound
    ]  # cvxpy does not accept chained constraints, e.g.: 0 <= x <= 1

    # Formulate and Solve Problem
    objective = cvx.Minimize(t)
    problem = cvx.Problem(objective, constraints)

    try:
        # try using Mosek before any other solver for SDP problem solving.
        if cvx.MOSEK in cvx.installed_solvers():
            problem.solve(solver=cvx.MOSEK)
        else:
            problem.solve(solver=cvx.SCS)
    except MosekException:
        # catches invalid MosekException invalid license.
        problem.solve(solver=cvx.SCS)

    return problem, a_opt


# endregion


# region Metropolis Hastings
def _metropolis_hastings(a: np.ndarray,
                         v_: np.ndarray,
                         column_major_out: bool = True,
                         version: int = 2) -> np.ndarray:
    """ Constructs a transition matrix using metropolis-hastings algorithm.

    Note:
        The input Matrix hould have no transient states/absorbent nodes,
        but this is not enforced or verified.

    Args:
        a:
            A symmetric adjency matrix.
        `v_`:
            A stochastic vector that is the steady state of the resulting
            transition matrix.
        column_major_out:
             Indicates whether to return transition_matrix output
            is in row or column major form.
        version:
             Indicates which version of the algorith should be used
            (default is version 2).

    Returns:
        An unlabeled transition matrix with steady state ``v_``.

    Raises:
        DistributionShapeError:
            When the length of ``v_`` is not the same as the matrix `a`.
        MatrixNotSquareError:
            When matrix `a` is not a square matrix.
    """

    if v_.shape[0] != a.shape[1]:
        raise DistributionShapeError(
            "distribution shape: {}, proposal matrix shape: {}".format(
                v_.shape, a.shape))
    if a.shape[0] != a.shape[1]:
        raise MatrixNotSquareError(
            "rows: {}, columns: {}, expected square matrix".format(
                a.shape[0], a.shape[1]))

    shape: Tuple[int, int] = a.shape
    size: int = a.shape[0]

    rw: np.ndarray = _construct_random_walk_matrix(a)
    if version == 1:
        rw = rw.transpose()

    r: np.ndarray = _construct_rejection_matrix(rw, v_)

    m: np.ndarray = np.zeros(shape=shape)
    for i in range(size):
        for j in range(size):
            if i != j:
                m[i, j] = rw[i, j] * min(1, r[i, j])
        if version == 1:
            m[i, i] = _get_diagonal_entry_probability_v1(rw, r, i)
        elif version == 2:
            m[i, i] = _get_diagonal_entry_probability_v2(m, i)

    if column_major_out:
        return m.transpose()

    return m


def _construct_random_walk_matrix(a: np.ndarray) -> np.ndarray:
    """Builds a random walk matrix over the given adjacency matrix

    Args:
        a:
            Any adjacency matrix.

    Returns:
        A matrix representing the performed random walk.
    """
    # Version 1.
    # shape = a.shape
    # size = shape[0]
    # rw: np.ndarray = np.zeros(shape=shape)
    # for i in range(size):
    #     # all possible states reachable from state i, including self
    #     degree: Any = np.sum(a[i, :])
    #     for j in range(size):
    #         rw[i, j] = a[i, j] / degree
    # return rw
    # Version 2 - Returns Column Major Random Walk, similar to MatLab.
    #   To return a equivalent of version 1 output, transpose the result.
    return a / np.sum(a, axis=1)


def _construct_rejection_matrix(rw: np.ndarray, v_: np.ndarray) -> np.ndarray:
    """Builds a matrix of rejection probabilities for a given random walk.

    Args:
        rw:
            a random_walk over an adjacency matrix
        `v_`:
            a stochastic desired distribution vector

    Returns:
        A matrix whose entries are acceptance probabilities for ``rw``.
    """
    shape = rw.shape
    size = shape[0]
    r = np.zeros(shape=shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(size):
            for j in range(size):
                r[i, j] = (v_[j] * rw[j, i]) / (v_[i] * rw[i, j])
    return r


def _get_diagonal_entry_probability_v1(
        rw: np.ndarray, r: np.ndarray, i: int) -> np.float64:
    """Helper function used during the metropolis-hastings algorithm.

    Calculates the value that should be assigned to the entry ``(i, i)`` of the
    transition matrix being calculated by the metropolis hastings algorithm
    by considering the rejection probability over the random walk that was
    performed on an adjacency matrix.

    Note:
        This method does considers element-wise rejection probabilities
        for random walk matrices. If you wish to implement a modification of
        the metropolis-hastings algorithm and you do not utilize rejection
        matrices use :py:func:`_get_diagonal_entry_probability_v2` instead.

    Args:
        rw:
            A random walk over an adjacency matrix.
        r:
            A matrix whose entries contain acceptance probabilities for ``rw``.
        i:
            The diagonal-index of ``rw`` where summation needs to
            be performed on. E.g.: ``rw[i, i]``.

    Returns:
        A probability to be inserted at entry ``(i, i)`` of the transition
        matrix outputed by the :py:func:`_metropolis_hastings`.
    """
    size: int = rw.shape[0]
    pii: np.float64 = rw[i, i]
    for k in range(size):
        pii += rw[i, k] * (1 - min(1, r[i, k]))
    return pii


def _get_diagonal_entry_probability_v2(m: np.ndarray, i: int) -> np.float64:
    """Helper function used during the metropolis-hastings algorithm.

    Calculates the value that should be assigned to the entry ``(i, i)`` of the
    transition matrix being calculated by the metropolis hastings algorithm
    by considering the rejection probability over the random walk that was
    performed on an adjacency matrix.

    Note:
        This method does not consider element-wise rejection probabilities
        for random walk matrices. If you wish to implement a modification of
        the metropolis-hastings algorithm and you utilize rejection matrices
        use :py:func:`_get_diagonal_entry_probability_v1` instead.

    Args:
        m:
            The matrix to receive the diagonal entry value.
        i:
            The diagonal entry index. E.g.: ``m[i, i]``.

    Returns:
        A probability to be inserted at entry ``(i, i)`` of the transition matrix
        outputed by the :py:func:`_metropolis_hastings`.
    """
    return 1 - np.sum(m[i, :])


# endregion


# region Helpers
def get_mixing_rate(m: np.ndarray) -> float:
    """Calculats the fast mixing rate the input matrix.

    The fast mixing rate of matrix ``m`` is the highest eigenvalue that is
    smaller than one. If returned value is ``1.0`` than the matrix has transient
    states or absorbent nodes and as a result is not a markov matrix.

    Args:
        m:
            A matrix.

    Returns:
        The highest eigenvalue of ``m`` that is smaller than one or one.
    """
    size = m.shape[0]

    if size != m.shape[1]:
        raise MatrixNotSquareError(
            "Can not compute eigenvalues/vectors with non-square matrix")
    m = m - (np.ones((size, size)) / size)
    eigenvalues, eigenvectors = np.linalg.eig(m)
    mixing_rate = np.max(np.abs(eigenvalues))
    return mixing_rate.item()


def new_vector(size: int) -> np.ndarray:
    u_ = np.random.random_sample(size)
    u_ /= np.sum(u_)
    return u_


def new_symmetric_matrix(
        size: int, allow_sloops: bool = True, force_sloops: bool = True
) -> np.ndarray:
    """Generates a random symmetric matrix.

    The generated adjacency matrix does not have transient state sets or
    absorbent nodes and can effectively represent a network topology
    with bidirectional connections between :py:class:`network nodes
    <app.domain.network_nodes.Node>`.

    Args:
        size:
             The length of the square matrix.
        allow_sloops:
            Indicates if the generated adjacency matrix allows diagonal
            entries representing self-loops. If ``False``, then, all diagonal
            entries must be zeros. Otherwise, they can be zeros or ones.
        force_sloops:
            Indicates if the diagonal of the generated matrix should be
            filled with ones. If ``False`` valid diagonal entries are
            decided by ``allow_self_loops`` param. Otherwise, diagonal entries
            are filled with ones. If ``allow_self_loops`` is ``False``
            and ``enforce_loops`` is ``True``, an error is raised.

    Returns:
        The adjency matrix representing the connections between a
        groups of :py:class:`network nodes <app.domain.network_nodes.Node>`.

    Raises:
        IllegalArgumentError:
            When ``allow_self_loops`` (``False``) conflicts with
            ``enforce_loops`` (``True``).
    """
    if not allow_sloops and force_sloops:
        raise IllegalArgumentError("Can not invoke new_symmetric_matrix with:\n"
                                   "    [x] allow_sloops=False\n"
                                   "    [x] force_sloops=True")
    secure_random = random.SystemRandom()
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            if i == j:
                if not allow_sloops:
                    m[i, i] = 0
                elif force_sloops:
                    m[i, i] = 1
                else:
                    m[i, i] = __new_edge_val__(secure_random)
            else:
                m[i, j] = m[j, i] = __new_edge_val__(secure_random)
    return m


def new_symmetric_connected_matrix(
        size: int, allow_sloops: bool = True, force_sloops: bool = True
) -> np.ndarray:
    """Generates a random symmetric matrix which is also connected.

    See :py:func:`new_symmetric_matrix` and :py:func:`make_connected`.

    Args:
        size:
            The length of the square matrix.
        allow_sloops:
            See :py:func:`~app.domain.helpers.matrices.new_symmetric_matrix`
            for clarifications.
        force_sloops:
            See :py:func:`~app.domain.helpers.matrices.new_symmetric_matrix`
            for clarifications.

    Returns:
        A matrix that represents an adjacency matrix that is also connected.
    """
    m = np.asarray(new_symmetric_matrix(size))
    if not is_connected(m):
        m = make_connected(m)
    return m


def make_connected(m: np.ndarray) -> np.ndarray:
    """Turns a matrix into a connected matrix that could represent a
    connected graph.

    Args:
        m: The matrix to be made connected.

    Returns:
        A connected matrix. If ``m`` was symmetric the modified matrix will
        also be symmetric.
    """
    size = m.shape[0]
    # Use guilty until proven innocent approach for both checks
    for i in range(size):
        is_absorbent_or_transient: bool = True
        for j in range(size):
            # Ensure state i can reach and be reached by some other state j
            if m[i, j] == 1 and i != j:
                is_absorbent_or_transient = False
                break
        if is_absorbent_or_transient:
            j = random_index(i, size)
            m[i, j] = m[j, i] = 1
    return m


def is_symmetric(m: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if a matrix is symmetric by performing element-wise equality
    comparison on entries of ``m`` and  ``m.T``.

    Args:
        m:
            The matrix to be verified.
        tol:
            The tolerance used to verify the entries of the ``m`` (default
            is 1e-8).

    Returns:
        ``True`` if the ``m`` is symmetric, else ``False``.
    """
    return np.all(np.abs(m - m.transpose()) < tol)


def is_connected(m: np.ndarray, directed: bool = False) -> bool:
    """Checks if a matrix is connected by counting the number of connected
    components.

    Args:
        m:
            The matrix to be verified.
        directed:
            If ``m`` edges are directed, i.e., if ``m`` is an adjency
            matrix in which the edges bidirectional. ``False`` means they
            are. ``True`` means they are not.

    Returns:
        ``True`` if the matrix is a connected graph, else ``False``.
    """
    n, cc_labels = connected_components(m, directed=directed)
    return n == 1


def __new_edge_val__(random_generator: random.SystemRandom) -> np.float64:
    p = random_generator.uniform(0.0, 1.0)
    return np.ceil(p) if p >= 0.5 else np.floor(p)
# endregion
