import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize
from data_structures import COMPLEX_TYPES, REAL_TYPES

np.set_printoptions(linewidth=np.inf)


def create_matrix_from_diagonals(diagonals):
    """
    Function takes L diagonals of dimension L and creates
    a matrix where the ith diagonal is the ith main diagonal
    of the new matrix.
    """
    L, L = diagonals.shape
    target_matrix = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        diagonal = diagonals[i]
        for j in range(L):
            target_matrix[j][(j + i) % L] = diagonal[j]
    return target_matrix


def reverse_cov_fft(cov_hat):
    """
    This is good up to a phase.
    """
    return np.fft.ifft(np.fft.ifft(cov_hat.T).conj().T)


def signal_power_spectrum_from_data(data_fft):
    """
    :note: sigma is accounted for in different part of the algorithm.
    """
    power_spectra = np.power(np.abs(data_fft), 2.0)
    return np.mean(power_spectra, axis=0)


def signal_trispectrum_from_data(data_fft):
    """
    :note: the algorithm is extremely inefficient due to the lack of utilizing
    NumPy's power.
    """
    N, L = data_fft.shape
    trispectrum = np.zeros((L, L, L), dtype=np.complex128)
    for k1 in range(L):
        for k2 in range(L):
            for k3 in range(L):
                data_trispectra = data_fft[:, k1] * data_fft[:, k2].conj() \
                                  * data_fft[:, k3] * data_fft[:, (k1 - k2 + k3) % L].conj()
                trispectrum[k1, k2, k3] = np.mean(data_trispectra, axis=0, dtype=np.complex128)

    return trispectrum


def recover_c_x_estimator(data, sigma=0, num_type=np.complex128):
    """
    Implementation of algorithm 1 in the paper.
    """
    N, L = data.shape

    data_fft = np.fft.fft(data)
    p_y_estimator = signal_power_spectrum_from_data(data_fft)
    t_y_estimator = signal_trispectrum_from_data(data_fft)
    G_arr = [np.outer(p_y_estimator, p_y_estimator)]
    constraints = []
    for i in range(1, L):
        G_arr.append(cp.Variable((L, L), complex=True))
        constraints.append(G_arr[i] >> 0)
    expression = 0
    for k1 in range(L):
        for k2 in range(L):
            for m in range(L):
                if num_type in COMPLEX_TYPES:
                    # noinspection PyTypeChecker
                    expression += cp.power(cp.abs(
                        t_y_estimator[k1 % L, (k1 + m) % L, (k2 + m) % L]
                        - G_arr[(k2 - k1) % L][k1, (k1 + m) % L]
                        - G_arr[m][k1, k2]
                    ), 2)
                elif num_type in REAL_TYPES:
                    # noinspection PyTypeChecker
                    expression += cp.power(cp.abs(
                        t_y_estimator[k1 % L, (k1 + m) % L, (k2 + m) % L]
                        - G_arr[(k2 - k1) % L][k1, (k1 + m) % L]
                        - G_arr[m][k1, k2]
                        - G_arr[(k1 + k2 + m) % L][(-k2) % L, (-k2 - m) % L]
                    ), 2)
                else:
                    raise ValueError(f"Invalid num_type: {num_type}")
    obj = cp.Minimize(expression)
    problem = cp.Problem(obj, constraints=constraints)
    problem.solve()
    print(problem.status)
    G_arr = [G_arr[i].value for i in range(1, len(G_arr))]

    d_estimates = [p_y_estimator]
    for G in G_arr:
        w, v = np.linalg.eig(G)
        largest_eigval = np.max(w)
        largest_eigvec = v[:, np.argmax(w)]
        d_estimates.append(np.sqrt(largest_eigval) * largest_eigvec)
    cov_estimator = create_matrix_from_diagonals(np.array(d_estimates)) - L * np.diag(np.full(L, sigma ** 2))
    return cov_estimator


def get_H_matrix(C_x, i, j):
    # TODO: roll is extremely inefficient, improve efficiency.
    rotated_c_x = np.roll(C_x, (-i, -j), axis=(0, 1))
    return C_x * rotated_c_x.conj()  # Hadamard product


def get_V_matrix(H, vectors_amount):
    """
    :param H: Hermitian or real symmetric matrix.
    """
    L1, L = H.shape
    assert L1 == L
    w, v = np.linalg.eig(H)
    arg_max_eigvals = w.argsort()[-vectors_amount:][::-1]
    return v[:, arg_max_eigvals]


def get_e_m(m, L):
    """
    Given m, L, return a vector of length L with
    1 in the mth place :
    Content: (0,...,0,1,0,...,0)
    Indices:  0.......m,.....L-1
    """
    assert m < L
    e_m = np.zeros(L, dtype=np.complex128)
    e_m[m] = 1
    return e_m


def solve_ambiguities(C_x, r=None):
    """
    Implementation of algorithm 2 in the paper.
    """
    L1, L = C_x.shape
    assert L1 == L
    if r is None:
        r_squared = np.ceil(np.sqrt(L) - 1) ** 2
    else:
        r_squared = r ** 2

    vectors_amount = min(r_squared, L)
    V_array = np.zeros((L, L, vectors_amount), dtype=np.complex128)
    for i in range(L):
        H_ii = get_H_matrix(C_x, i, i)
        V_array[i] = get_V_matrix(H_ii, vectors_amount=vectors_amount)

    Z_array = np.zeros((L, L ** 2, vectors_amount ** 2), dtype=np.complex128)
    for i in range(L):
        # Z_array[i] = np.kron(V_array[i], V_array[(i + 1) % L].conj())
        Z_array[i] = np.kron(V_array[(i + 1) % L].conj(), V_array[i])
    block_diag = scipy.linalg.block_diag(*list(Z_array))

    M_array = np.zeros((L, L, L ** 2), dtype=np.complex128)
    for m in range(L):
        e_m = get_e_m(m=m, L=L)
        circulant = scipy.linalg.circulant(e_m).flatten(order='F')
        for i in range(L):
            H_i_i1 = get_H_matrix(C_x, i, (i + 1) % L)
            M_array[i][m] = H_i_i1.flatten(order='F') * circulant
    M_mat = np.hstack(M_array).T
    A = np.hstack((M_mat, -block_diag))
    w, v = np.linalg.eig(A.conj().T @ A)

    V = v[:, np.argmin(w)]
    # Recover phases
    phi = np.zeros(L, dtype=np.complex128)
    k = 0
    for m in range(1, L):
        phi[m] = - np.sum(np.angle(V[1:m + 1])) \
                 + (m / L) * np.sum(np.angle(V[0:L])) \
                 + (2 * np.pi * k * m) / L
    phases = np.exp(-1j * phi)
    cov_estimator = C_x * scipy.linalg.circulant(phases)
    return reverse_cov_fft(cov_estimator)


def fully_recover_c_x(data, sigma=0, num_type=np.complex128, r=None):
    c_x_estimator = recover_c_x_estimator(data, sigma, num_type)
    return solve_ambiguities(c_x_estimator, r)
