import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize

np.set_printoptions(linewidth=np.inf)


def get_fft(x):
    return np.fft.fft(x)


def diag_wrap(matrix, k):
    L, L2 = matrix.shape
    assert L == L2
    return np.concatenate((np.diag(matrix, k), np.diag(matrix, k - L)))


def signal_power_spectrum_from_data(data_fft, sigma):
    N, L = data_fft.shape
    power_spectra = np.power(np.abs(data_fft), 2.0)
    return np.mean(power_spectra, axis=0) - np.power(sigma, 2.0) * L


def signal_trispectrum_from_data(data_fft):
    # TODO make efficient.
    N, L = data_fft.shape
    trispectrum = np.zeros((L, L, L), dtype='complex128')
    for k1 in range(L):
        for k2 in range(L):
            for k3 in range(L):
                data_trispectra = data_fft[:, k1] * data_fft[:, k2].conj() \
                                  * data_fft[:, k3] * data_fft[:, (k1 - k2 + k3) % L].conj()
                trispectrum[k1, k2, k3] = np.mean(data_trispectra, axis=0, dtype='complex128')

    return trispectrum


def signal_trispectrum_from_cov_hat(cov_hat):
    L, L2 = cov_hat.shape
    assert L == L2
    trispectrum = np.zeros((L, L, L), dtype='complex128')
    for k1 in range(L):
        for k2 in range(L):
            for k3 in range(L):
                trispectrum[k1, k2, k3] = cov_hat[k1, k2] * cov_hat[(k3 - k2 + k1) % L, k3].conj() \
                                          + cov_hat[k1, (k3 - k2 + k1) % L] * cov_hat[k2, k3].conj()
    return trispectrum


def calculate_error_up_to_circulant(c_x_est, cov_fft):
    """
    Calculates the error between C_x estimator and FFT of the original Covariance matrix
    disregarding the circulant angles multiplication.
    """
    # TODO: make efficient
    L, L2 = cov_fft.shape
    assert L == L2

    solutions = []

    for i in range(1, L):
        def _objective(phi):
            return np.linalg.norm(diag_wrap(c_x_est, k=i) - diag_wrap(cov_fft, k=i) * np.exp(1j * phi)) ** 2

        solutions.append(scipy.optimize.minimize_scalar(_objective, bounds=(0, 2 * np.pi), method='bounded',
                                                        options={'xatol': 1e-10}))

    error = 0
    phi = []
    for sol in solutions:
        error += sol.fun
        phi.append(sol.x)
    # Accounting for the 1 in the circulant matrix
    error += np.linalg.norm(np.diag(c_x_est) - np.diag(cov_fft)) ** 2
    return np.sqrt(error), phi


def recover_c_x_estimator(data, sigma=0):
    N, L = data.shape

    data_fft = get_fft(data)
    p_y_estimator = signal_power_spectrum_from_data(data_fft, sigma)
    print(p_y_estimator)
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
                # noinspection PyTypeChecker
                expression += cp.power(cp.abs(
                    t_y_estimator[k1 % L, (k1 + m) % L, (k2 + m) % L]
                    - G_arr[(k2 - k1) % L][k1, (k1 + m) % L]
                    - G_arr[m][k1, k2]
                    # - G_arr[(k1+k2+m) % L] [(-k2) % L, (-k2 -m) % L]
                ), 2)
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
    d_estimates = np.array(d_estimates)
    cov_estimator = create_matrix_from_diagonals(d_estimates)
    return cov_estimator


def roll_xs(x_samples):
    N, L = x_samples.shape
    for i in range(N):
        x_samples[i] = np.roll(x_samples[i], i % L)

    return x_samples


def generate_xs(n):
    v_1 = np.array([1, 2, 3, 4, 5], dtype='complex128')
    lambda_1 = 0.5
    x_samples = np.outer(np.random.normal(0, np.square(lambda_1) / 2, size=n) +
                         np.random.normal(0, np.square(lambda_1) / 2, size=n) * 1j, v_1)
    return x_samples


def get_cov_hat(x_samples):
    """
    :note: The samples here must not be rolled, or at least
    all of them rolled with the same offset.
    """
    x_samples_fft = get_fft(x_samples)
    cov_hat = np.mean(np.einsum('bi,bo->bio', x_samples_fft, x_samples_fft.conj()), axis=0)
    return cov_hat


def create_matrix_from_diagonals(diagonals):
    L, L = diagonals.shape
    target_matrix = np.zeros((L, L), dtype='complex128')
    for i in range(L):
        diagonal = diagonals[i]
        for j in range(L):
            target_matrix[j][(j + i) % L] = diagonal[j]
    return target_matrix


def get_H_matrix(C_x, i, j):
    # TODO: roll is extremely inefficient, improve efficiency.
    rotated_c_x = np.copy(C_x)
    rotated_c_x = np.roll(rotated_c_x, -i, axis=0)
    rotated_c_x = np.roll(rotated_c_x, -j, axis=1)
    return C_x * rotated_c_x.conj()  # Hadamard product


def get_S_matrix(X):
    L1, L = X.shape
    assert L1 == L
    S = np.zeros((L ** 2, L ** 2))
    for i in range(L):
        for j in range(L):
            S[L * i:L * i + L, L * j:L * j + L] = get_H_matrix(X, i, j)
    return S


def get_K_matrix(S):
    L_squared, L_squared2 = S.shape
    assert L_squared == L_squared2
    w, v = np.linalg.eig(S)
    # v @ np.diag(w) @ v.T
    K = v @ np.sqrt(np.diag(w))
    return K


def get_V_matrix(H, r=1):
    """
    :param H: Hermitian or real symmetric matrix.
    """
    L1, L = H.shape
    assert L1 == L
    assert r ** 2 < L
    w, v = np.linalg.eig(H)
    arg_max_eigvals = w.argsort()[-r ** 2:][::-1]
    return v[:, arg_max_eigvals]


def get_e_m(m, L):
    assert m < L
    e_m = np.zeros(L)
    e_m[m] = 1
    return e_m


def solve_ambiguities(C_x, r=1):
    L1, L = C_x.shape
    assert L1 == L

    V_array = np.zeros((L, L, r ** 2))
    for i in range(L):
        V_array[i] = get_V_matrix(get_H_matrix(C_x, i, i), r=r)
    Z_array = np.zeros((L, L ** 2, r ** 4))
    for i in range(L):
        Z_array[i] = np.kron(V_array[i], V_array[(i + 1) % L].conj())
    block_diag = scipy.linalg.block_diag(*list(Z_array))

    M_array = np.zeros((L, L, L ** 2))
    for m in range(L):
        e_m = get_e_m(m=m, L=L)
        circulant = scipy.linalg.circulant(e_m).T.flatten(order='F')
        for i in range(L):
            M_array[i][m] = get_H_matrix(C_x, i, (i + 1) % L).flatten(order='F') * circulant
    M_mat = np.hstack(M_array).T
    A = np.hstack((M_mat, -block_diag))
    print(A.shape)
    u, s, vh = np.linalg.svd(A)
    print("Singular values(last): ", s[-1])
    print("Singular values(before last): ", s[-2])


if __name__ == "__main__":
    x_samples = generate_xs(n=10000)
    _, L = x_samples.shape
    cov_matrix = np.mean(np.einsum('bi,bo->bio', x_samples, x_samples), axis=0)
    cov_hat = get_cov_hat(x_samples)

    print("cov_hat: ", get_cov_hat(x_samples))

    c_x_estimator = recover_c_x_estimator(roll_xs(x_samples))

    print(calculate_error_up_to_circulant(c_x_estimator, cov_hat))
