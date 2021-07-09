import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize


def diag_wrap(matrix, k):
    L, L2 = matrix.shape
    assert L == L2
    return np.concatenate((np.diag(matrix, k), np.diag(matrix, L - k)))


def signal_power_spectrum_from_data(data_fft, sigma):
    N, L = data_fft.shape
    power_spectra = np.power(np.abs(data_fft), 2.0)
    return np.mean(power_spectra, axis=0) - np.power(sigma, 2.0) * L


def signal_trispectrum_from_data(data_fft, sigma):
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


def get_cov_matrix(signal):
    L = signal.shape[0]
    x_hat = np.fft.fft(signal)
    return np.outer(x_hat, x_hat.conjugate())


def signal_trispectrum_from_signal(signal):
    cov_mat = get_cov_matrix(signal)
    L = signal.shape[0]
    trispectrum = np.zeros((L, L, L), dtype='complex128')
    for k1 in range(L):
        for k2 in range(L):
            for k3 in range(L):
                trispectrum[k1, k2, k3] = cov_mat[k1, k2] * cov_mat[(k3 - k2 + k1) % L, k3].conj() \
                                          + cov_mat[k1, (k3 - k2 + k1) % L] * cov_mat[k2, k3].conj()
    return trispectrum


def calculate_error(c_x_est, cov_fft):
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
        # print(sol)
        error += sol.fun
        phi.append(sol.x)
    # Accounting for the 1 in the circulant matrix
    error += np.linalg.norm(np.diag(c_x_est) - np.diag(cov_fft)) ** 2
    return np.sqrt(error), phi


def recover_cov_estimator(data, sigma=0):
    N, L = data.shape

    data_fft = np.fft.fft(data)
    p_y_estimator = signal_power_spectrum_from_data(data_fft, sigma)
    print(p_y_estimator)
    t_y_estimator = signal_trispectrum_from_data(data_fft, sigma)
    print("Estimated trispectrum: ", t_y_estimator)
    G_arr = []
    G_arr.append(np.outer(p_y_estimator, p_y_estimator))
    for i in range(1, L):
        G_arr.append(cp.Variable((L, L), PSD=True))
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
    problem = cp.Problem(obj)
    problem.solve(qcp='True')
    print(problem.status)
    print("G=", G_arr[1].value)
    return G_arr[1].value


x = np.array([1, 2])
L = x.shape[0]
x_hat = np.fft.fft(x)
cov_hat = np.outer(x_hat, x_hat.conjugate())
print("cov_hat: ", np.real(cov_hat))
print("Real trispectrum: ", signal_trispectrum_from_signal(x))

G = recover_cov_estimator(np.array([x, np.roll(x, 1), x, np.roll(x, 1)]))
w, v = np.linalg.eig(G)
largest_eigval = np.max(w)
largest_eigvec = v[:, np.argmax(w)]
a = np.sqrt(largest_eigval) * largest_eigvec
print("d_1 estimate=", a)
d_1_actual = np.real(diag_wrap(cov_hat, 1))
print("d_1 actual: ", d_1_actual)

print(calculate_error(np.array([[9, a[0]], [a[1], 1]]), cov_hat))
