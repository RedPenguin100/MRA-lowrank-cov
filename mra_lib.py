import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize


def diag_wrap(matrix, k):
    return np.concatenate((np.diag(matrix, k), np.diag(matrix, -k)))


def signal_power_spectrum_from_data(data_fft, sigma):
    N, L = data_fft.shape
    power_spectra = np.power(np.abs(data_fft), 2.0)
    return np.mean(power_spectra, axis=0) - np.power(sigma, 2.0) * L


def signal_trispectrum_from_data(data_fft, sigma):
    # TODO make efficient.
    N, L = data_fft.shape
    trispectrum = np.zeros((L, L, L))
    for k1 in range(L):
        for k2 in range(L):
            for k3 in range(L):
                data_trispectra = data_fft[:, k1] * data_fft[:, k2].conj() \
                                  * data_fft[:, k3] * data_fft[:, (k1 - k2 + k3) % L].conj()
                trispectrum[k1, k2, k3] = np.mean(data_trispectra, axis=0)

    return trispectrum


def calculate_error(c_x_est, cov_fft):
    L, L2 = cov_fft.shape
    assert L == L2

    solutions = []

    for i in range(1, L):
        def _objective(phi):
            return np.linalg.norm(diag_wrap(c_x_est, k=i) - diag_wrap(cov_fft, k=i) * np.exp(1j * phi)) ** 2

        bnds = (0, 2 * np.pi)
        solutions.append(scipy.optimize.minimize_scalar(_objective, bounds=(0, 2 * np.pi), method='bounded',
                                                        tol=1e-14))

    error = 0
    phi = []
    for sol in solutions:
        print(sol)
        error += sol.fun
        phi.append(sol.x)
    # Accounting for the 1 in the circulant matrix
    error += np.linalg.norm(np.diag(c_x_est) - np.diag(cov_fft)) ** 2
    return np.sqrt(error), phi


def recover_cov_estimator(data, sigma=0):
    data_fft = np.fft.fft(data)
    p_y_estimator = signal_power_spectrum_from_data(data_fft, sigma)
    t_y_estimator = signal_trispectrum_from_data(data_fft, sigma)
