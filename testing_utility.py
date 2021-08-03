import numpy as np
import scipy.optimize
import scipy.linalg
from data_structures import REAL_TYPES, COMPLEX_TYPES


# Error calculation
def calculate_error_up_to_shifts(cov_estimator, cov_real):
    """
    Function calculates the error of algorithms 1 + 2.
    :note: This is the suggested error calculating method of the paper.
    """
    error = np.inf
    L, L1 = cov_estimator.shape
    assert L == L1
    assert (L, L) == cov_real.shape

    for i in range(L):
        error = np.min((np.linalg.norm(cov_estimator - cov_real, ord='fro'), error))
        cov_real = np.roll(cov_real, (1, 1), axis=(0, 1))
    return np.power(error / np.linalg.norm(cov_real, ord='fro'), 2)


# Utility
def signal_trispectrum_from_cov_hat(cov_hat, sigma=0, num_type=np.complex128):
    """
    :note: if there is noise (sigma), then we account for it in the
    main diagonal of cov_hat before calculating the tri-spectrum.
    """
    L, L2 = cov_hat.shape
    assert L == L2
    trispectrum = np.zeros((L, L, L), dtype=np.complex128)
    cov_hat = np.copy(cov_hat) + L * np.diag(np.full(L, sigma ** 2))
    for k1 in range(L):
        for k2 in range(L):
            for k3 in range(L):
                if num_type in COMPLEX_TYPES:
                    trispectrum[k1, k2, k3] = cov_hat[k1, k2] * cov_hat[(k3 - k2 + k1) % L, k3].conj() \
                                              + cov_hat[k1, (k3 - k2 + k1) % L] * cov_hat[k2, k3].conj()
                elif num_type in REAL_TYPES:
                    trispectrum[k1, k2, k3] = cov_hat[k1, k2] * cov_hat[(k3 - k2 + k1) % L, k3].conj() \
                                              + cov_hat[k1, (k3 - k2 + k1) % L] * cov_hat[k2, k3].conj() \
                                              + cov_hat[k1, ((-k3) + L) % L] * cov_hat[
                                                  k2, (-(k3 - k2 + k1) + L) % L].conj()
                else:
                    raise ValueError(f"Value error: unknown num_type={num_type}")
    return trispectrum


def diag_wrap(matrix, k):
    """
    Function returns the kth "main wrapped diagonal" of a square matrix.
    """
    L, L2 = matrix.shape
    assert L == L2
    return np.concatenate((np.diag(matrix, k), np.diag(matrix, k - L)))


def calculate_error_up_to_circulant(c_x_est, cov_fft):
    """
    Calculates the error between C_x estimator and FFT of the original covariance matrix
    disregarding the circulant angles multiplication.
    May be viewed as error calculation for algorithm 1.
    :note: This is not an algorithm necessary for the paper, and as such it is part of the test file.
    """
    L, L2 = cov_fft.shape
    assert L == L2

    solutions = []

    # Note: in i==0 the angle is 0 and so no reason to account it here.
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
    # Accounting for the error of the case where the angle is 0.
    error += np.linalg.norm(np.diag(c_x_est) - np.diag(cov_fft)) ** 2
    return np.sqrt(error), phi
