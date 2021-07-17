import pytest
import scipy.linalg

from mra_lib import *


def test_trispectrum_sanity():
    data = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    data_fft = np.fft.fft(data)
    res1 = signal_trispectrum_from_data(data_fft)
    res2 = signal_trispectrum_from_data(np.fft.fft(np.array([[1, 2, 3]])))

    assert pytest.approx(res1) == res2


def test_error_sanity():
    err, phi = calculate_error_up_to_circulant(np.array([[1, 2], [1, 2]]), np.array([[1, 2], [1, 2]]))
    assert pytest.approx(err, abs=1e-5) == 0
    assert pytest.approx(phi[0], abs=1e-10) == 0


def test_error_with_circulant():
    original = np.array([[1, 2], [3, 4]])
    circulant = scipy.linalg.circulant([1, np.exp(1j * np.pi / 4)])
    distorted = original * circulant
    print(original)
    print(distorted)
    err, phi = calculate_error_up_to_circulant(distorted, original)
    assert pytest.approx(err, abs=1e-5) == 0
    assert pytest.approx(phi[0]) == np.pi / 4


def test_error_circulant_distortion_no_phi():
    """
    Calculate the circulant distortion when introducing error
    unrelated to phi
    """
    original = np.array([[1, 2], [3, 4]])
    circulant = scipy.linalg.circulant([1, np.exp(1j * np.pi / 4)])
    distorted = original * circulant + np.array([[0, 0], [0, 100]])
    print(original)
    print(distorted)
    err, phi = calculate_error_up_to_circulant(distorted, original)
    assert err > 1
    assert pytest.approx(phi[0]) == np.pi / 4


def test_trispectrum_identity():
    x_samples, v_arr = generate_xs(100000)
    x_samples_fft = get_fft(x_samples)
    tri_from_data = signal_trispectrum_from_data(x_samples_fft)
    tri_from_cov = signal_trispectrum_from_cov_hat(get_cov_hat(v_arr))
    assert pytest.approx(np.mean(tri_from_data / tri_from_cov), abs=1e-2) == 1


def test_get_H():
    x_samples, v_arr = generate_xs(100000)
    _, L = x_samples.shape
    cov_hat = get_cov_hat(v_arr)
    c_x = recover_c_x_estimator(roll_xs(x_samples))
    for i in range(L):
        h_ii_estimator = get_H_matrix(c_x, i, i)
        h_ii = get_H_matrix(cov_hat, i, i)
        assert pytest.approx(np.mean(h_ii / h_ii_estimator), abs=1e-2) == 1


def test_get_K():
    symmetric_psd = np.array([[1, 2], [2, 5]])
    K = get_K_matrix(symmetric_psd)
    assert pytest.approx(np.linalg.norm(K @ K.conj().T - symmetric_psd)) == 0


def test_solve_ambiguities():
    np.random.seed(42)
    lambdas = [1, 0.75, 0.5]
    r = len(lambdas)
    L = 10
    x_samples, v_arr = generate_xs(100000, L=L, lambdas=lambdas)
    cov_hat = get_cov_hat_from_v_arr(v_arr, lambdas)
    c_x = recover_c_x_estimator(roll_xs(x_samples))
    print(calculate_error_up_to_circulant(cov_hat, c_x))

    cov_estimator = solve_ambiguities(c_x, r=r)
    error = np.inf
    cov_mat = get_cov_mat_from_v_arr(v_arr, lambdas)
    cov_estimator_no_fft = reverse_cov_fft(cov_estimator)
    for i in range(L):
        error = np.min((np.linalg.norm(cov_estimator_no_fft - cov_mat, ord='fro'), error))
        cov_mat = np.roll(cov_mat, (1,1), axis=(0, 1))
    print(f"Final error: {error / np.linalg.norm(cov_mat, ord='fro')}")
    print(calculate_error_up_to_circulant(cov_hat, cov_estimator))
    err, phi = calculate_error_up_to_circulant(cov_hat, cov_estimator)
    phi = np.insert(phi,0,0)
    accurate_estimator = cov_estimator * scipy.linalg.circulant(np.exp(-1j * phi))
    for i in range(L):
        error = np.min((np.linalg.norm(reverse_cov_fft(accurate_estimator) - cov_mat, ord='fro'), error))
        cov_mat = np.roll(cov_mat, (1, 1), axis=(0, 1))
    print(f"Final error: {error / np.linalg.norm(cov_mat, ord='fro')}")
