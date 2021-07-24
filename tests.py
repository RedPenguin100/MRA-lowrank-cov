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


@pytest.mark.parametrize('setting,error', [
    (Setting(n=1000000, L=3, r=1, num_type=np.longdouble, sigma=0), 1e-1),
    (Setting(n=1000000, L=3, r=1, num_type=np.complex128, sigma=0), 1e-1)
])
def test_trispectrum_identity(setting, error):
    lambdas = np.array([1])
    vec = np.array([1, 2, 3])
    vec = vec / np.linalg.norm(vec)
    signal_vectors = SignalVectors(vectors=np.array([vec], dtype=setting.num_type))
    underlying_signal = UnderlyingSignal(signal_vectors=signal_vectors,
                                         signal_distribution_sample=SignalDistributionSample(
                                             lambdas=lambdas * setting.r, setting=setting))
    x_samples = underlying_signal.x_samples
    v_arr = underlying_signal.signal_vectors.vectors
    sigma = setting.sigma

    x_samples_fft = get_fft(default_sample_noising(default_sample_shuffle(x_samples, num_type=setting.num_type),
                                                   sigma=sigma))
    tri_from_data = signal_trispectrum_from_data(x_samples_fft)
    very_real_cov_hat = get_cov_hat_from_v_arr(v_arr, lambdas=lambdas)
    tri_from_cov = signal_trispectrum_from_cov_hat(very_real_cov_hat, sigma=sigma, num_type=setting.num_type)
    ratio = np.mean(tri_from_data / tri_from_cov)
    assert pytest.approx(np.linalg.norm(tri_from_data - tri_from_cov), abs=error) == 0


def test_get_H():
    lambdas = [1]
    L = 5
    x_samples, v_arr = default_x_samples_generation(200000, L=L, lambdas=lambdas)
    _, L = x_samples.shape
    cov_hat = get_cov_hat_from_v_arr(v_arr, lambdas)
    c_x = recover_c_x_estimator(default_sample_shuffle(x_samples))
    for i in range(L):
        h_ii_estimator = get_H_matrix(c_x, i, i)
        h_ii = get_H_matrix(cov_hat, i, i)
        assert pytest.approx(np.linalg.norm(h_ii - h_ii_estimator) / np.linalg.norm(h_ii), abs=1e-2) == 0


def test_solve_ambiguities_complex():
    lambdas = [1, 0.75, 0.5]
    r = len(lambdas)
    L = 10
    sigma = 0.1
    n = 1000
    x_samples, v_arr = default_x_samples_generation(n, L=L, lambdas=lambdas)
    c_x = recover_c_x_estimator(default_sample_noising(default_sample_shuffle(x_samples), sigma), sigma)
    print(calculate_error_up_to_circulant(c_x, get_cov_hat_from_v_arr(v_arr, lambdas)))

    cov_estimator = solve_ambiguities(c_x, r=r)

    cov_mat = get_cov_mat_from_v_arr(v_arr, lambdas)
    cov_estimator_no_fft = reverse_cov_fft(cov_estimator)
    error = calculate_error_up_to_shifts(cov_mat, cov_estimator_no_fft)
    print(f"Final error: {error}")
    assert pytest.approx(error, abs=1e-1) == 0


def test_solve_ambiguities_real():
    lambdas = [1, 0.75, 0.5]
    r = len(lambdas)
    L = 10
    sigma = 0.1
    n = 1000
    num_type = np.longdouble
    x_samples, v_arr = default_x_samples_generation(n, L=L, lambdas=lambdas, num_type=num_type)
    c_x = recover_c_x_estimator(default_sample_noising(default_sample_shuffle(x_samples),
                                                       sigma, num_type=num_type),
                                sigma, num_type=num_type)

    print(calculate_error_up_to_circulant(c_x, get_cov_hat_from_v_arr(v_arr, lambdas)))
    cov_estimator = solve_ambiguities(c_x, r=r)

    cov_mat = get_cov_mat_from_v_arr(v_arr, lambdas)
    cov_estimator_no_fft = reverse_cov_fft(cov_estimator)
    error = calculate_error_up_to_shifts(cov_mat, cov_estimator_no_fft)
    print(f"Final error: {error}")
    assert pytest.approx(error, abs=1e-1) == 0
