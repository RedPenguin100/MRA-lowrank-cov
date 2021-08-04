import pytest
import scipy.linalg

from mra_lib import *
from data_structures import *
from testing_utility import *


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
    (Setting(n=100000, L=3, r=1, num_type=np.longdouble, sigma=0), 1e-1),
    (Setting(n=100000, L=3, r=1, num_type=np.complex128, sigma=0), 1e-1)
])
def test_trispectrum_identity(setting, error):
    lambdas = np.array([0.3])
    vec = np.array([1, 2, 3])
    signal_vectors = SignalVectors(vectors=np.array([vec], dtype=setting.num_type))
    underlying_signal = UnderlyingSignal(signal_vectors=signal_vectors,
                                         signal_distribution_sample=SignalDistributionSample(
                                             lambdas=lambdas * setting.r, setting=setting))
    observed_signal = ObservedSignal(underlying_signal=underlying_signal, sigma=setting.sigma)

    y_samples_fft = np.fft.fft(observed_signal.y_samples)
    tri_from_data = signal_trispectrum_from_data(y_samples_fft)
    tri_from_cov = signal_trispectrum_from_cov_hat(underlying_signal.get_cov_hat(), sigma=setting.sigma,
                                                   num_type=setting.num_type)
    actual_error = np.linalg.norm(tri_from_data - tri_from_cov)
    print(actual_error)
    assert pytest.approx(actual_error, abs=error) == 0


def test_get_H():
    lambdas = [1]
    setting = Setting(n=200000, L=5, r=len(lambdas))
    signal_ds = SignalDistributionSample(lambdas=lambdas, setting=setting)
    underlying_signal = UnderlyingSignal(signal_distribution_sample=signal_ds)
    observed_signal = ObservedSignal(underlying_signal=underlying_signal, sigma=setting.sigma)

    c_x = recover_c_x_estimator(observed_signal.y_samples)
    for i in range(setting.L):
        h_ii_estimator = get_H_matrix(c_x, i, i)
        h_ii = get_H_matrix(underlying_signal.get_cov_hat(), i, i)
        assert pytest.approx(np.linalg.norm(h_ii - h_ii_estimator) / np.linalg.norm(h_ii), abs=1e-2) == 0


@pytest.mark.parametrize('setting,error', [
    (Setting(n=5000, r=3, L=10, sigma=0.1, num_type=np.complex128), 1e-1),
    (Setting(n=5000, r=3, L=10, sigma=0.1, num_type=np.longdouble), 1e-1),
    (Setting(n=5000, r=3, L=10, sigma=0.0, num_type=np.complex128), 1e-1),
    (Setting(n=5000, r=3, L=10, sigma=0.0, num_type=np.longdouble), 1e-1),
    (Setting(n=50000, r=3, L=10, sigma=0.0, num_type=np.complex128), 1e-1),
    (Setting(n=50000, r=3, L=10, sigma=0.0, num_type=np.longdouble), 1e-1),
    (Setting(n=5000, r=2, L=5, sigma=0.0, num_type=np.complex128), 1e-1),
    (Setting(n=5000, r=2, L=5, sigma=0.0, num_type=np.longdouble), 1e-1),
])
def test_solve_ambiguities(setting, error):
    signal_ds = SignalDistributionSample(setting=setting)
    underlying_signal = UnderlyingSignal(signal_distribution_sample=signal_ds)
    observed_signal = ObservedSignal(underlying_signal=underlying_signal, sigma=setting.sigma)

    c_x = recover_c_x_estimator(observed_signal.y_samples, setting.sigma, num_type=setting.num_type)
    print(calculate_error_up_to_circulant(c_x, underlying_signal.get_cov_hat()))

    cov_estimator = solve_ambiguities(c_x, r=setting.r)

    error = calculate_error_up_to_shifts(underlying_signal.get_cov_mat(), cov_estimator)
    print(f"Final error: {error}")
    assert pytest.approx(error, abs=error) == 0


@pytest.mark.parametrize('setting,error', [
    (Setting(n=5000, r=None, L=10, sigma=0.1, num_type=np.complex128), 1e-1),
    (Setting(n=5000, r=None, L=10, sigma=0.1, num_type=np.longdouble), 1e-1),
])
def test_solve_ambiguities_custom_lambdas(setting, error):
    lambdas = [1, 0.75, 0.5]
    setting.r = len(lambdas)
    print(setting)

    signal_ds = SignalDistributionSample(setting=setting, lambdas=lambdas)
    underlying_signal = UnderlyingSignal(signal_distribution_sample=signal_ds)
    observed_signal = ObservedSignal(underlying_signal=underlying_signal, sigma=setting.sigma)

    c_x = recover_c_x_estimator(observed_signal.y_samples, setting.sigma, num_type=setting.num_type)
    print(calculate_error_up_to_circulant(c_x, underlying_signal.get_cov_hat()))

    cov_estimator = solve_ambiguities(c_x, r=setting.r)

    error = calculate_error_up_to_shifts(underlying_signal.get_cov_mat(), cov_estimator)
    print(f"Final error: {error}")
    assert pytest.approx(error, abs=error) == 0
