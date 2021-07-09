import pytest
import numpy as np
import scipy.linalg

from mra_lib import *


def test_trispectrum_sanity():
    data = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    data_fft = np.fft.fft(data)
    res1 = signal_trispectrum_from_data(data_fft, 0)
    res2 = signal_trispectrum_from_data(np.fft.fft(np.array([[1, 2, 3]])), 0)

    assert pytest.approx(res1) == res2


def test_error_sanity():
    err, phi = calculate_error(np.array([[1, 2], [1, 2]]), np.array([[1, 2], [1, 2]]))
    assert pytest.approx(err, abs=1e-5) == 0
    assert pytest.approx(phi[0], abs=1e-10) == 0


def test_error_with_circulant():
    original = np.array([[1, 2], [3, 4]])
    circulant = scipy.linalg.circulant([1, np.exp(1j * np.pi / 4)])
    distorted = original * circulant
    print(original)
    print(distorted)
    err, phi = calculate_error(distorted, original)
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
    err, phi = calculate_error(distorted, original)
    assert err > 1
    assert pytest.approx(phi[0]) == np.pi / 4


def test_trispectrum_anomaly():
    x = np.array([12321312,1,2,3,5,111112332,2,32
                  ])
    data = np.array([x])
    data_fft = np.fft.fft(data)
    ss = signal_trispectrum_from_signal(x)
    sd = signal_trispectrum_from_data(data_fft, 0)
    assert pytest.approx(ss) == sd * 2