import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize
import warnings

np.set_printoptions(linewidth=np.inf)

REAL_TYPES = [np.double, np.longdouble, np.float64, np.float128]
COMPLEX_TYPES = [np.complex, np.complex128]


class Setting:
    def __init__(self, n: int, L: int, r: int, sigma: float = 0.0, num_type=np.complex128):
        self.n = n
        self.L = L
        self.r = r if r is not None else 0
        self.sigma = sigma
        self.num_type = num_type
        if self.r > self.L:
            raise ValueError(f"Error: r={self.r} cannot be larger than L={self.L}")
        if self.r == 0:
            warnings.warn("Inside Setting: self.r={0}. Please be sure to set it later.")
        if self.sigma < 0:
            raise ValueError(f"Error: sigma={self.sigma} cannot be negative!")

    def __str__(self):
        return f"Setting: n={self.n}, L={self.L}, r={self.r}, sigma={self.sigma}, num_type={self.num_type}"

    @staticmethod
    def get_default():
        return Setting(n=10000, L=5, r=1, sigma=0, num_type=np.complex128)


class SignalVectors:
    def __init__(self, vectors=None, generation_method=None, setting: Setting = None):
        if vectors is not None:
            self.vectors = vectors
            return

        if setting is None:
            raise ValueError("Error: can't generate signal vectors with setting=None.")
        self.vectors = SignalVectors.generate_vectors(generation_method, setting)

    @staticmethod
    def generate_vectors(generation_method, setting: Setting):
        L, r, num_type = setting.L, setting.r, setting.num_type
        if generation_method is None or generation_method == 'default':
            v_arr = np.random.uniform(0, 1, size=(r, L))
            return v_arr
        raise ValueError(f"Unknown generation method {generation_method} for signal vectors.")


class SignalDistributionSample:
    def __init__(self, lambdas, setting: Setting, generation_method=None):
        self.lambdas = lambdas
        self.setting = setting
        self.validate(lambdas, self.setting)
        self.mus = np.zeros(setting.r)
        if generation_method is None or generation_method == 'default':
            self.distribution = 'normal'
            self.sample = normal_distribution(mu=self.mus, sigma=np.sqrt(lambdas), size=(setting.n, setting.r),
                                              num_type=setting.num_type)
        else:
            raise ValueError(f"Unknown generation method: {generation_method}")

    @staticmethod
    def validate(lambdas, setting: Setting):
        if setting is None:
            raise ValueError("Setting cannot be none when generating SignalDistributionSample.")
        if setting.r != len(lambdas):
            raise ValueError(f"Inconsistent lambda values: {len(lambdas)} and r in setting: {setting.r}")


class UnderlyingSignal:
    def __init__(self, signal_distribution_sample: SignalDistributionSample,
                 signal_vectors: SignalVectors = None,
                 x_samples=None):

        self.signal_distribution_sample = signal_distribution_sample
        self.setting = signal_distribution_sample.setting
        self.signal_vectors = signal_vectors if signal_vectors is not None else SignalVectors(setting=self.setting)

        # If we are just given the underlying signal, we will take it.
        if x_samples is not None:
            self.x_samples = x_samples
            return

        generated_x_samples = np.zeros((self.setting.n, self.setting.L), dtype=self.setting.num_type)
        # TODO: make efficient
        for i in range(self.setting.r):
            v_i = self.signal_vectors.vectors[i, :]
            generated_x_samples += np.outer(self.signal_distribution_sample.sample[:, i], v_i)
        self.x_samples = generated_x_samples
        self.cov_hat = None
        self.cov_mat = None

    def get_cov_hat(self):
        if self.cov_hat is not None:
            return self.cov_hat
        sqrt_lambdas = np.sqrt(self.signal_distribution_sample.lambdas)
        fft_complete_underlying = get_fft((sqrt_lambdas * self.signal_vectors.vectors.T).T)
        self.cov_hat = np.sum(np.einsum('bi,bo->bio',
                                        fft_complete_underlying, fft_complete_underlying.conj()), axis=0)
        return self.cov_hat

    def get_cov_mat(self):
        if self.cov_mat is not None:
            return self.cov_mat
        sqrt_lambdas = np.sqrt(self.signal_distribution_sample.lambdas)
        complete_underlying = (sqrt_lambdas * self.signal_vectors.vectors.T).T
        self.cov_mat = np.sum(np.einsum('bi,bo->bio',
                                        complete_underlying, complete_underlying.conj()), axis=0)
        return self.cov_mat


class ObservedSignal:
    def __init__(self, y_samples=None, underlying_signal: UnderlyingSignal = None,
                 shuffle_method=None, sigma: float = None, distribution=None):
        if y_samples is not None:
            self.y_samples = y_samples
            self.sigma = None
            self.shuffle_method = None
            return

        self.distribution = 'normal' if distribution is None \
                                        or distribution == 'default' \
                                        or distribution == 'normal' else distribution
        self.underlying_signal = underlying_signal
        self.x_samples = self.underlying_signal.x_samples
        self.shuffle_method = self._get_shuffle_method(shuffle_method)
        self.sigma = 0 if sigma is None else sigma
        self.setting = self.underlying_signal.setting
        self.setting.sigma = self.sigma

        # Calculate the observed signal.
        self.y_samples = self.shuffle_method(self.x_samples) + self._get_noise(size=self.x_samples.shape,
                                                                               num_type=self.setting.num_type,
                                                                               sigma=self.sigma,
                                                                               distribution=self.distribution)

    @staticmethod
    def _get_shuffle_method(shuffle_method):
        if shuffle_method is None or shuffle_method == 'default':
            return default_sample_shuffle
        raise ValueError(f"Unknown shuffle_method: {shuffle_method}")

    @staticmethod
    def _get_noise(size, num_type, sigma=0, distribution=None):
        if distribution is None or distribution == 'default' or distribution == 'normal':
            return normal_distribution(mu=np.zeros_like(sigma), sigma=sigma, size=size, num_type=num_type)
        raise ValueError(f"Unknown noise distribution method: {distribution}")


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


def diag_wrap(matrix, k):
    """
    Function returns the kth "main wrapped diagonal" of a square matrix.
    """
    L, L2 = matrix.shape
    assert L == L2
    return np.concatenate((np.diag(matrix, k), np.diag(matrix, k - L)))


def get_fft(x):
    return np.fft.fft(x)


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


def calculate_error_up_to_circulant(c_x_est, cov_fft):
    """
    Calculates the error between C_x estimator and FFT of the original covariance matrix
    disregarding the circulant angles multiplication.
    May be viewed as error calculation for algorithm 1.
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


def calculate_error_up_to_shifts(cov_estimator, cov_real):
    """
    Function calculates the error of algorithms 1 + 2.
    """
    error = np.inf
    L, L1 = cov_estimator.shape
    assert L == L1
    assert (L, L) == cov_real.shape

    for i in range(L):
        error = np.min((np.linalg.norm(cov_estimator - cov_real, ord='fro'), error))
        cov_real = np.roll(cov_real, (1, 1), axis=(0, 1))
    return error / np.linalg.norm(cov_real, ord='fro')


def recover_c_x_estimator(data, sigma=0, num_type=np.complex128):
    """
    Implementation of algorithm 1 in the paper.
    """
    N, L = data.shape

    data_fft = get_fft(data)
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
    d_estimates = np.array(d_estimates)
    cov_estimator = create_matrix_from_diagonals(d_estimates) - L * np.diag(np.full(L, sigma ** 2))
    return cov_estimator


def default_sample_shuffle(x_samples):
    N, L = x_samples.shape
    rolled_samples = np.zeros(x_samples.shape, dtype=x_samples.dtype)
    for i in range(N):
        rolled_samples[i] = np.roll(x_samples[i], i % L)

    return rolled_samples


def normal_distribution(mu, sigma, size, num_type):
    if num_type in COMPLEX_TYPES:
        return np.random.normal(mu, sigma / np.sqrt(2), size=size) + \
               np.random.normal(mu, sigma / np.sqrt(2), size=size) * 1j
    if num_type in REAL_TYPES:
        return np.random.normal(mu, sigma, size=size)
    raise ValueError(f"Invalid num_type for normal distribution: {num_type}")


def default_sample_noising(x_samples, sigma=0, num_type=np.complex128):
    return x_samples + normal_distribution(mu=0, sigma=sigma, size=x_samples.shape, num_type=num_type)


def get_cov(x_samples):
    return np.mean(np.einsum('bi,bo->bio', x_samples, x_samples.conj()), axis=0)


def get_cov_hat(x_samples):
    """
    Cov hat is the covariance of the fourier transformed vectors,
    not the fourier transform of the covariance matrix.
    """
    fft_samples = get_fft(x_samples)
    return get_cov(fft_samples)


def get_H_matrix(C_x, i, j):
    # TODO: roll is extremely inefficient, improve efficiency.
    rotated_c_x = np.roll(C_x, (-i, -j), axis=(0, 1))
    return C_x * rotated_c_x.conj()  # Hadamard product


def get_V_matrix(H, r_squared):
    """
    :param H: Hermitian or real symmetric matrix.
    """
    L1, L = H.shape
    assert L1 == L
    w, v = np.linalg.eig(H)
    arg_max_eigvals = w.argsort()[-r_squared:][::-1]
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
        r_squared = L - 1
    else:
        r_squared = r ** 2

    V_array = np.zeros((L, L, r_squared), dtype=np.complex128)
    for i in range(L):
        H_ii = get_H_matrix(C_x, i, i)
        V_array[i] = get_V_matrix(H_ii, r_squared=r_squared)

    Z_array = np.zeros((L, L ** 2, r_squared ** 2), dtype=np.complex128)
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
    return cov_estimator


if __name__ == "__main__":
    np.random.seed(42)
    L = 10
    lambdas = [1, 0.75, 0.5]

    setting = Setting(n=10000, L=10, r=len(lambdas))
    signal_ds = SignalDistributionSample(lambdas, setting)
    underlying_signal = UnderlyingSignal(signal_ds)

    c_x_estimator = recover_c_x_estimator(default_sample_shuffle(underlying_signal.x_samples))
    print(calculate_error_up_to_circulant(c_x_estimator, underlying_signal.get_cov_hat()))
