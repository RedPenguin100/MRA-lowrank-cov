import warnings
import numpy as np

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
            self._normalize_vectors()
            return

        if setting is None:
            raise ValueError("Error: can't generate signal vectors with setting=None.")
        self.vectors = SignalVectors.generate_vectors(generation_method, setting)
        self._normalize_vectors()

    def _normalize_vectors(self):
        self.vectors = self.vectors / np.linalg.norm(self.vectors)

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
        fft_complete_underlying = np.fft.fft((sqrt_lambdas * self.signal_vectors.vectors.T).T)
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

        self.distribution = 'normal' if distribution is None or distribution == 'default' or distribution == 'normal' else distribution
        self.underlying_signal = underlying_signal
        self.x_samples = self.underlying_signal.x_samples
        self.shuffle_method = self._get_shuffle_method(shuffle_method)
        self.sigma = 0 if sigma is None else sigma
        self.setting = self.underlying_signal.setting
        self.setting.sigma = self.sigma
        self.num_type = self.setting.num_type

        # Calculate the observed signal.
        shuffled_samples = self.shuffle_method(self.x_samples)
        if sigma == 0:
            self.y_samples = shuffled_samples
        else:
            self.y_samples = shuffled_samples + self._get_noise(size=self.x_samples.shape,
                                                                num_type=self.setting.num_type,
                                                                sigma=self.sigma,
                                                                distribution=self.distribution)

    @staticmethod
    def _get_shuffle_method(shuffle_method):
        if shuffle_method is None or shuffle_method == 'default':
            return default_sample_shuffle
        elif shuffle_method == 'no_shuffle':
            return lambda x: x
        raise ValueError(f"Unknown shuffle_method: {shuffle_method}")

    @staticmethod
    def _get_noise(size, num_type, sigma, distribution=None):
        if distribution is None or distribution == 'default' or distribution == 'normal':
            return normal_distribution(mu=np.zeros_like(sigma), sigma=sigma, size=size, num_type=num_type)
        raise ValueError(f"Unknown noise distribution method: {distribution}")


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
