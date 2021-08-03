import copy
import multiprocessing
import time

from data_structures import *
from mra_lib import fully_recover_c_x
from testing_utility import *
import numpy as np


class Experiment:
    def __init__(self, setting: Setting, lambdas, key=None):
        self.key = key
        self.setting = setting
        self.lambdas = lambdas

        # Not executed yet.
        self.observed_signal = None
        self.error = np.inf
        self.execution_time = np.inf
        self.signal_generation_distribution = None
        self.noise_distribution = None

    def run(self, queue=None):
        start_alg = time.time()
        if self.observed_signal is None:
            signal_ds = SignalDistributionSample(lambdas=self.lambdas, setting=self.setting,
                                                 generation_method=self.signal_generation_distribution)
            underlying_signal = UnderlyingSignal(signal_distribution_sample=signal_ds)
            self.observed_signal = ObservedSignal(underlying_signal=underlying_signal, sigma=self.setting.sigma,
                                                  distribution=self.noise_distribution)

        c_x_estimator = fully_recover_c_x(self.observed_signal.y_samples,
                                          sigma=self.setting.sigma,
                                          num_type=self.setting.num_type,
                                          r=self.setting.r)
        self.error = calculate_error_up_to_shifts(c_x_estimator,
                                                  cov_real=self.observed_signal.underlying_signal.get_cov_mat())
        end_alg = time.time()
        self.execution_time = end_alg - start_alg

        if queue is not None:
            queue.put((self.key, self.get_result()))

    def get_result(self):
        return Result(self.error, self.observed_signal.setting, execution_time=self.execution_time,
                      noise_distribution=self.noise_distribution)


class Result:
    def __init__(self, error: float, setting: Setting, execution_time=np.inf,
                 noise_distribution=None):
        self.error = error
        self.setting = setting
        self.execution_time = execution_time
        self.noise_distribution = noise_distribution

    def __str__(self):
        str = ""
        str += "\n\nExperiment:\n"
        str += f"Settings: {self.setting} \n"
        str += f"Error: {self.error}\n"
        str += f"Time for algorithm: {self.execution_time}\n"
        if self.noise_distribution is not None:
            str += f"Noise distribution: {self.noise_distribution}"
        return str


if __name__ == "__main__":
    start = time.time()
    m = multiprocessing.Manager()
    experiments = m.Queue()
    cpu_count = multiprocessing.cpu_count()
    print(f"Machine has: {cpu_count} cpus. Running with {cpu_count - 1} processes.")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    print("Getting workers ready...\n")
    experiment_amount = 0
    d = {}
    for L in range(10, 11):
        for r in [1]:
            for sigma in np.sqrt([0, 0.05, 0.1]):
                for n in [100,
                          310, 1000, 3100, 10000, 31000, 100000, 310000,
                          1000000, 3100000
                          ]:
                    for distribution in ['laplace', 'normal']:
                        key = (L, r, sigma, n, distribution)
                        experiment_amount += 1
                        lambdas = np.random.uniform(0, 1, size=r)
                        lambdas_normalized = lambdas / sum(lambdas)

                        setting = Setting(n=n, L=L, r=r, sigma=sigma, num_type=np.complex128)
                        experiment = Experiment(setting=setting, lambdas=lambdas_normalized, key=key)
                        experiment.noise_distribution = distribution
                        # experiment.run(experiments)
                        pool.apply_async(experiment.run, args=(experiments,))
                        # pool.apply_async(experiment.run, args=(experiments,))

                        print(f"Starting worker for {setting}")
    print(f"Running {experiment_amount} experiments. Hold tight!")
    pool.close()
    pool.join()
    print("Done joining")
    end = time.time()
    total_execution_time = end - start
    individual_time = 0
    results = []
    for _ in range(experiment_amount):
        key, result = experiments.get()
        individual_time += result.execution_time
        results.append(result)
        d[key] = (result.error, result.execution_time)
    data = ""
    for result in results:
        print(result)
        data += str(result)

    print(f"Execution finished. Total time: {total_execution_time}")
    data += f"Execution finished. Total time: {total_execution_time}\n"
    print("Total individual time: ", individual_time)
    data += f"Total individual time: {individual_time}\n"
    print(f"Time saved: {individual_time - total_execution_time}")
    data += f"Time saved: {individual_time - total_execution_time}\n"
    with open(str(time.time()) + '.txt', 'w+') as f:
        f.write(data)

    for res in results:
        if res.noise_distribution is not None:
            print("({},{},{},{})".format(res.noise_distribution, res.setting, res.error, res.execution_time))
        else:
            print("({},{},{})".format(res.setting, res.error, res.execution_time))
    print(d)
