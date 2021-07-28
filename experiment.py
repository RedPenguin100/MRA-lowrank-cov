import multiprocessing
import time

from data_structures import *
from mra_lib import fully_recover_c_x
from testing_utility import *
import numpy as np


class Experiment:
    def __init__(self, setting: Setting, lambdas):
        self.setting = setting
        self.lambdas = lambdas

        # Not executed yet.
        self.observed_signal = None
        self.error = np.inf
        self.execution_time = np.inf

    def run(self, queue=None):
        start_alg = time.time()
        if self.observed_signal is None:
            signal_ds = SignalDistributionSample(lambdas=self.lambdas, setting=self.setting)
            underlying_signal = UnderlyingSignal(signal_distribution_sample=signal_ds)
            self.observed_signal = ObservedSignal(underlying_signal=underlying_signal, sigma=self.setting.sigma)

        c_x_estimator = fully_recover_c_x(self.observed_signal.y_samples,
                                          sigma=self.setting.sigma,
                                          num_type=self.setting.num_type,
                                          r=self.setting.r)
        self.error = calculate_error_up_to_shifts(c_x_estimator, self.observed_signal.underlying_signal.get_cov_mat())
        end_alg = time.time()
        self.execution_time = end_alg - start_alg

        if queue is not None:
            queue.put(self.get_result())

    def get_result(self):
        return Result(self.error, self.observed_signal.setting, execution_time=self.execution_time)

    def __str__(self):
        str = ""
        str += "\n\nExperiment:\n"
        str += f"Settings: {self.observed_signal.setting} \n"
        str += f"Error: {self.error}"
        return str


class Result:
    def __init__(self, error: float, setting: Setting, execution_time=np.inf):
        self.error = error
        self.setting = setting
        self.execution_time = execution_time

    def __str__(self):
        str = ""
        str += "\n\nExperiment:\n"
        str += f"Settings: {self.setting} \n"
        str += f"Error: {self.error}\n"
        str += f"Time for algorithm: {self.execution_time}"
        return str


def run():
    return 5 + 3


if __name__ == "__main__":
    start = time.time()
    m = multiprocessing.Manager()
    experiments = m.Queue()
    cpu_count = multiprocessing.cpu_count()
    print(f"Machine has: {cpu_count} cpus. Running with {cpu_count - 1} processes.")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    print("Getting workers ready...\n")
    experiment_amount = 0
    for L in range(10, 50):
        for r in range(1, L):
            for sigma in [0, 0.01, 0.05]:
                experiment_amount += 1
                lambdas = np.random.uniform(0, 1, size=r)
                lambdas_normalized = lambdas / sum(lambdas)

                setting = Setting(n=100000, L=L, r=r, sigma=sigma, num_type=np.complex128)
                experiment = Experiment(setting=setting, lambdas=lambdas_normalized)
                res = pool.apply_async(experiment.run, args=(experiments,))
                print(f"Starting worker for {setting}")
    pool.close()
    pool.join()

    end = time.time()
    total_execution_time = end - start
    individual_time = 0
    for _ in range(experiment_amount):
        result = experiments.get()
        individual_time += result.execution_time
        print(result)
    print(f"Execution finished. Total time: {total_execution_time}")
    print("Total individual time: ", individual_time)
    print(f"Time saved: {individual_time - total_execution_time}")
