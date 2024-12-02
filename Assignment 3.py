import simpy
import random
import numpy as np
import scipy.stats as stats

# Parameters
ARRIVAL_INTERVAL = 25
PREPARATION_INTERVAL = 40
SURGERY_TIME_AVG = 20
RECOVERY_TIME_AVG = 40
SIM_TIME = 1000
NUM_SAMPLES = 20
WARMUP_PERIOD = 200

# Configuration parameters for different setups
configurations = [
    (3, 4),  # 3 preparation rooms, 4 recovery rooms
    (3, 5),  # 3 preparation rooms, 5 recovery rooms
    (4, 5)  # 4 preparation rooms, 5 recovery rooms
]


# Generate a random service time based on an exponential distribution
def generate_service_time(avg_time):
    return random.expovariate(1.0 / avg_time)


class Patient:
    # Iniziate the instance of the class Patient
    def __init__(self, env, name, hospital):
        self.env = env # create environment
        self.name = name # create name of patient
        self.hospital = hospital # create hospital of the patient
        self.preparation_time = generate_service_time(hospital.preparation_time_avg) # personal preparation time
        self.surgery_time = generate_service_time(SURGERY_TIME_AVG) # personal surgery time
        self.recovery_time = generate_service_time(RECOVERY_TIME_AVG) # personal recovery time

    # Process the patient runs through the hospital
    def process(self):
        # Enter preparation phase
        with self.hospital.preparation_facilities.request() as request:
            yield request
            yield self.env.timeout(self.preparation_time) # time needed for preparation; different for every instance of Patient

        # Enter surgery phase
        with self.hospital.operating_theater.request() as request:
            yield request
            self.hospital.operating_theater_utilization += self.surgery_time
            yield self.env.timeout(self.surgery_time) # time needed for surgery; different for every instance of Patient

        # Enter recovery phase
        with self.hospital.recovery_rooms.request() as request:
            yield request
            yield self.env.timeout(self.recovery_time) # time needed for recovery; different for every instance of Patient


class Hospital:
    # Iniziate the instance of the class Hospital
    def __init__(self, env, preparation_units, recovery_units):
        self.env = env
        self.preparation_time_avg = PREPARATION_INTERVAL
        self.preparation_facilities = simpy.Resource(env, capacity=preparation_units)
        self.operating_theater = simpy.Resource(env, capacity=1)
        self.recovery_rooms = simpy.Resource(env, capacity=recovery_units)
        self.operating_theater_utilization = 0
        self.max_utilization = 0
        self.blocking_events = 0
        self.total_examinations = 0

# generates a steady patient flow
def patient_generator(env, hospital):
    i = 0
    while True:
        yield env.timeout(ARRIVAL_INTERVAL)
        i += 1
        patient = Patient(env, f"Patient {i}", hospital)
        env.process(patient.process())

# SimulationProcess Monitor in eternal loop
def monitor(env, hospital, queue_lengths, blocking_probabilities):
    while True:
        if env.now > WARMUP_PERIOD:
            queue_length = len(hospital.preparation_facilities.queue)
            queue_lengths.append(queue_length)
            all_recovery_busy = len(hospital.recovery_rooms.queue) == hospital.recovery_rooms.capacity
            hospital.total_examinations += 1
            if all_recovery_busy:
                hospital.blocking_events += 1
            blocking_probabilities.append(
                hospital.blocking_events / hospital.total_examinations if hospital.total_examinations > 0 else 0)
        yield env.timeout(10)

# Start the simulation
def run_simulation(preparation_units, recovery_units):
    # Creates separate environment for each run
    env = simpy.Environment()
    hospital = Hospital(env, preparation_units=preparation_units, recovery_units=recovery_units)
    queue_lengths = []
    blocking_probabilities = []
    env.process(patient_generator(env, hospital))
    env.process(monitor(env, hospital, queue_lengths, blocking_probabilities))
    env.run(until=SIM_TIME)

    avg_queue_length = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0
    avg_blocking_probability = sum(blocking_probabilities) / len(blocking_probabilities)
    avg_operating_utilization = hospital.operating_theater_utilization / (SIM_TIME - WARMUP_PERIOD)

    return avg_queue_length, avg_blocking_probability, avg_operating_utilization

# Gather the values for the output of the configurations
def perform_experiments(configurations, num_samples):
    results = {config: {"avg_queue_lengths": [], "avg_blocking_probabilities": [], "avg_utilizations": []} for config in
               configurations}

    for i in range(num_samples):
        random.seed(i)  # Seed each paired sample with the same seed for paired comparison
        for config in configurations:
            preparation_units, recovery_units = config
            res = run_simulation(preparation_units, recovery_units)
            results[config]["avg_queue_lengths"].append(res[0])
            results[config]["avg_blocking_probabilities"].append(res[1])
            results[config]["avg_utilizations"].append(res[2])

    return results

# Gathering values for the 0.95 confidence level
def compute_statistics(results, num_samples):
    confidence_level = 0.95
    statistics = {}

    for config, data in results.items():
        queue_mean = np.mean(data["avg_queue_lengths"])
        queue_std_err = stats.sem(data["avg_queue_lengths"])

        blocking_mean = np.mean(data["avg_blocking_probabilities"])
        blocking_std_err = stats.sem(data["avg_blocking_probabilities"])

        util_mean = np.mean(data["avg_utilizations"])
        util_std_err = stats.sem(data["avg_utilizations"])

        t_value = stats.t.ppf((1 + confidence_level) / 2, num_samples - 1)

        statistics[config] = {
            "queue_length_mean": queue_mean,
            "queue_length_ci": (queue_mean - t_value * queue_std_err, queue_mean + t_value * queue_std_err),
            "blocking_prob_mean": blocking_mean,
            "blocking_prob_ci": (
            blocking_mean - t_value * blocking_std_err, blocking_mean + t_value * blocking_std_err),
            "utilization_mean": util_mean,
            "utilization_ci": (util_mean - t_value * util_std_err, util_mean + t_value * util_std_err),
        }
    return statistics

# Gathering values for the comparison of the different configurations
def compare_configurations(configurations, results, num_samples):
    differences = {}
    for (config_a, config_b) in [(configurations[0], configurations[1]), (configurations[1], configurations[2])]:
        queue_diffs = [a - b for a, b in
                       zip(results[config_a]["avg_queue_lengths"], results[config_b]["avg_queue_lengths"])]
        block_diffs = [a - b for a, b in zip(results[config_a]["avg_blocking_probabilities"],
                                             results[config_b]["avg_blocking_probabilities"])]
        util_diffs = [a - b for a, b in
                      zip(results[config_a]["avg_utilizations"], results[config_b]["avg_utilizations"])]

        queue_diff_mean = np.mean(queue_diffs)
        queue_diff_std_err = stats.sem(queue_diffs)

        blocking_diff_mean = np.mean(block_diffs)
        blocking_diff_std_err = stats.sem(block_diffs)

        util_diff_mean = np.mean(util_diffs)
        util_diff_std_err = stats.sem(util_diffs)

        t_value = stats.t.ppf((1 + 0.95) / 2., num_samples - 1)

        differences[(config_a, config_b)] = {
            "queue_length_difference_mean": queue_diff_mean,
            "queue_length_difference_ci": (
            queue_diff_mean - t_value * queue_diff_std_err, queue_diff_mean + t_value * queue_diff_std_err),
            "blocking_prob_difference_mean": blocking_diff_mean,
            "blocking_prob_difference_ci": (
            blocking_diff_mean - t_value * blocking_diff_std_err, blocking_diff_mean + t_value * blocking_diff_std_err),
            "utilization_difference_mean": util_diff_mean,
            "utilization_difference_ci": (
            util_diff_mean - t_value * util_diff_std_err, util_diff_mean + t_value * util_diff_std_err),
        }

    return differences


results = perform_experiments(configurations, NUM_SAMPLES)
statistics = compute_statistics(results, NUM_SAMPLES)
differences = compare_configurations(configurations, results, NUM_SAMPLES)

# Print results
for config, stats in statistics.items():
    print(f"Configuration {config} Statistics:")
    print(
        f"   Average Queue Length: {stats['queue_length_mean']:.2f} (95% CI: {stats['queue_length_ci'][0]:.2f}, {stats['queue_length_ci'][1]:.2f})")
    print(
        f"   Average Blocking Probability: {stats['blocking_prob_mean']:.4f} (95% CI: {stats['blocking_prob_ci'][0]:.4f}, {stats['blocking_prob_ci'][1]:.4f})")
    print(
        f"   Average Utilization: {stats['utilization_mean']:.2f} (95% CI: {stats['utilization_ci'][0]:.2f}, {stats['utilization_ci'][1]:.2f})")
    print()

for configs, diff in differences.items():
    print(f"Difference between Configurations {configs}:")
    print(
        f"   Queue Length Difference Mean: {diff['queue_length_difference_mean']:.2f} (95% CI: {diff['queue_length_difference_ci'][0]:.2f}, {diff['queue_length_difference_ci'][1]:.2f})")
    print(
        f"   Blocking Probability Difference Mean: {diff['blocking_prob_difference_mean']:.4f} (95% CI: {diff['blocking_prob_difference_ci'][0]:.4f}, {diff['blocking_prob_difference_ci'][1]:.4f})")
    print(
        f"   Utilization Difference Mean: {diff['utilization_difference_mean']:.2f} (95% CI: {diff['utilization_difference_ci'][0]:.2f}, {diff['utilization_difference_ci'][1]:.2f})")
    print()