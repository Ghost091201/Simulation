import simpy
import random
import numpy as np
import scipy.stats as stats

# Parameters
SIM_TIME = 1000
NUM_SAMPLES = 10
WARMUP_PERIOD = 200

# Possible configuration options for distributions
interarrival_options = [
    ('exponential', 25),
    ('exponential', 22.5),
    ('uniform', (20, 30)),
    ('uniform', (20, 25))
]

preparation_options = [
    ('exponential', 40),
    ('uniform', (30, 50))
]

recovery_options = [
    ('exponential', 40),
    ('uniform', (30, 50))
]

# Fractional factorial design 2^(6-3)
factorial_experiments = [
    (3, 4, interarrival_options[0], preparation_options[0], recovery_options[0]),  # Base case
    (3, 4, interarrival_options[1], preparation_options[1], recovery_options[0]),
    (3, 5, interarrival_options[2], preparation_options[1], recovery_options[1]),
    (3, 5, interarrival_options[3], preparation_options[0], recovery_options[1]),
    (4, 4, interarrival_options[0], preparation_options[1], recovery_options[1]),
    (4, 4, interarrival_options[1], preparation_options[0], recovery_options[1]),
    (4, 5, interarrival_options[2], preparation_options[1], recovery_options[0]),
    (4, 5, interarrival_options[3], preparation_options[0], recovery_options[0]),
]


# Generates a random time based on the specified distribution
def generate_time(distribution_type, params):
    if distribution_type == 'exponential':
        return random.expovariate(1.0 / params)
    elif distribution_type == 'uniform':
        return random.uniform(*params)
    else:
        raise ValueError("Unsupported distribution type")


class Patient:
    # Iniziate the instance of the class Patient
    def __init__(self, env, name, hospital, interarrival_dist):
        self.env = env
        self.name = name
        self.hospital = hospital
        self.preparation_time = generate_time(*hospital.preparation_dist)
        self.surgery_time = generate_time('exponential', 20)
        self.recovery_time = generate_time(*hospital.recovery_dist)
        self.arrival_time = generate_time(*interarrival_dist)

    # Process the patient runs through the hospital
    def process(self):
        # Wait for arrival time
        yield self.env.timeout(self.arrival_time)

        # Enter preparation phase
        with self.hospital.preparation_facilities.request() as request:
            yield request
            yield self.env.timeout(self.preparation_time)

        # Enter surgery phase
        with self.hospital.operating_theater.request() as request:
            yield request
            self.hospital.operating_theater_utilization += self.surgery_time
            yield self.env.timeout(self.surgery_time)

        # Enter recovery phase
        with self.hospital.recovery_rooms.request() as request:
            yield request
            yield self.env.timeout(self.recovery_time)


class Hospital:
    # Iniziate the instance of the class Hospital
    def __init__(self, env, preparation_units, recovery_units, prep_dist, recov_dist):
        self.env = env
        self.preparation_dist = prep_dist
        self.recovery_dist = recov_dist
        self.preparation_facilities = simpy.Resource(env, capacity=preparation_units)
        self.operating_theater = simpy.Resource(env, capacity=1)
        self.recovery_rooms = simpy.Resource(env, capacity=recovery_units)
        self.operating_theater_utilization = 0
        self.blocking_events = 0
        self.total_examinations = 0

# generates a patient flow
def patient_generator(env, hospital, interarrival_dist):
    i = 0
    while True:
        patient = Patient(env, f"Patient {i}", hospital, interarrival_dist)
        env.process(patient.process())
        i += 1
        yield env.timeout(patient.arrival_time)

# SimulationProcess Monitor in eternal loop
def monitor(env, hospital, queue_lengths, blocking_probabilities):
    while True:
        if env.now > WARMUP_PERIOD:
            queue_length = len(hospital.preparation_facilities.queue)
            queue_lengths.append(queue_length)
            all_recovery_busy = len(hospital.recovery_rooms.queue) >= hospital.recovery_rooms.capacity
            hospital.total_examinations += 1
            if all_recovery_busy:
                hospital.blocking_events += 1
            blocking_probabilities.append(
                hospital.blocking_events / hospital.total_examinations if hospital.total_examinations > 0 else 0)
        yield env.timeout(10)

# Start the simulation
def run_simulation(preparation_units, recovery_units, interarrival_dist, prep_dist, recov_dist):
    env = simpy.Environment()
    hospital = Hospital(env, preparation_units=preparation_units, recovery_units=recovery_units,
                        prep_dist=prep_dist, recov_dist=recov_dist)
    queue_lengths = []
    blocking_probabilities = []
    env.process(patient_generator(env, hospital, interarrival_dist))
    env.process(monitor(env, hospital, queue_lengths, blocking_probabilities))
    env.run(until=SIM_TIME)

    avg_queue_length = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0
    avg_blocking_probability = sum(blocking_probabilities) / len(blocking_probabilities)
    avg_operating_utilization = hospital.operating_theater_utilization / (SIM_TIME - WARMUP_PERIOD)

    return avg_queue_length, avg_blocking_probability, avg_operating_utilization

# Gather the values for the output of the configurations
def perform_experiments(factorial_experiments, num_samples):
    results = {exp: {"avg_queue_lengths": [], "avg_blocking_probabilities": [], "avg_utilizations": []} for exp in
               factorial_experiments}

    for i in range(num_samples):
        random.seed(i)
        for experiment in factorial_experiments:
            preparation_units, recovery_units, interarrival_dist, prep_dist, recov_dist = experiment
            res = run_simulation(preparation_units, recovery_units, interarrival_dist, prep_dist, recov_dist)
            results[experiment]["avg_queue_lengths"].append(res[0])
            results[experiment]["avg_blocking_probabilities"].append(res[1])
            results[experiment]["avg_utilizations"].append(res[2])

    return results

# Gathering values for the 0.95 confidence level
def compute_statistics(results, num_samples):
    confidence_level = 0.95
    statistics = {}

    for exp, data in results.items():
        queue_mean = np.mean(data["avg_queue_lengths"])
        queue_std_err = stats.sem(data["avg_queue_lengths"])

        blocking_mean = np.mean(data["avg_blocking_probabilities"])
        blocking_std_err = stats.sem(data["avg_blocking_probabilities"])

        util_mean = np.mean(data["avg_utilizations"])
        util_std_err = stats.sem(data["avg_utilizations"])

        t_value = stats.t.ppf((1 + confidence_level) / 2, num_samples - 1)

        statistics[exp] = {
            "queue_length_mean": queue_mean,
            "queue_length_ci": (queue_mean - t_value * queue_std_err, queue_mean + t_value * queue_std_err),
            "blocking_prob_mean": blocking_mean,
            "blocking_prob_ci": (
                blocking_mean - t_value * blocking_std_err, blocking_mean + t_value * blocking_std_err),
            "utilization_mean": util_mean,
            "utilization_ci": (util_mean - t_value * util_std_err, util_mean + t_value * util_std_err),
        }
    return statistics


results = perform_experiments(factorial_experiments, NUM_SAMPLES)
statistics = compute_statistics(results, NUM_SAMPLES)

# Print results
for exp, stats in statistics.items():
    print(f"Experiment {exp} Statistics:")
    print(
        f"   Average Queue Length: {stats['queue_length_mean']:.2f} (95% CI: {stats['queue_length_ci'][0]:.2f}, {stats['queue_length_ci'][1]:.2f})")
    print(
        f"   Average Blocking Probability: {stats['blocking_prob_mean']:.4f} (95% CI: {stats['blocking_prob_ci'][0]:.4f}, {stats['blocking_prob_ci'][1]:.4f})")
    print(
        f"   Average Utilization: {stats['utilization_mean']:.2f} (95% CI: {stats['utilization_ci'][0]:.2f}, {stats['utilization_ci'][1]:.2f})")
    print()
