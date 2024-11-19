import simpy
import random

# Parameters
ARRIVAL_INTERVAL = 10
PREPARATION_UNITS = 3
PREPARATION_TIME_AVG = 40
SURGERY_UNITS = 1
SURGERY_TIME_AVG = 30
RECOVERY_UNITS = 3
RECOVERY_TIME_AVG = 40
SIM_TIME = 500

# Generate a random service time based on an exponential distribution
def generate_service_time(avg_time):
    return random.expovariate(1.0 / avg_time)

class Patient:
    # Iniziate the instance of the class Patient
    def __init__(self, env, name, hospital):
        self.env = env # create environment
        self.name = name # create name of patient
        self.hospital = hospital # create hospital of the patient
        self.preparation_time = generate_service_time(PREPARATION_TIME_AVG) # personal preparation time
        self.surgery_time = generate_service_time(SURGERY_TIME_AVG) # personal surgery time
        self.recovery_time = generate_service_time(RECOVERY_TIME_AVG) # personal recovery time
    
    # Process the patient runs through the hospital
    def process(self):
        # Enter preparation phase
        with self.hospital.preparation_facilities.request() as request:
            yield request
            yield self.env.timeout(self.preparation_time) # time needed for preparation; different for every instance of Patient
            print(f"{self.name} finished preparation at {self.env.now:.2f}")

        # Enter surgery phase
        with self.hospital.operating_theater.request() as request:
            yield request
            self.hospital.operating_theater_utilization += self.surgery_time # update parameter for calculating utilization of the operating theater
            yield self.env.timeout(self.surgery_time) # time needed for surgery; different for every instance of Patient
            print(f"{self.name} finished surgery at {self.env.now:.2f}")

        # Enter recovery phase
        with self.hospital.recovery_rooms.request() as request:
            yield request
            yield self.env.timeout(self.recovery_time) # time needed for recovery; different for every instance of Patient
            print(f"{self.name} finished recovery at {self.env.now:.2f}")


class Hospital:
    # Iniziate the instance of the class Hospital
    def __init__(self, env):
        self.env = env # create environment
        self.preparation_facilities = simpy.Resource(env, capacity=PREPARATION_UNITS) # creating number of preparation pools
        self.operating_theater = simpy.Resource(env, capacity=SURGERY_UNITS) # creating number of surgery pools
        self.recovery_rooms = simpy.Resource(env, capacity=RECOVERY_UNITS) # creating number of recovery pools
        self.operating_theater_utilization = 0  # To measure the utilization
        self.max_utilization = 0  # Maximum observed utilization

# generates a steady patient flow
def patient_generator(env, hospital):
    i = 0
    while True:
        yield env.timeout(ARRIVAL_INTERVAL)
        i += 1
        patient = Patient(env, f"Patient {i}", hospital)
        env.process(patient.process())

# SimulationProcess Monitor in eternal loop
def monitor(env, hospital):
    while True:
        queue_length = len(hospital.preparation_facilities.queue)
        queue_lengths.append(queue_length)  # Save the length of the queue
        print(f"Queued Patients at {env.now:.2f}: {queue_length}") # Output of the queue length at this moment
        current_utilization = hospital.operating_theater_utilization / (env.now + 1) * 100 # calculates the utilization of the operating theater in percent
        hospital.max_utilization = max(hospital.max_utilization, current_utilization) # updates the highest value of the utilization (if needed)
        print(f"Operating Theater Utilization: {current_utilization:.2f}%") # Output of the current operating theater utilization
        yield env.timeout(10)  # Monitor every 10 time units

# Set random seed for reproducibility
random.seed(42)

# Create environment and start processes
env = simpy.Environment()
hospital = Hospital(env)
queue_lengths = [] # List to hold queue lengths
env.process(patient_generator(env, hospital))
env.process(monitor(env, hospital))

# Run simulation
env.run(until=SIM_TIME)

print("Simulation ended.")
print()
print()

# Output the highest utilization observed
print(f"Max Operating Theater Utilization: {hospital.max_utilization:.2f}%")

# Calculate and output the average utilization of the operating theater
average_utilization = (hospital.operating_theater_utilization / (SIM_TIME * SURGERY_UNITS)) * 100
print(f"Average Operating Theater Utilization: {average_utilization:.2f}%")

# Calculate and output the average length of the preparation queue
if queue_lengths:
    average_queue_length = sum(queue_lengths) / len(queue_lengths)
    print(f"Average Length of the Preparation Queue: {average_queue_length:.2f}")
else:
    print("No queue lengths recorded.")