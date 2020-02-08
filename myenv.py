import math
import numpy as np
import random

from numpy.random.mtrand import normal
from pydtmc import MarkovChain


# so we dont change values
# random.seed(0)
# np.random.seed(0)


class MyEnv:
    # Parameters for free flow traffic
    STD_SPEED = 10;
    MIN_V_SPEED, MAX_V_SPEED = 80 * 1000 / (60 * 60), 120 * 1000 / (60 * 60)
    density_jam = 0.25;  # per meter
    velocity_free = 38.8889;  # mps
    Mean_v = 100 * 1000 / (60 * 60);  # m/s
    density = 0;  # number of vehicles per meter (Density)

    # Get rondom speed
    def get_truncated_normal(self):
        speed = self.STD_SPEED * normal(0.0, 1.0) + self.get_expected_velocity()
        while speed < 0 or speed > self.MAX_V_SPEED or speed < self.MIN_V_SPEED:
            speed = self.STD_SPEED * normal(0.0, 1.0) + self.get_expected_velocity()
        return speed

    # Calculate the expected velocity
    def get_expected_velocity(self):
        return self.velocity_free * (1 - (self.density) / self.density_jam)

    # number of contents in the library
    number_of_contents = 1000;

    # Period size (number of time slots)
    T = 0;

    delta_t = 1

    # Requests every time slots
    requests = []

    # Avialable contents on cars caches every time slots
    available = []

    # Generate contents sizes
    contents_sizes = []

    # RSU cache list (at the beginning it is empty)
    RSU_cache = [];

    # The cost ratio between hit and energy for taking content from a car.
    hit_energy_ratio = 0.5;

    # Total reward from one episode
    total_reward = 0



    # counter of car
    i = 0

    # Total energy used to upload to RSU
    total_energy = 0

    # Total amount of download
    total_download = 0

    # Total amount of download
    total_request_amount = 0

    # Markov step length (changes every ...)
    markov_step_length = 0

    # The coverage range of RSU
    RSU_coverage_range = 1000

    # Get each car location with time slots
    Matrix = []

    # Temproraily store content that will be stored when car leaves the converage
    available_contents_to_cache = []

    # Number of vehicles generate to the simulation
    number_of_vehicles = 0

    requests_for_contents = [0] * number_of_contents

    # arrival of vehicles
    arrival_times = []

    min_size, max_size = 200, 401

    # RSU cache size
    RSU_cache_size = 1000
    max_cached_contents = 2

    RSU_cache_upload = 0

    misses = [0] * number_of_contents

    avialable_shift = 0

    downrate = 100
    uprate = 100

    within_range = []
    served = []
    fetched = []
    vehicle_to_cache = []

    vehicle_no_contents = 1

    # Number of Actions [0 or 1]
    no_actions = 3 * vehicle_no_contents

    speed = []
    episode = 0
    no_episodes = 0
    observation_length = (no_actions - 1) * 4 + max_cached_contents * 2  # 5 cars = 10 contents = 30 features + 10

    # initial function
    def __init__(self, density=0, T=500, number_of_contents=100):
        self.total_reward = 0
        self.RSU_cache = []
        self.t = 0
        self.total_energy = 0
        self.total_download = 0
        self.total_request_amount = 1
        self.number_of_contents = number_of_contents
        self.T = T

        if (density != 0):
            self.density = density

        self.Lamda = self.density * self.Mean_v;

        # Arrival of vehicls
        arr = np.random.poisson(self.Lamda, size=int(self.T));

        # Total Number of vehicles
        self.number_of_vehicles = sum(arr);

        self.markov_step_length = int((self.number_of_vehicles) / 1)

        # Veicle speed
        self.speed = [0] * self.number_of_vehicles

        # Arrival time of each vehicle
        self.arrival_times = [0] * self.number_of_vehicles;

        # Get each vehicle arrival time
        counter1 = 0;
        counter3 = 0;
        counter4 = 0;

        for val in arr:
            counter2 = counter1 + val
            if (counter1 < counter2):
                for i in range(counter2 - counter1):
                    self.arrival_times[counter3] = counter4
                    counter3 = counter3 + 1

            counter4 = counter4 + 1
            counter1 = counter2

        self.served = [0] * self.number_of_vehicles

        self.fetched = [[0 for x in range(self.vehicle_no_contents)] for y in range(self.number_of_vehicles)]

        self.Matrix = [[0 for x in range(self.T)] for y in range(self.number_of_vehicles)]

        self.available = [[0 for x in range(self.vehicle_no_contents)] for y in range(self.number_of_vehicles)]

        # Generate speed for each vehicle
        for i in range(self.number_of_vehicles):
            self.speed[i] = self.get_truncated_normal();
            passed = False;
            for n in range(self.T):
                if (self.arrival_times[i] > n or passed):
                    self.Matrix[i][n] = math.inf;
                elif (self.Matrix[i][n - 1] != math.inf and self.Matrix[i][n - 1] + self.speed[
                    i] >= self.RSU_coverage_range):
                    passed = True;
                    self.Matrix[i][n] = -1;  # car left
                elif (self.arrival_times[i] == n):
                    self.Matrix[i][n] = 0
                else:
                    self.Matrix[i][n] = self.Matrix[i][n - 1] + self.speed[i]

        self.requests = self.generate_mc(['1.8', '1.4'], self.markov_step_length,
                                         2 * self.number_of_vehicles * self.vehicle_no_contents)

        available = self.generate_mc(['1.8', '1.4'], self.markov_step_length,
                                     2 * self.number_of_vehicles * self.vehicle_no_contents, available=True)

        vehicle_counter = 0
        content_counter = 0
        for i in available:
            self.available[vehicle_counter][content_counter] = i
            content_counter += 1
            if (content_counter == self.vehicle_no_contents):
                content_counter = 0
                vehicle_counter += 1
            if vehicle_counter == self.number_of_vehicles:
                break

        random.seed(0)
        self.contents_sizes = [random.randrange(self.min_size, self.max_size, 1) for i in
                               range(self.number_of_contents)]

    # Function to generate zipf with certain max value and size
    def generate_zipf(self, a, T, transition, available=False):
        values = np.random.zipf(a, T)
        index = [];
        for i in range(len(values)):
            #values[i] = values[i]
            if (a == "1.4"):
                values[i] = values[i] + 3
            values[i] = values[i] % self.number_of_contents

            #if (available):
            #    values[i] = values[i] + self.avialable_shift
            #    values[i] = values[i] % self.number_of_contents

            # if request = available, shift the avialable by one
            #if (available and self.requests[transition * T + i] == values[i]):
                #values[i] = values[i] + 1
                #values[i] = values[i] % self.number_of_contents
            # if (values[i] > self.number_of_contents - 1):
            # index.append(i)
        # values = np.delete(values, index)
        return values[:T];

    # Function to generate 2 State Markov Chain and call generate_zipf
    def generate_mc(self, states, window, T, available=False):
        # set transition
        p = np.array([[0.4, 0.6], [0.75, 0.25]])
        mc = MarkovChain(p, states)

        # set of transions
        transitions = mc.walk(int(T / window))
        data = [];
        for transition in range(len(transitions)):
            for c in self.generate_zipf(float(transitions[transition]), window, transition=transition,
                                        available=available):
                data.append(c)
        return data

    # function to calculate how much is free in the RSU cache
    def RSU_cache_free(self):
        total = 0
        for c in self.RSU_cache:
            total += self.contents_sizes[c]
        return self.RSU_cache_size - total

    # bring least requsted item index until t
    def least_requested_item_index(self):
        items = [0] * len(self.RSU_cache)
        counter = 0
        for c in self.RSU_cache:
            items[counter] = self.requests_for_contents[c]
            counter += 1
        return self.RSU_cache[items.index(min(items))]

    # remove item(s) to make enough space in the RSU cache
    def remove_to_replace(self, size):
        while size > self.RSU_cache_free():
            self.RSU_cache.remove(self.least_requested_item_index())

    def addToCache(self, c):
        self.remove_to_replace(self.contents_sizes[c])
        # make sure the RSU cache is not overloaded
        if (self.contents_sizes[c] <= self.RSU_cache_free()):
            self.RSU_cache.append(c)
        else:
            print("NO enough capacity " + str(self.contents_sizes[c]) + " " + str(self.RSU_cache_free()))

    def step(self, action):
        Done = False

        content_c = 0
        vehicle_i = -1
        if (action != self.no_actions - 1):
            content_c = action % self.vehicle_no_contents
            vehicle_i = int((action - content_c) / self.vehicle_no_contents)
        #if(self.episode == self.no_episodes - 1):
            #print("content, vehilce", content_c, vehicle_i)
        reward, download, incentive = self.downlink_resources_allocation(vehicle_i, content_c)

        # update total reward, energy and download
        self.total_reward += reward
        self.total_download += download
        self.total_energy += incentive

        self.t += 1

        if (self.t == self.T - 1):
            Done = True
            observation_ = [0] * (self.observation_length)
        else:
            for i in range(self.number_of_vehicles):
                #print("xxxxx", i, self.t)
                if (self.Matrix[i][self.t] == 0):
                    self.within_range.append(i)
                    self.requests_for_contents[self.requests[self.i]] += 1
                    self.total_request_amount += self.contents_sizes[self.requests[self.i]]
                    self.i += 1
                elif (self.Matrix[i][self.t] == -1 and i in self.within_range):
                    self.within_range.remove(i)

            observation_ = self.build_observation()

        return reward, observation_, Done

    def build_observation(self):
        observation_ = [0] * (self.observation_length);

        # Building observation vector
        # observation_[0] = (1 - observation_[0])
        obs_counter = 0
        for i in range(min([self.no_actions - 1, len(self.within_range)])):
            # if(-1 in self.Matrix[i][:]):
            # observation_[obs_counter] = self.Matrix[i][:].index(-1) - self.t
            # obs_counter += 1
            for c in range(self.vehicle_no_contents):
                observation_[obs_counter] = int(
                    (self.requests_for_contents[self.available[self.within_range[i]][c]] * 100) / sum(
                        self.requests_for_contents))
                observation_[obs_counter + 1] = self.fetched[self.within_range[i]][c]
                observation_[obs_counter + 2] = self.contents_sizes[self.available[self.within_range[i]][c]]
                if (-1 in self.Matrix[self.within_range[i]][:]):
                    observation_[obs_counter + 3] = self.Matrix[self.within_range[i]][:].index(-1) - self.t
                else:
                    observation_[obs_counter + 3] = 0

                obs_counter += 4

        obs_counter = self.observation_length - self.max_cached_contents * 2
        for c in range(min(self.max_cached_contents, len(self.RSU_cache))):
            observation_[obs_counter] = int((self.requests_for_contents[self.RSU_cache[c]] * 100) / sum(
                self.requests_for_contents))
            observation_[obs_counter + 1] = self.contents_sizes[self.RSU_cache[c]]
            obs_counter += 2
        return observation_

    def is_previously_scheduled(self, content):
        for i in self.vehicle_to_cache:
            if (self.available[i] == content):
                return True
        return False

    def sufficient_resources(self):
        # slots=0
        # available_slots = -1
        # for i in self.vehicle_to_cache:
        #     slots += int((self.available[i] - self.fetched[i])/self.uprate)
        #
        # if(-1 in self.Matrix[self.i]):
        #     available_slots = self.Matrix[self.i][:].index(-1) - self.Matrix[self.i][:].index(0)
        #
        # if(slots - available_slots >=0):
        #     return True
        # else:
        return False

    to_complete_serve = -1

    def downlink_resources_allocation(self, vehicle, content):
        reward = 0
        download = 0
        incentive = 0

        # downlink
        if (self.to_complete_serve == -1 or self.to_complete_serve not in self.within_range
                or self.requests[self.to_complete_serve] not in self.RSU_cache):
            self.to_complete_serve = -1
            for i in self.within_range:
                if (self.requests[i] in self.RSU_cache and self.served[i] < self.contents_sizes[self.requests[i]]):
                    self.to_complete_serve = i
                    break

        if (self.to_complete_serve != -1):
            self.served[self.to_complete_serve] += self.downrate

            if (self.served[self.to_complete_serve] >= self.contents_sizes[self.requests[self.to_complete_serve]]):
                reward += self.contents_sizes[self.requests[self.to_complete_serve]]
                download += self.contents_sizes[self.requests[self.to_complete_serve]]
                self.served[self.to_complete_serve] = self.contents_sizes[self.requests[self.to_complete_serve]]
                self.to_complete_serve = -1

        # uplink
        if (vehicle != -1):
            #print("------vehicle", len(self.within_range))
            if (vehicle < len(self.within_range)):

                vehicle_id = self.within_range[vehicle]
                if (self.available[vehicle_id][content] in self.RSU_cache):
                    #reward -= 1000
                    h = 1
                    # self.total_reward += 10
                else:
                    if (self.fetched[vehicle_id][content] + self.uprate >= self.contents_sizes[self.available[vehicle_id][content]]):
                        reward -= self.hit_energy_ratio * (self.contents_sizes[self.available[vehicle_id][content]]
                                                           - self.fetched[vehicle_id][content])

                        incentive += self.hit_energy_ratio * (self.contents_sizes[self.available[vehicle_id][content]]
                                - self.fetched[vehicle_id][content])

                        self.fetched[vehicle_id][content] = self.contents_sizes[self.available[vehicle_id][content]]
                        self.addToCache(self.available[vehicle_id][content])
                    else:
                        reward -= self.hit_energy_ratio * self.uprate
                        incentive += self.hit_energy_ratio * self.uprate
                        self.fetched[vehicle_id][content] += self.uprate

            else:
                h = 1
                #reward -= 1000
                # self.total_reward += 10

        return reward, download, incentive

    def reset(self, dont=False):
        self.total_reward = 0
        self.RSU_cache = []
        self.i = 0
        self.t = 0
        self.total_energy = 0
        self.total_download = 0
        self.RSU_cache_upload = 0
        self.total_request_amount = 1

        self.within_range = []
        self.vehicle_to_cache = []

        self.served = [0] * self.number_of_vehicles

        self.fetched = [[0 for x in range(self.vehicle_no_contents)] for y in range(self.number_of_vehicles)]
        self.to_complete_serve = -1
        if (dont == False):
            self.requests_for_contents = [0] * self.number_of_contents
        return [0] * (self.observation_length);
