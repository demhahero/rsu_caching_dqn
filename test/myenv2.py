import math
import numpy as np
import random

from numpy.random.mtrand import normal
from pydtmc import MarkovChain

# so we dont change values
random.seed(0)
np.random.seed(0)


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
    number_of_contents = 50;

    # Period size (number of time slots)
    T = 500;

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

    # Number of Actions [0 or 1]
    no_actions = 2;

    # RSU cache size
    RSU_cache_size = 150

    # counter of car
    i = 0

    # Total energy used to upload to RSU
    total_energy = 0

    # Total amount of download
    total_download = 0

    # Total amount of download
    total_request_amount = 0

    # Markov step length (changes every ...)
    markov_step_length = int(T / 5)

    # The coverage range of RSU
    RSU_coverage_range = 100

    # Get each car location with time slots
    Matrix = []

    # Temproraily store content that will be stored when car leaves the converage
    available_contents_to_cache = []

    # Number of vehicles generate to the simulation
    number_of_vehicles = 0

    requests_for_contents = [0] * number_of_contents


    #arrival of vehicles
    arrival_times = []


    # initial function
    def __init__(self, density=0):
        self.total_reward = 0
        self.RSU_cache = []
        self.t = 0
        self.total_energy = 0
        self.total_download = 0
        self.total_request_amount = 1
        if (density != 0):
            self.density = density

        self.Lamda = self.density * self.Mean_v;

        # Arrival of vehicls
        arr = np.random.poisson(self.Lamda, size=int(self.T));

        # Total Number of vehicles
        self.number_of_vehicles = sum(arr);

        # Veicle speed
        speed = [0] * self.number_of_vehicles

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

        self.Matrix = [[0 for x in range(self.T)] for y in range(self.number_of_vehicles)]

        # Generate speed for each vehicle
        for i in range(self.number_of_vehicles):
            speed[i] = self.get_truncated_normal();
            passed = False;
            for n in range(self.T):
                if (self.arrival_times[i] > n or passed):
                    self.Matrix[i][n] = math.inf;
                elif (self.Matrix[i][n - 1] != math.inf and self.Matrix[i][n - 1] + speed[
                    i] >= self.RSU_coverage_range):
                    passed = True;
                    self.Matrix[i][n] = -1;  # car left
                elif (self.arrival_times[i] == n):
                    self.Matrix[i][n] = 0
                else:
                    self.Matrix[i][n] = self.Matrix[i][n - 1] + speed[i];

        self.requests = self.generate_mc(['1.2', '2'], int(self.number_of_vehicles),
                                         self.number_of_vehicles + int(self.number_of_vehicles / 5))
        self.available = self.generate_mc(['3', '1.1'], int(self.number_of_vehicles / 5),
                                          self.number_of_vehicles + int(self.number_of_vehicles / 5))
        self.contents_sizes = [random.randrange(40, 60, 1) for i in range(self.number_of_contents)]

    # Function to generate zipf with certain max value and size
    def generate_zipf(self, a, T):
        values = np.random.zipf(a, 100 * T)
        index = [];
        for i in range(len(values)):
            values[i] = values[i] - 1
            if (values[i] > self.number_of_contents - 1):
                index.append(i)
        values = np.delete(values, index)
        return values[:T];

    # Function to generate 2 State Markov Chain and call generate_zipf
    def generate_mc(self, states, window, T):
        # set transition
        p = np.array([[0.2, 0.8], [0.6, 0.4]])
        mc = MarkovChain(p, states)

        # set of transions
        transitions = mc.walk(int(T / window))
        data = [];
        for i in range(len(transitions)):
            for c in self.generate_zipf(float(transitions[i]), window):
                data.append(c)
        return data

    # function to calculate how much is free in the RSU cache
    def RSU_cache_free(self):
        total = 0
        for i in self.RSU_cache:
            total += self.contents_sizes[i]
        return self.RSU_cache_size - total

    # bring least requsted item index until t
    def least_requested_item_index(self, t):
        items = [0] * len(self.RSU_cache)
        for i in range(t):
            if (self.requests[i] in self.RSU_cache):
                items[self.RSU_cache.index(self.requests[i])] += 1
        return items.index(min(items))

    # remove item(s) to make enough space in the RSU cache
    def remove_to_replace(self, size, t):
        while True:
            if (size > self.RSU_cache_free()):
                self.removeFromCache(self.least_requested_item_index(t))
            else:
                break;
        return True

    def addToCache(self, item_index):
        # make sure the RSU cache is not overloaded
        if (self.contents_sizes[item_index] <= self.RSU_cache_free()):
            self.RSU_cache.append(item_index)

    def removeFromCache(self, index):
        # It makes two copies of lists; one from the start until the index but without it (a[:index])
        # and one after the index till the last element (a[index+1:])
        self.RSU_cache = self.RSU_cache[:index] + self.RSU_cache[index + 1:]

    def step(self, action):
        reward = 0
        energy = 0
        download = 0
        total_request = 0

        Done = False

        # next observation
        observation_ = [0] * (self.number_of_contents + 2);

        total_request = self.contents_sizes[self.requests[self.i]]

        if (self.requests[self.i] in self.RSU_cache):
            #print("request found");
            reward += self.contents_sizes[self.requests[self.i]]
            download += self.contents_sizes[self.requests[self.i]]


        for i_b in self.available_contents_to_cache:
            try:
                if(self.Matrix[i_b][:].index(-1) < self.Matrix[self.i][:].index(0)):
                    #print("added"+str(i_b))
                    if (self.RSU_cache_free() < self.contents_sizes[
                        self.available[i_b]]):  # check if there is no enough space in the RSU cache
                        # there is no sufficient space, so we have to remove item(s) from the RSU cache
                        self.remove_to_replace(self.contents_sizes[self.available[i_b]],
                                               i_b)
                    self.addToCache(self.available[i_b])
                    self.available_contents_to_cache.remove(i_b)
            except:
                print("An exception occurred "+str(i_b))

        if (self.available[self.i] not in self.RSU_cache and action == 1):
            #print(self.i)
            self.available_contents_to_cache.append(self.i);
            reward -= self.hit_energy_ratio * self.contents_sizes[self.available[self.i]]
            energy += self.hit_energy_ratio * self.contents_sizes[self.available[self.i]]

        observation_[self.number_of_contents] = self.available[self.i + 1]

        observation_[self.number_of_contents + 1] = self.requests[self.i]

        self.requests_for_contents[self.requests[self.i]] += 1

        # write whatever exsits in the RSU cache now
        for c in range(self.number_of_contents):

            observation_[c] += self.requests_for_contents[c] / sum(self.requests_for_contents)
            if (c in self.RSU_cache):
                observation_[c] += 1

        for c in self.available_contents_to_cache:
            observation_[self.available[c]] += 1
        #print(observation_)

        # update total reward, energy and download
        self.total_reward += reward
        self.total_download += download
        self.total_energy += energy
        self.total_request_amount += total_request

        self.i += 1

        if (self.i == self.number_of_vehicles):
            Done = True

        return reward, observation_, Done


    def reset(self):
        self.total_reward = 0
        self.RSU_cache = []
        self.i = 0
        self.t = 0
        self.total_energy = 0
        self.total_download = 0
        self.total_request_amount = 1
        self.available_contents_to_cache = []
        self.requests_for_contents = [0] * self.number_of_contents
        return [0] * (self.number_of_contents + 2);
