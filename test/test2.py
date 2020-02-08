import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotting
#
# x = np.random.choice([0, 1], size=(10000,), p=[1./3, 2./3])


# with open("pickle_value", "wb") as f:
#     pickle.dump(x, f)

with open("pickle_value", "rb") as f:
    stats = pickle.load(f)

plotting.plot_episode_stats(stats, smoothing_window=200)