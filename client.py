# client.py
#%%
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import shared_memory

import matplotlib
#matplotlib.use('Qt5Agg')  # Ensure Qt5 backend is used

from simulation import simulation_dtype
from shared_memory_ipc import *

simulation_data_shared = connect_to_shm_numpy_structure(simulation_dtype, 'simulation_data')

write_command, read_command, pop_command = create_methods_for_shm_command(simulation_data_shared)

def plot_real_time_concentration(fig, ax, data = sim_shared):
    #cbar = plt.colorbar(im, ax=ax)
    plt.title("Real-Time Concentration")

    im.set_data(data["c"])
    ax.set_title(f"Real-Time Concentration (Timestep: {data["timestep"]})")

    # Update color scale limits
    im.set_clim(np.min(data["c"]), np.max(data["c"]))
    #cbar.update_normal(im)

    plt.draw()
#%%

#%%
fig, ax = plt.subplots()
im = ax.imshow(sim_shared["c"], cmap="viridis", vmin=0, vmax=1)
#%%
plot_real_time_concentration(fig, ax)

# %%
