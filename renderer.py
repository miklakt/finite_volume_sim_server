# renderer.py
import numpy as np
import matplotlib.pyplot as plt
from shared_memory_ipc import *

# Define grid size and parameters
nx, ny = 128, 128

# Define the structured data type (must match simulation.py)
simulation_dtype = np.dtype([
    ('timestep', np.int32),
    ('C', np.float32, (ny, nx)),
    ('Phi', np.float32, (ny, nx)),
    #('flux_x', np.float32, (ny, nx)),
    #('flux_y', np.float32, (ny, nx)),
])

# Attempt to attach to existing shared memory
try:
    sim_shm = shared_memory.SharedMemory(name='simulation_data')
    sim_shared = np.ndarray((1,), dtype=simulation_dtype, buffer=sim_shm.buf)
    print("Connected to shared memory.")
except FileNotFoundError:
    print("Shared memory not found. Ensure that the simulation is running.")
    exit(1)

# Set up Matplotlib figure
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((ny, nx)), cmap='hot', interpolation='nearest', origin='lower', extent=[0, nx, 0, ny])
ax.set_title('Concentration')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Concentration')

print("Starting real-time plotting.")

# Flag to control the plotting loop
plotting = True

# Define a handler for the close event
def on_close(event):
    print("Matplotlib window closed.")
    plotting = False

# Connect the handler to the figure
fig.canvas.mpl_connect('close_event', on_close)

try:
    while plotting:
        # Read data from shared memory
        timestep = sim_shared['timestep'][0]
        C = sim_shared['C'][0].copy()
        Phi = sim_shared['Phi'][0].copy()
        #flux_x = sim_shared['flux_x'][0].copy()
        #flux_y = sim_shared['flux_y'][0].copy()

        # Update the plot (e.g., plotting concentration)
        im.set_data(C)
        ax.set_title(f"Concentration at timestep {timestep}")

        # Update color scale limits
        im.set_clim(np.min(C), np.max(C))
        cbar.update_normal(im)

        plt.draw()
        plt.pause(0.1)

        time.sleep(0.1)

        if not plt.fignum_exists(fig.number):
            print("Figure closed.")
            break

except KeyboardInterrupt:
    print("Plotting stopped by user.")
finally:
    # Clean up
    sim_shm.close()
    print("Shared memory closed.")
