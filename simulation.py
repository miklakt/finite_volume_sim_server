#%% calculation.py
import time
import threading
from multiprocessing import shared_memory
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from shared_memory_ipc import *

status = "non-initialized"
nx, ny = 128, 128

simulation_dtype = np.dtype([
    ('nx', np.int32, ()),
    ('ny', np.int32, ()),
    ('timestep', np.int32, ()),
    ('dt', np.float32, ()),
    ('c', np.float32, (ny, nx)),
    #('u', np.float32, (ny, nx)),
    #('flux_x', np.float32, (ny, nx)),
    #('flux_y', np.float32, (ny, nx)),
    ('command', "S16", ()),
])


def init_concentration(data):
    data["nx"], data["ny"] = nx, ny
    # concentration = np.zeros((height, width), dtype=np.float32)
    # concentration[:, 0] = 1.0  # Left boundary with constant concentration
    # square_size = nx // 4  # Size of the square (adjust as needed)
    square_size = 50
    start_x = nx // 2 - square_size // 2
    end_x = start_x + square_size
    start_y = ny // 2 - square_size // 2
    end_y = start_y + square_size
    data["c"][start_y:end_y, start_x:end_x] = 1.0
    print("Field initialized, BC applied")
#%%

KERNEL_CODE = """
__global__ void diffusion(float *C, int nx, int ny, float dt)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * nx + ix;

    if(ix > 0 && ix < nx-1 && iy > 0 && iy < ny-1)
    {
        float C0 = C[idx];
        float C_left = C[idx - 1];
        float C_right = C[idx + 1];
        float C_up = C[idx - nx];
        float C_down = C[idx + nx];


        // Central differencing scheme with potential field
        float diff_x = (C_right - 2*C0 + C_left);
        float diff_y = (C_up - 2*C0 + C_down);


        C[idx] = C0 + dt * (diff_x + diff_y);
    }
    else
    {
        // Left boundary condition
        if(ix == 0)
            C[idx] = 1.0;

        // Upper and lower boundaries (mirror)
        if(iy == 0 && ix > 0 && ix < nx-1)
            C[idx] = C[idx + nx];
        if(iy == ny -1 && ix > 0 && ix < nx-1)
            C[idx] = C[idx - nx];
    }
}
"""
#%%
def run_simulation(data, update_shm_every = 1000, command_timeout = 10, limit_runs = 100):
    global status
    write_command, read_command, pop_command = create_methods_for_shm_command(data)
    #Create a new context for this thread
    diffusion_kernel = SourceModule(KERNEL_CODE)
    diffuse = diffusion_kernel.get_function("diffusion")

    concentration = data["c"]
    nx, ny = data["nx"], data["ny"]
    timestep = data["timestep"]
    dt = data["dt"]

    block_size = (16, 16, 1)
    grid_size = (int((nx + block_size[0] - 1) // block_size[0]),
                int((ny + block_size[1] - 1) // block_size[1]))

    try:
        #Copy to GPU
        concentration_gpu = cuda.mem_alloc(concentration.nbytes)
        cuda.memcpy_htod(concentration_gpu, concentration)

        nrun = 0
        while True:
            command = pop_command()

            #if no 'start' command received
            if not(command) and not(status == "running"):
                print(f"Status:{status}")
                print("Waiting for command...")
                start_time = time.time()
                while status != "running":
                    elapsed_time = time.time() - start_time
                    if elapsed_time > command_timeout:
                        print("Timeout reached. Simulation will be stopped")
                        status = "timeout"
                        return
                    command = pop_command()
                    if command =="start": status = "running"
                    time.sleep(1)

            if command == "stop":
                data["timestep"] = timestep
                data["c"] = concentration
                status = "stopped"
                return
            elif command == "pause":
                status = "paused"
                time.sleep(1)
                continue
            elif command == "start":
                status = "running"
            
            
            # Run the diffusion kernel
            for t in range(update_shm_every):
                #__global__ void diffusion(float *C, int nx, int ny, float dt)
                diffuse(concentration_gpu,
                        nx, ny, dt,
                        block=block_size, grid=grid_size)
                timestep += 1

            # Copy result back CPU
            cuda.memcpy_dtoh(concentration, concentration_gpu)

            # Update shared memory
            data["timestep"] = timestep
            data["c"] = concentration.copy()
            nrun +=1
            print(f"\rExecuted {nrun} times. Time step {timestep} data written to shared memory.", end='', flush=True)
            if limit_runs is not None:
                if nrun>=limit_runs:
                    print("Hit execution limit")
                    write_command("stop")

    except Exception as e:
        print(f"Thread encountered an error: {e}")
    finally:
        print("Simulation stopped")

if __name__ == "__main__":
    simulation_data_shared , shm = create_shm_numpy_structure(simulation_dtype, "simulation_data")
    simulation_data_shared["dt"] = 0.1
    init_concentration(simulation_data_shared)
    run_simulation(simulation_data_shared, 100)
    shm.unlink()
    shm.close()
    print("Shared memory is deallocated")