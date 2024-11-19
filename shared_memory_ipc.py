from multiprocessing import shared_memory
import numpy as np
from numpy.typing import DTypeLike, NDArray

COMMAND_LENGTH = 16

def print_numpy_dtype(numpy_dtype: DTypeLike) -> None:
    print("Data structure")
    print("-"*45)
    print("{:<15}{:<20}{:>10}".format(*"field    dtype   size".split()))
    for name in numpy_dtype.names:
        print("{:<15}{:<20}{:>10}".format(str(name), str(numpy_dtype[name]), str(numpy_dtype[name].itemsize)))
    print("-"*45)

def allocate_shm_for_stuctured_data(numpy_dtype: DTypeLike, name: str, shape = ()):
    if shape == tuple():
        size = numpy_dtype.itemsize
    else:
        size = int(np.prod(shape))*numpy_dtype.itemsize
    try:
        shm = shared_memory.SharedMemory(name, create = True, size = size)
        print(f"Created shared memory block: '{name}' with size {size} bytes.")
    except FileExistsError:
        print(f"Shared memory block '{name}' already exists. Overwriting it.")
        existing_shm = shared_memory.SharedMemory(name=name)
        existing_shm.unlink()  # Deletes the existing shared memory block
        # Create a new shared memory block
        shm = shared_memory.SharedMemory(name=name, create=True, size = size)
        print(f"Recreated shared memory block: '{name}' with size {size} bytes.")
    print_numpy_dtype(numpy_dtype)
    return shm

def create_shm_numpy_structure(numpy_dtype: DTypeLike, name: str, shape = ()):
    shm = allocate_shm_for_stuctured_data(numpy_dtype, name)
    shm_ndarray = np.ndarray(shape, dtype=numpy_dtype, buffer=shm.buf)
    return shm_ndarray, shm

def connect_to_shm_numpy_structure(numpy_dtype: DTypeLike, name: str, shape = ()):
    try:
        shm = shared_memory.SharedMemory(name=name)
        sim_ndarray = np.ndarray(shape, dtype=numpy_dtype, buffer=shm.buf)
        print("Connected to simulation data shared memory.")
    except FileNotFoundError:
        print("Shared memory not found. Ensure that the simulation is running.")
        exit(1)
    return sim_ndarray


def accessor_shm_command_field(shm, command_str_length = COMMAND_LENGTH):
    dtype = f"S{command_str_length}"
    command = np.ndarray((), dtype, buffer = shm.buf[-15:])
    return command

def create_methods_for_shm_command(shm_ndarray : NDArray, command_str_length = COMMAND_LENGTH):
    def write_command(command:str):
        if command is None: return write_command("")
        #shm_cmd.buf[:COMMAND_LENGTH] = command.encode('utf-8').ljust(COMMAND_LENGTH)
        shm_ndarray["command"] = command
        if (command):
            print(f"Command '{command}' written to shared memory")
        return

    def read_command():
        # retrieved_string = bytes(shm_cmd.buf[:]).decode('utf-8').strip()
        retrieved_string = str(shm_ndarray["command"].astype(str))
        print(f"Command '{retrieved_string}' read from shared memory")
        return retrieved_string

    def pop_command():
        #retrieved_string = bytes(shm_cmd.buf[:]).decode('utf-8').strip()
        retrieved_string = str(shm_ndarray["command"].astype(str))
        write_command("")
        if retrieved_string: print(f"Command '{retrieved_string}' pop from shared memory")
        return retrieved_string
    
    return write_command, read_command, pop_command

# def allocate_shared_memory_for_stuctured_data(numpy_dtype: DTypeLike, name: str):
#     try:
#         shm = shared_memory.SharedMemory(name, create = True, size = simulation_dtype.itemsize)
#         print(f"Created shared memory block: '{name}' with size {size} bytes.")
#     except FileExistsError:
#         print(f"Shared memory block '{name}' already exists. Overwriting it.")
#         existing_shm = shared_memory.SharedMemory(name=name)
#         existing_shm.unlink()  # Deletes the existing shared memory block
#         # Create a new shared memory block
#         shm = shared_memory.SharedMemory(name=name, create=True, size=size)
#         _shm_data = shared_memory.SharedMemory("simulation_data", size = simulation_dtype.itemsize)
#         print(f"Recreated shared memory block: '{name}' with size {size} bytes.")
#     print_numpy_dtype(numpy_dtype)
#     return shm