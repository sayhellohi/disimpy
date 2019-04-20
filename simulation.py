import math
import numba
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64, xoroshiro128p_normal_float64


def calculate_signal(phases):
    """Return total diffusion-attenuated MRI signal based on individual spins' phase shifts"""
    signals = np.real(np.sum(np.exp(1j*phases), axis = 1))
    return signals
                              

def diffusion_simulation(substrate, n_of_spins, seed, dt, gradient_array, trajectories = False):
    """Run diffusion MRI simulation"""    
    
    # Define simulation parameters
    gamma = 267.513e6
    diffusivity = substrate.diffusivity
    step_length = np.sqrt(6 * diffusivity * dt)
    n_of_spins = int(n_of_spins)
    n_of_steps = gradient_array.shape[1]
    n_of_measurements = gradient_array.shape[2]

    # Define blocksize (threads/block) and gridsize (blocks/grid) and initiate stream
    block_size = 256 # Using 512 here results in CudaAPIError: Call to cuLaunchKernel results in CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES which is probably due to CUDA registers limitation (registers per thread * threads per block)
    grid_size = int(math.ceil(float(n_of_spins) / block_size))
    stream = cuda.stream()
        
    # Instantiate pseudo random number generator
    rng_states = create_xoroshiro128p_states(grid_size * block_size, seed=seed, stream=stream)
    
    # Allocate device memory and move necessary arrays to device
    d_phases = cuda.to_device(np.zeros((n_of_measurements, n_of_spins)), stream=stream)
    d_g_x = cuda.to_device(gradient_array[0,:,:], stream=stream)
    d_g_y = cuda.to_device(gradient_array[1,:,:], stream=stream)
    d_g_z = cuda.to_device(gradient_array[2,:,:], stream=stream)

    # Keep track of walker trajectories while running simulation
    if trajectories:

        trajectories = np.zeros((3, n_of_spins, n_of_steps))

        if substrate.type == 'ball':      
            radius = substrate.radius
            initial_positions = substrate.fill_uniformly(n_of_spins, seed)    
            trajectories[:,:,0] = initial_positions
            d_positions = cuda.to_device(initial_positions, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_ball[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                              d_phases, rng_states, time_point, \
                                                              n_of_spins, gamma, step_length, dt, \
                                                              radius)
                stream.synchronize()
                trajectories[:,:,time_point] = d_positions.copy_to_host(stream=stream)
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")

        if substrate.type == 'cylinder':      
            radius = substrate.radius      
            orientation = substrate.orientation
            initial_positions = substrate.fill_uniformly(n_of_spins, seed)    
            trajectories[:,:,0] = initial_positions
            d_positions = cuda.to_device(initial_positions, stream=stream)
            d_orientation = cuda.to_device(orientation, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_cylinder[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                                  d_phases, rng_states, time_point, \
                                                                  n_of_spins, gamma, step_length, dt, \
                                                                  radius, d_orientation)
                stream.synchronize()
                trajectories[:,:,time_point] = d_positions.copy_to_host(stream=stream)
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")

        elif substrate.type == 'free':        
            initial_positions = np.zeros((3, n_of_spins))
            trajectories[:,:,0] = initial_positions
            d_positions = cuda.to_device(initial_positions, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_free[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                              d_phases, rng_states, time_point, \
                                                              n_of_spins, gamma, step_length, dt)
                stream.synchronize()
                trajectories[:,:,time_point] = d_positions.copy_to_host(stream=stream)
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")

        elif substrate.type == 'mesh':        
            triangles = substrate.triangles
            initial_positions = substrate.fill_uniformly(n_of_spins)        
            trajectories[:,:,0] = initial_positions
            d_positions = cuda.to_device(initial_positions, stream=stream)
            d_triangles = cuda.to_device(triangles.ravel(), stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_mesh[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                               d_phases, rng_states, time_point, \
                                                               n_of_spins, gamma, step_length, dt, \
                                                               d_triangles)
                stream.synchronize()
                trajectories[:,:,time_point] = d_positions.copy_to_host(stream=stream)
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")

        elif substrate.type == 'plane':        
            directions = np.append(substrate.v1, substrate.v2)
            initial_positions = np.zeros((3, n_of_spins))        
            trajectories[:,:,0] = initial_positions
            d_positions = cuda.to_device(initial_positions, stream=stream)
            d_directions = cuda.to_device(directions, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_plane[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                               d_phases, rng_states, time_point, \
                                                               n_of_spins, gamma, step_length, dt, \
                                                               d_directions)
                stream.synchronize()
                trajectories[:,:,time_point] = d_positions.copy_to_host(stream=stream)
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")

        elif substrate.type == 'stick':        
            direction = substrate.orientation / np.linalg.norm(substrate.orientation)
            initial_positions = np.zeros((3, n_of_spins))        
            trajectories[:,:,0] = initial_positions
            d_positions = cuda.to_device(initial_positions, stream=stream)
            d_direction = cuda.to_device(direction, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_stick[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                               d_phases, rng_states, time_point, \
                                                               n_of_spins, gamma, step_length, dt, \
                                                               d_direction)
                stream.synchronize()
                trajectories[:,:,time_point] = d_positions.copy_to_host(stream=stream)
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")
                
        # Calculate signal and return results
        phases = d_phases.copy_to_host(stream=stream)
        signals = calculate_signal(phases)
        return signals, trajectories

    # Run simulation without saving trajectories
    else:    
    
        if substrate.type == 'ball':      
            radius = substrate.radius
            initial_positions = substrate.fill_uniformly(n_of_spins, seed)    
            d_positions = cuda.to_device(initial_positions, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_ball[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                              d_phases, rng_states, time_point, \
                                                              n_of_spins, gamma, step_length, dt, \
                                                              radius)
                stream.synchronize()
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")

        if substrate.type == 'cylinder':      
            radius = substrate.radius      
            orientation = substrate.orientation
            initial_positions = substrate.fill_uniformly(n_of_spins, seed)    
            d_positions = cuda.to_device(initial_positions, stream=stream)
            d_orientation = cuda.to_device(orientation, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_cylinder[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                                  d_phases, rng_states, time_point, \
                                                                  n_of_spins, gamma, step_length, dt, \
                                                                  radius, d_orientation)
                stream.synchronize()
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")
                
        elif substrate.type == 'free':        
            initial_positions = np.zeros((3, n_of_spins))
            d_positions = cuda.to_device(initial_positions, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_free[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                              d_phases, rng_states, time_point, \
                                                              n_of_spins, gamma, step_length, dt)
                stream.synchronize()
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")

        elif substrate.type == 'plane':        
            directions = np.append(substrate.v1, substrate.v2)
            initial_positions = np.zeros((3, n_of_spins))        
            d_positions = cuda.to_device(initial_positions, stream=stream)
            d_directions = cuda.to_device(directions, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_plane[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                               d_phases, rng_states, time_point, \
                                                               n_of_spins, gamma, step_length, dt, \
                                                               d_directions)
                stream.synchronize()
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")
                
        elif substrate.type == 'stick':        
            orientation = substrate.orientation
            initial_positions = np.zeros((3, n_of_spins))        
            d_positions = cuda.to_device(initial_positions, stream=stream)
            d_orientation = cuda.to_device(orientation, stream=stream)
            for time_point in range(1, n_of_steps):
                cuda_step_stick[grid_size, block_size, stream](d_positions, d_g_x, d_g_y, d_g_z, \
                                                               d_phases, rng_states, time_point, \
                                                               n_of_spins, gamma, step_length, dt, \
                                                               d_orientation)
                stream.synchronize()
                print(str(np.round((time_point/n_of_steps)*100, 0)) + ' %', end="\r")
        
        # Calculate signal and return results
        phases = d_phases.copy_to_host(stream=stream)
        signals = calculate_signal(phases)
        return signals


@cuda.jit()
def cuda_step_ball(positions, g_x, g_y, g_z, phases, rng_states, time_point, n_of_spins, gamma, step_length, dt, radius):
    """Kernel function for diffusion inside a sphere"""
    
    # Global thread index on a 1D grid
    thread_id = cuda.grid(1)
    if thread_id >= n_of_spins:
        return

    # Generate random unit step
    step = cuda.local.array(3, numba.double)
    step[0] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[1] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[2] = xoroshiro128p_normal_float64(rng_states, thread_id)
    normalizing_factor = math.sqrt(step[0]**2 + step[1]**2 + step[2]**2)
    step[0] = step[0] / normalizing_factor
    step[1] = step[1] / normalizing_factor
    step[2] = step[2] / normalizing_factor

    # Check for intersection and reflect the step off the surface
    i = 0
    max_iter = 1e4
    check_intersection = True
    intersection = cuda.local.array(3, numba.double)
    normal_vector = cuda.local.array(3, numba.double)
    while check_intersection and i < max_iter:
        i += 1
        t = ball_intersection_check(positions[:, thread_id], step, radius)
        if t <= step_length:
            intersection[0] = positions[0, thread_id] + t*step[0]
            intersection[1] = positions[1, thread_id] + t*step[1]
            intersection[2] = positions[2, thread_id] + t*step[2]
            normal_vector[0] = -intersection[0]
            normal_vector[1] = -intersection[1]
            normal_vector[2] = -intersection[2]
            normalizing_factor = math.sqrt(normal_vector[0]**2 + normal_vector[0]**2 + normal_vector[0]**2)
            normal_vector[0] /= normalizing_factor
            normal_vector[1] /= normalizing_factor
            normal_vector[2] /= normalizing_factor
            reflect_step(positions[:, thread_id], step, intersection, normal_vector, step_length)
        else:
            check_intersection = False
            positions[0, thread_id] = positions[0, thread_id] + step_length*step[0]
            positions[1, thread_id] = positions[1, thread_id] + step_length*step[1]
            positions[2, thread_id] = positions[2, thread_id] + step_length*step[2]
    
    # Calculate phase shift
    for measurement in range(g_x.shape[1]):
        phases[measurement, thread_id] += gamma * dt * \
                                          (g_x[time_point, measurement] * positions[0, thread_id] + \
                                           g_y[time_point, measurement] * positions[1, thread_id] + \
                                           g_z[time_point, measurement] * positions[2, thread_id])


@cuda.jit(device = True)
def ball_intersection_check(r0, step, radius):
    """Return distance from spin to intersection in step direction"""        
    t = -(step[0]*r0[0] + step[1]*r0[1] + step[2]*r0[2]) + math.sqrt((step[0]*r0[0] + step[1]*r0[1] + step[2]*r0[2])**2 \
        - (r0[0]**2+r0[1]**2+r0[2]**2) + radius**2)
    return t


@cuda.jit()
def cuda_step_cylinder(positions, g_x, g_y, g_z, phases, rng_states, time_point, n_of_spins, gamma, step_length, dt, radius, orientation):
    """Kernel function for diffusion inside a sphere"""
    
    # Global thread index on a 1D grid
    thread_id = cuda.grid(1)
    if thread_id >= n_of_spins:
        return

    # Generate random unit step
    step = cuda.local.array(3, numba.double)
    step[0] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[1] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[2] = xoroshiro128p_normal_float64(rng_states, thread_id)
    normalizing_factor = math.sqrt(step[0]**2 + step[1]**2 + step[2]**2)
    step[0] = step[0] / normalizing_factor
    step[1] = step[1] / normalizing_factor
    step[2] = step[2] / normalizing_factor

    # Check for intersection and reflect the step off the surface
    i = 0
    max_iter = 1e4
    check_intersection = True
    intersection = cuda.local.array(3, numba.double)
    normal_vector = cuda.local.array(3, numba.double)
    while check_intersection and i < max_iter:
        i += 1
        t = cylinder_intersection_check(positions[:, thread_id], step, orientation, radius)
        if t <= step_length:
            intersection[0] = positions[0, thread_id] + t*step[0]
            intersection[1] = positions[1, thread_id] + t*step[1]
            intersection[2] = positions[2, thread_id] + t*step[2]
            normal_vector[0] = (intersection[0]*orientation[0]+intersection[1]*orientation[1]+intersection[2]*orientation[2])*orientation[0] - intersection[0]
            normal_vector[1] = (intersection[0]*orientation[0]+intersection[1]*orientation[1]+intersection[2]*orientation[2])*orientation[1] - intersection[1]
            normal_vector[2] = (intersection[0]*orientation[0]+intersection[1]*orientation[1]+intersection[2]*orientation[2])*orientation[2] - intersection[2]
            normalizing_factor = math.sqrt(normal_vector[0]**2 + normal_vector[0]**2 + normal_vector[0]**2)
            normal_vector[0] /= normalizing_factor
            normal_vector[1] /= normalizing_factor
            normal_vector[2] /= normalizing_factor
            reflect_step(positions[:, thread_id], step, intersection, normal_vector, step_length)
        else:
            check_intersection = False
            positions[0, thread_id] = positions[0, thread_id] + step_length*step[0]
            positions[1, thread_id] = positions[1, thread_id] + step_length*step[1]
            positions[2, thread_id] = positions[2, thread_id] + step_length*step[2]

    # Calculate phase shift
    for measurement in range(g_x.shape[1]):
        phases[measurement, thread_id] += gamma * dt * \
                                          (g_x[time_point, measurement] * positions[0, thread_id] + \
                                           g_y[time_point, measurement] * positions[1, thread_id] + \
                                           g_z[time_point, measurement] * positions[2, thread_id])


@cuda.jit(device = True)
def cylinder_intersection_check(r0, step, orientation, radius):
    """Return distance from spin to intersection in step direction"""  
    A = 1 - (step[0]*orientation[0]+step[1]*orientation[1]+step[2]*orientation[2])**2
    B = 2 * (r0[0]*step[0]+r0[1]*step[1]+r0[2]*step[2] - (r0[0]*orientation[0]+r0[1]*orientation[1]+r0[2]*orientation[2]) * (step[0]*orientation[0]+step[1]*orientation[1]+step[2]*orientation[2]))
    C = r0[0]**2+r0[1]**2+r0[2]**2 - radius**2 -(r0[0]*orientation[0]+r0[1]*orientation[1]+r0[2]*orientation[2])**2    
    t = (-B + math.sqrt(B**2 - 4*A*C)) / (2*A)
    return t


@cuda.jit()
def cuda_step_free(positions, g_x, g_y, g_z, phases, rng_states, time_point, n_of_spins, gamma, step_length, dt):
    """Kernel function for free diffusion"""
    
    # Global thread index on a 1D grid
    thread_id = cuda.grid(1)
    if thread_id >= n_of_spins:
        return

    # Generate random step
    step = cuda.local.array(3, numba.double)
    step[0] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[1] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[2] = xoroshiro128p_normal_float64(rng_states, thread_id)
    normalizing_factor = math.sqrt(step[0]**2 + step[1]**2 + step[2]**2)
    step[0] = step_length * step[0] / normalizing_factor
    step[1] = step_length * step[1] / normalizing_factor
    step[2] = step_length * step[2] / normalizing_factor

    # Update positions
    positions[0, thread_id] = positions[0, thread_id] + step[0]
    positions[1, thread_id] = positions[1, thread_id] + step[1]
    positions[2, thread_id] = positions[2, thread_id] + step[2]
    
    # Calculate phase shift
    for measurement in range(g_x.shape[1]):
        phases[measurement, thread_id] += gamma * dt * \
                                          (g_x[time_point, measurement] * positions[0, thread_id] + \
                                           g_y[time_point, measurement] * positions[1, thread_id] + \
                                           g_z[time_point, measurement] * positions[2, thread_id])

@cuda.jit()
def cuda_step_mesh(positions, g_x, g_y, g_z, phases, rng_states, time_point, n_of_spins, gamma, step_length, dt, triangles):
    """Kernel function for mesh simulations"""
    
    # Global thread index on a 1D grid
    thread_id = cuda.grid(1)
    if thread_id >= n_of_spins:
        return

    # Generate random step
    step = cuda.local.array(3, numba.double)
    step[0] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[1] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[2] = xoroshiro128p_normal_float64(rng_states, thread_id)
    normalizing_factor = math.sqrt(step[0]**2 + step[1]**2 + step[2]**2)
    step[0] = step[0] / normalizing_factor
    step[1] = step[1] / normalizing_factor
    step[2] = step[2] / normalizing_factor

    """
    # Just cancel step when colliding with a barrier
    for triangle_idx in range(0,len(triangles), 9):
        A = triangles[triangle_idx:triangle_idx+3]
        B = triangles[triangle_idx+3:triangle_idx+6]
        C = triangles[triangle_idx+6:triangle_idx+9]
        t = triangle_intersection_check(A, B, C, positions[:, thread_id], step)
        if t > 0 and t <= step_length:
            step[0] = 0
            step[1] = 0
            step[2] = 0
            break
    positions[0, thread_id] = positions[0, thread_id] + step[0]*step_length
    positions[1, thread_id] = positions[1, thread_id] + step[1]*step_length
    positions[2, thread_id] = positions[2, thread_id] + step[2]*step_length
    """

    # Proper intersection check with reflection
    i = 0
    max_iter = 1e6
    check_intersection = True
    intersection = cuda.local.array(3, numba.double)
    normal_vector = cuda.local.array(3, numba.double)
    while check_intersection and i < max_iter:
        #if i > max_iter:
            # THROW AN ERROR MESSAGE SOMEHOW FROM HERE   
        i += 1
        for triangle_idx in range(0,len(triangles), 9):
            A = triangles[triangle_idx:triangle_idx+3]
            B = triangles[triangle_idx+3:triangle_idx+6]
            C = triangles[triangle_idx+6:triangle_idx+9]
            t = triangle_intersection_check(A, B, C, positions[:, thread_id], step)
            if t > 0 and t < step_length:
                intersection[0] = positions[0, thread_id] + t*step[0]
                intersection[1] = positions[1, thread_id] + t*step[1]
                intersection[2] = positions[2, thread_id] + t*step[2]
                normal_vector[0] = (B[1]-A[1])*(C[2]-A[2]) - (B[2]-A[2])*(C[1]-A[1])
                normal_vector[1] = (B[2]-A[2])*(C[0]-A[0]) - (B[0]-A[0])*(C[2]-A[2])
                normal_vector[2] = (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
                normalizing_factor = math.sqrt(normal_vector[0]**2 + normal_vector[1]**2 + normal_vector[2]**2)
                normal_vector[0] = normal_vector[0] / normalizing_factor
                normal_vector[1] = normal_vector[1] / normalizing_factor
                normal_vector[2] = normal_vector[2] / normalizing_factor
                reflect_step(positions[:, thread_id], step, intersection, normal_vector, step_length) 
                break
            elif triangle_idx == len(triangles) - 9:
                check_intersection = False
                positions[0, thread_id] = positions[0, thread_id] + step[0]*step_length
                positions[1, thread_id] = positions[1, thread_id] + step[1]*step_length
                positions[2, thread_id] = positions[2, thread_id] + step[2]*step_length
    
    # Calculate phase shift
    for measurement in range(g_x.shape[1]):
        phases[measurement, thread_id] += gamma * dt * \
                                          (g_x[time_point, measurement] * positions[0, thread_id] + \
                                           g_y[time_point, measurement] * positions[1, thread_id] + \
                                           g_z[time_point, measurement] * positions[2, thread_id])


@cuda.jit(device=True)
def triangle_intersection_check(A, B, C, r0, unit_step):
    T = cuda.local.array(3, numba.double)
    E_1 = cuda.local.array(3, numba.double)
    E_2 = cuda.local.array(3, numba.double)
    P = cuda.local.array(3, numba.double)
    Q = cuda.local.array(3, numba.double)
    T[0] = r0[0] - A[0]
    T[1] = r0[1] - A[1] 
    T[2] = r0[2] - A[2] 
    E_1[0] = B[0] - A[0]
    E_1[1] = B[1] - A[1]
    E_1[2] = B[2] - A[2]
    E_2[0] = C[0] - A[0]
    E_2[1] = C[1] - A[1]
    E_2[2] = C[2] - A[2]
    P[0] = unit_step[1] * E_2[2] - unit_step[2] * E_2[1]
    P[1] = unit_step[2] * E_2[0] - unit_step[0] * E_2[2]
    P[2] = unit_step[0] * E_2[1] - unit_step[1] * E_2[0]
    Q[0] = T[1] * E_1[2] - T[2] * E_1[1]
    Q[1] = T[2] * E_1[0] - T[0] * E_1[2]
    Q[2] = T[0] * E_1[1] - T[1] * E_1[0]
    det = (P[0]*E_1[0] + P[1]*E_1[1] + P[2]*E_1[2])  
    if det != 0:    
        t = 1/det * (Q[0]*E_2[0] + Q[1]*E_2[1] + Q[2]*E_2[2])
        u = 1/det * (P[0]*T[0] + P[1]*T[1] + P[2]*T[2])
        v = 1/det * (Q[0]*unit_step[0] + Q[1]*unit_step[1] + Q[2]*unit_step[2])
        if u >= 0 and u <= 1 and v >= 0 and v <= 1 and u + v <= 1:
            return t
        else:
            return np.nan


@cuda.jit()
def cuda_step_plane(positions, g_x, g_y, g_z, phases, rng_states, time_point, n_of_spins, gamma, step_length, dt, directions):
    """Kernel function for 2D diffusion"""
    
    # Global thread index on a 1D grid
    thread_id = cuda.grid(1)
    if thread_id >= n_of_spins:
        return

    # Allocate local memory
    step = cuda.local.array(3, numba.double)

    # Generate random step
    phi = xoroshiro128p_uniform_float64(rng_states, thread_id) * 6.283185307179586
    step[0] = math.cos(phi) * directions[0] + math.sin(phi) * directions[3]
    step[1] = math.cos(phi) * directions[1] + math.sin(phi) * directions[4]
    step[2] = math.cos(phi) * directions[2] + math.sin(phi) * directions[5]
    step[0] = step_length * step[0]
    step[1] = step_length * step[1]
    step[2] = step_length *  step[2]

    # Update positions
    positions[0, thread_id] = positions[0, thread_id] + step[0]
    positions[1, thread_id] = positions[1, thread_id] + step[1]
    positions[2, thread_id] = positions[2, thread_id] + step[2]
    
    # Calculate phase shift
    for measurement in range(g_x.shape[1]):
        phases[measurement, thread_id] += gamma * dt * \
                                          (g_x[time_point, measurement] * positions[0, thread_id] + \
                                           g_y[time_point, measurement] * positions[1, thread_id] + \
                                           g_z[time_point, measurement] * positions[2, thread_id])


@cuda.jit()
def cuda_step_stick(positions, g_x, g_y, g_z, phases, rng_states, time_point, n_of_spins, gamma, step_length, dt, orientation):
    """Kernel function for 1D diffusion"""
    
    # Global thread index on a 1D grid
    thread_id = cuda.grid(1)
    if thread_id >= n_of_spins:
        return

    # Allocate local memory
    step = cuda.local.array(3, numba.double)

    # Generate random step
    if xoroshiro128p_uniform_float64(rng_states, thread_id) > .5:
        step[0] = orientation[0] * step_length
        step[1] = orientation[1] * step_length
        step[2] = orientation[2] * step_length
    else:
        step[0] = -orientation[0] * step_length
        step[1] = -orientation[1] * step_length
        step[2] = -orientation[2] * step_length

    # Update positions
    positions[0, thread_id] = positions[0, thread_id] + step[0]
    positions[1, thread_id] = positions[1, thread_id] + step[1]
    positions[2, thread_id] = positions[2, thread_id] + step[2]
    
    # Calculate phase shift
    for measurement in range(g_x.shape[1]):
        phases[measurement, thread_id] += gamma * dt * \
                                          (g_x[time_point, measurement] * positions[0, thread_id] + \
                                           g_y[time_point, measurement] * positions[1, thread_id] + \
                                           g_z[time_point, measurement] * positions[2, thread_id])


@cuda.jit(device = True)
def reflect_step(r0, step, intersection, normal_vector, step_length):
    """Reflect the step off the surface at collision point"""
    
    # Calculate distance to intersection point and update step length
    step_length -= math.sqrt((r0[0] - intersection[0])**2 + (r0[1] - intersection[1])**2 + (r0[2] - intersection[2])**2)
    
    # Calculate reflection off the surface
    reflected_x = -r0[0] + 2*intersection[0] + 2*normal_vector[0]*((r0[0] - intersection[0])*normal_vector[0] + (r0[1] - intersection[1])*normal_vector[1] + (r0[2] - intersection[2])*normal_vector[2])
    reflected_y = -r0[1] + 2*intersection[1] + 2*normal_vector[1]*((r0[0] - intersection[0])*normal_vector[0] + (r0[1] - intersection[1])*normal_vector[1] + (r0[2] - intersection[2])*normal_vector[2])
    reflected_z = -r0[2] + 2*intersection[2] + 2*normal_vector[2]*((r0[0] - intersection[0])*normal_vector[0] + (r0[1] - intersection[1])*normal_vector[1] + (r0[2] - intersection[2])*normal_vector[2])
    
    # Update step direction and spin position
    step[0] = reflected_x - intersection[0]
    step[1] = reflected_y - intersection[1]
    step[2] = reflected_z - intersection[2]
    normalizing_factor = math.sqrt(step[0]**2+step[1]**2+step[2]**2)
    step[0] /= normalizing_factor    
    step[1] /= normalizing_factor    
    step[2] /= normalizing_factor  
    
    epsilon = 1e-6
    
    r0[0] = intersection[0] + epsilon*step_length*step[0]
    r0[1] = intersection[1] + epsilon*step_length*step[1]
    r0[2] = intersection[2] + epsilon*step_length*step[2]
    
    return
    
    
    