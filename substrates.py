import math
import numpy as np
import numba
from numba import cuda
import disimpy.utils as utils
import disimpy.simulation as simulation
from numba.cuda.random import create_xoroshiro128p_states, \
                              xoroshiro128p_uniform_float64

class Ball():
    """Substrate object ball.
    
    """
    def __init__(self, diffusivity, radius):
        """Initiate instance of object.

        Parameters
        ----------
        diffusivity : double
        radius : double
        
        """
        self.type = 'ball'
        self.diffusivity = diffusivity
        self.radius = radius   
    
    def fill_uniformly(self, n_of_spins, seed = 0):
        """
        Calculate positions for spins evenly distributed in the substrate.
        
        Parameters
        ----------
            n_of_spins : int
                Number of spins.
            seed : int
                Seed for random number generation.
        
        Returns
        -------
            positions : numpy array
                Calculated positions for spins.
        """
        np.random.seed(seed)
        positions = (np.random.random((3,int(3 * n_of_spins))) - .5) \
                    * 2 * self.radius
        positions = positions[:,np.linalg.norm(positions, axis = 0) 
                              < self.radius]
        positions = positions[:,0:int(n_of_spins)] 
        return positions 


class Cylinder():
    """Substrate object cylinder.
    
    """
    def __init__(self, diffusivity, radius, orientation):
        """Initiate instance of object.

        Parameters
        ----------
        diffusivity : double
        radius : double
        orientation : numpy array
        
        """
        self.type = 'cylinder'
        self.diffusivity = diffusivity
        self.radius = radius
        self.orientation = np.array(orientation / np.linalg.norm(orientation))    
    
    def fill_uniformly(self, n_of_spins, seed = 0):
        """
        Calculate positions for spins evenly distributed in the substrate.
        
        Parameters
        ----------
            n_of_spins : int
                Number of spins.
            seed : int
                Seed for random number generation.
        
        Returns
        -------
            positions : numpy array
                Calculated positions for spins.
        """
        np.random.seed(seed)
        positions = (np.random.random((2,int(3 * n_of_spins))) - .5) \
                    * 2 * self.radius
        positions = positions[:,np.linalg.norm(positions, axis = 0) < 
                              self.radius]
        positions = positions[:,0:int(n_of_spins)] 
        positions = np.append(np.zeros(n_of_spins)[np.newaxis,:], 
                              positions, axis = 0)
        R = utils.rotation_matrix_to_align_k_with_v(np.array([1,0,0]), 
                                                    self.orientation)
        positions = np.matmul(R, positions)
        return positions 
    

class Free():
    """Substrate object free diffusion.
    
    """
    def __init__(self, diffusivity):
        """Initiate instance of object.

        Parameters
        ----------
        diffusivity : double
        
        """
        self.type = 'free'
        self.diffusivity = diffusivity


class Mesh():
    """Substrate object triangular mesh.
    
    """
    
    def __init__(self, diffusivity, fname):
        """Initiate instance of object.

        Parameters
        ----------
        diffusivity : double
        fname : string
        
        """
        self.type = 'mesh'
        self.diffusivity = diffusivity
        self.triangles = self.load_mesh(fname)
        
    def load_mesh(self, file):
        """Load mesh from .ply file
        
        Parameters
        ----------
        fname : string        
        
        Returns
        -------
        triangles : numpy array
        
        """
        with open(file, 'r') as mesh_file:
            mesh_data = mesh_file.readlines()    
        if mesh_data[2] == 'comment closed surface\n':
            n_of_vertices = int(mesh_data[3].split()[2])
            self.closed = True
        else:
            n_of_vertices = int(mesh_data[2].split()[2])
            self.closed = False
        first_vertice_idx = mesh_data.index('end_header\n')+1
        vertices = np.loadtxt(mesh_data[first_vertice_idx:first_vertice_idx+
                                        n_of_vertices])
        faces = np.loadtxt(mesh_data[first_vertice_idx+
                                     n_of_vertices::])[:,1:4]    
        triangles = np.zeros((faces.shape[0], 3, 3))
        for i in range(faces.shape[0]):
            triangles[i,:,:] = vertices[np.array(faces[i], dtype=int)]
        triangles = np.add(triangles, - np.min(np.min(triangles,0),0))
        return triangles
    
    def fill_uniformly(self, n_of_spins, seed):
        """
        Calculate positions for spins evenly distributed in the substrate.
        
        Parameters
        ----------
            n_of_spins : int
                Number of spins.
            seed : int
                Seed for random number generation.
        
        Returns
        -------
            positions : numpy array
            Calculated positions for spins.
        """
        triangles = self.triangles
        block_size = 256
        grid_size = int(math.ceil(float(n_of_spins) / block_size))
        stream = cuda.stream()
        rng_states = create_xoroshiro128p_states(grid_size * block_size, 
                                                 seed = seed, 
                                                 stream=stream)
        positions = np.zeros([3,n_of_spins])
        d_positions = cuda.to_device(positions, stream=stream)
        d_triangles = cuda.to_device(triangles.ravel(), stream=stream)
        d_max = cuda.to_device(np.max(np.max(triangles,0),0), stream=stream)
        fill_uniformly_cuda[grid_size, block_size, stream](d_positions, 
                                                           d_triangles, 
                                                           d_max, 
                                                           rng_states)
        stream.synchronize()
        positions = d_positions.copy_to_host(stream=stream)
        return positions    
            

class Plane():
    """Substrate object 2D diffusion.
    
    """
    def __init__(self, diffusivity, norm):   
        """Initiate instance of object.

        Parameters
        ----------
        diffusivity : double
        norm : numpy array
        
        """
        self.type = 'plane'
        self.diffusivity = diffusivity
        self.norm = np.array(norm)
        if np.all(norm == [1,0,0]):
                v_1 = np.cross(self.norm, np.array([0,1,0]))
        else:
                v_1 = np.cross(self.norm, np.array([1,0,0]))
        v_2 = np.cross(v_1, self.norm)
        v_1 = v_1 / np.linalg.norm(v_1)
        v_2 = v_2 / np.linalg.norm(v_2)
        self.v1 = v_1
        self.v2 = v_2


class Stick():
    """Substrate object 1D diffusion.
    
    """
    def __init__(self, diffusivity, orientation):
        """Initiate instance of object.

        Parameters
        ----------
        diffusivity : double
        orientation : numpy array
        
        """
        self.type = 'stick'
        self.diffusivity = diffusivity
        self.orientation = np.array(orientation / np.linalg.norm(orientation)) 
        
        

@cuda.jit()
def fill_uniformly_cuda(positions, triangles, max, rng_states):    
    """Cuda kernel function for calculating spin positions inside the 
        triangular mesh."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[1]:
        return    
    inside = False
    while not inside:
        intersections = 0
        r0 = cuda.local.array(3, numba.double)
        unit_step = cuda.local.array(3, numba.double)
        r0[0] = xoroshiro128p_uniform_float64(rng_states, thread_id) * max[0]
        r0[1] = xoroshiro128p_uniform_float64(rng_states, thread_id) * max[1]
        r0[2] = xoroshiro128p_uniform_float64(rng_states, thread_id) * max[2]
        unit_step[0] = xoroshiro128p_uniform_float64(rng_states, 
                                                     thread_id) - .5
        unit_step[1] = xoroshiro128p_uniform_float64(rng_states, 
                                                     thread_id) - .5
        unit_step[2] = xoroshiro128p_uniform_float64(rng_states, 
                                                     thread_id) - .5
        normalizing_factor = math.sqrt(unit_step[0]**2 + unit_step[1]**2 
                                       + unit_step[2]**2)
        unit_step[0] = unit_step[0] / normalizing_factor
        unit_step[1] = unit_step[1] / normalizing_factor
        unit_step[2] = unit_step[2] / normalizing_factor
        for triangle_idx in range(0,len(triangles), 9):
            A = triangles[triangle_idx:triangle_idx+3]
            B = triangles[triangle_idx+3:triangle_idx+6]
            C = triangles[triangle_idx+6:triangle_idx+9]
            t = simulation.triangle_intersection_check(A, B, C, r0, unit_step)
            if t > 0:
                intersections = intersections + 1
        if intersections % 2 != 0:
            inside = True
    positions[0,thread_id] = r0[0]
    positions[1,thread_id] = r0[1]
    positions[2,thread_id] = r0[2]