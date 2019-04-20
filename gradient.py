import numpy as np

def create_stejskal_tanner_gradient_array(G, DELTA, delta, n_of_steps, 
                                          direction):
    """Calculate gradient array representing a Stejskal-Tanner pulse sequence.

    Parameters
    ----------
    G : double
        Gradient magnitude.
    DELTA : double
        Diffusion time.
    delta : double
        Diffusion encoding time.
    n_of_steps : int
        Number of time steps.
    direction : numpy array
        Direction of the diffusion encoding gradient.

    Returns
    -------
    gradient_array : numpy array
        Gradient array.
    dt : double
        Time step duration.
        
    """
    direction = direction / np.linalg.norm(direction)
    t_max = DELTA + delta
    gradient_array = np.zeros((3, int(n_of_steps)))
    dt = t_max/n_of_steps
    gradient_array[0, 0:int(delta/dt)] = G * direction[0]
    gradient_array[2, 0:int(delta/dt)] = G * direction[1]
    gradient_array[1, 0:int(delta/dt)] = G * direction[2]
    gradient_array[0, int(DELTA*1/dt):int((DELTA+delta)*1/dt)] = -G * \
                                                                 direction[0]
    gradient_array[1, int(DELTA*1/dt):int((DELTA+delta)*1/dt)] = -G * \
                                                                 direction[1]
    gradient_array[2, int(DELTA*1/dt):int((DELTA+delta)*1/dt)] = -G * \
                                                                 direction[2]
    gradient_array = gradient_array[:,:,np.newaxis]
    return gradient_array, dt

def read_general_scheme_file(fname):
    """Read gradient information from a scheme file compatible with Camino.

    See http://camino.cs.ucl.ac.uk/index.php?n=Tutorials.GenwaveTutorial for
    details.

    Parameters
    ----------
    fname : string
        Name of scheme file.

    Returns
    -------
    gradient_array : numpy array
        Gradient array.
    dt : double
        Time step duration.
    
    """
    with open(fname, 'r') as schemefile:
        scheme_file_data = schemefile.readlines()
    n_of_measurements = len(scheme_file_data) - 1
    K = int(scheme_file_data[1].rstrip().split()[0])
    waveform_array = np.zeros((3, n_of_measurements, K))
    dt = float(scheme_file_data[1].rstrip().split()[1])    
    for i in range(n_of_measurements):
        waveform_array[:,i,:] = np.array(scheme_file_data[i+1].\
                                rstrip().split()[2::]).astype(np.double).\
                                reshape((K, 3)).T
    return waveform_array, dt
