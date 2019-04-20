import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def show_trajectories(trajectories):
    """Display spin trajectories as a 3D matplotlib figure.

    Parameters
    ----------
    trajectories : numpy array
        Numpy array of shape [3 x N x M] where N is the number of spins and
        M is the number of time steps.

    Returns
    -------
    None
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n_of_spins = trajectories.shape[1]
    for walker in range(n_of_spins):
        x = trajectories[0, walker, :]
        y = trajectories[1, walker, :]
        z = trajectories[2, walker, :]
        ax.plot(x, y, z, alpha = .6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.ticklabel_format(style='sci', scilimits=(0,0))
    plt.show()


def show_mesh(triangles, alpha_level = .5):
    """Display triangular mesh as 3D matplotlib figure.

    Parameters
    ----------
    trajectories : numpy array
        Numpy array of shape [N x 3 x 3] where N is the number of triangles.
        The second dimension represents points defining the triangle.
    alpha_level : double
        Number between 0 and 1 defining triangle opacity.


    Returns
    -------
    None
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([np.min(np.min(triangles,0),0)[0],
                 np.max(np.max(triangles,0),0)[0]])
    ax.set_ylim([np.min(np.min(triangles,0),0)[1],
                 np.max(np.max(triangles,0),0)[1]])
    ax.set_zlim([np.min(np.min(triangles,0),0)[2],
                 np.max(np.max(triangles,0),0)[2]]) 
    triangles = triangles.ravel()
    for triangle_idx in range(0,len(triangles), 9):
        A = triangles[triangle_idx:triangle_idx+3]
        B = triangles[triangle_idx+3:triangle_idx+6]
        C = triangles[triangle_idx+6:triangle_idx+9]
        vtx = np.array([A,B,C])
        tri = Poly3DCollection([vtx], alpha = alpha_level)
        face_color = np.random.random(3)
        tri.set_facecolor(face_color)
        ax.add_collection3d(tri)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.ticklabel_format(style='sci', scilimits=(0,0))
    plt.show()


def rotation_matrix_to_align_k_with_v(k, v):
    """Calculate rotation matrix to align vector k with vector v.

    Parameters
    ----------
    v : numpy array
        3D vector.
    k : numpy array
        3D vector.
     

    Returns
    -------
    R : numpy array
        Rotation matrix.

    """
    k = k / np.linalg.norm(k)
    v = v / np.linalg.norm(v)
    axis = np.cross(k, v)
    if np.linalg.norm(axis) == 0:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.arcsin(np.linalg.norm(np.cross(k, v)) /
                      (np.linalg.norm(k)*np.linalg.norm(v)))
    if np.dot(k, v) < 0:
        angle = np.pi - angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + \
        np.sin(angle) * K + \
        (1 - np.cos(angle)) * np.matmul(K, K)
    return R


def calculate_ST_b_value(G, delta, DELTA):
    """Calculate b-value of a Stejskal-Tanner pulse sequence.

    Parameters
    ----------
    G : double
        Gradient strength.
    delta : double
        Diffusion encoding time.
    DELTA : double
        Diffusion time.

    Returns
    -------
    double
        b-value.
        
    """
    gamma = 267.513e6
    bval = gamma**2 * G**2 * delta**2 * (DELTA - delta/3)
    return bval
