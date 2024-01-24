import numpy as np
from helper_ply import read_ply,write_ply
import os

def load_data(filepath: str):
    """Load point cloud data in binary format.

        Args:
            filepath (str): Path to binary format point cloud data.

        Returns:
            tuple: A tuple containing two numpy arrays. The first array contains
                the (x, y, z) coordinates of the points in the point cloud, and the
                second array contains the RGB colors of the points.
        """

    _, extension = os.path.splitext(filepath)
    if 'ply' in extension:
        data = read_ply(filepath)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.vstack([data['red'], data['green'], data['blue']]).T/255.0
    else:
        with open(filepath, 'rb') as f:
            buffer = f.read()
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('_', 'u1')])
        data = np.frombuffer(buffer, dtype=dtype)        
        points = np.vstack([data['x'], data['y'], data['z']]).T
        colors = np.vstack([data['r'], data['g'], data['b']]).T / 255.0
        # data = np.hstack((points, colors))
    return points, colors


def load_label(filepath: str):
    return np.load(filepath).astype(np.uint16)


def save_label(filepath: str, labels,points,colors):
    _, extension = os.path.splitext(filepath)
    if 'bin' in extension:
        with open(filepath, 'wb') as f:
            np.save(f, labels.astype(np.uint8))
    else:
        print(colors,labels)
        colors=colors*255
        write_ply(filepath,[points,colors.astype(np.uint16),labels.T],['x','y','z','red','green','blue','sem','ins'])
