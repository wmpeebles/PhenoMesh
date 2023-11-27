import open3d as o3d
import numpy as np
import cv2


class Mesh:
    def __init__(self):
        self.data: o3d.geometry.TriangleMesh() = None


class Image:
    def __init__(self):
        self.data: np.ndarray(shape=['w', 'h', 2]) = None
