import open3d as o3d
import os


class Fields:
    def __init__(self):
        """
        Class for storing multiple fields
        """
        self.fields = {}


class Field:
    def __init__(self):
        """
        The field containing plots and plants
        """
        self.observations = {}
        self.plots = {}


class Plot:
    def __init__(self, mesh=o3d.geometry.TriangleMesh):
        """
        A plot within a field
        """
        self.observations = {}
        self.plants = {}


class Plant:
    def __init__(self):
        """
        A plant within a plot
        """


