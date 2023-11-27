import os
import logging
import open3d as o3d
import numpy as np

class ImageImporter:
    def __init__(self, image_path):
        self.image_path = image_path