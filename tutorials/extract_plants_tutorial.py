import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from phenomesh.data import get_mesh
from phenomesh.procedures.extract_plants import GeometryFilter
from phenomesh.visualize import view

plot_mesh = get_mesh()

#view(plot_mesh, rotate_z=28)
view(plot_mesh, rotate_z=29.2)

soil_free_mesh = GeometryFilter(mesh=plot_mesh,
                                max_angle=-0.9,
                                within_radius=0.02).filter_mesh()

view(soil_free_mesh)