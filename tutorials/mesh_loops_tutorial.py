import os
import sys

#logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phenomesh.io.from_metashape import PLYMeshWithPNGTextures
from phenomesh.loops.mesh_loops import LoopSet, sample_normals

# Create a mesh observation
mesh_path = os.path.join("data", "plot_mesh", "2023-04-11_308", "2023-04-11_308.ply")

mesh = PLYMeshWithPNGTextures().import_mesh(ply_path=mesh_path)

# Now is the fun part - separating out the plants from the soil and non-plant objects
# To do this, we'll find loops in the mesh - loops represent cross-sections
loops = LoopSet()

# Choose normals (slicing direction)
vertical_normal = [[0, 0, 1]]
fast_normals = [[0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]]
quality_normals = sample_normals(3)

# Choose resolution (how far apart should the slices be per normal in meters?)
fast_resolution = 0.04
fast_resolution_2  = 0.01 #12 min, 1000s per iteration
quality_resolution = 0.005 #12 min, 1000s per iteration

# Slice the mesh
"""loops.from_mesh(mesh=mesh,
                normals=vertical_normal,
                resolution=quality_resolution,
                method="approximate")"""
loops.from_mesh(mesh=mesh,
                normals=fast_normals,
                resolution=quality_resolution,
                method="exact")

# Filter by skew (tilt or angle)
#loops_filtered = loops.filter(max_skew=0.2)
loops_filtered = loops.filter(max_skew=0.1)

# Filter by growth (Is this from part of the mesh where the surface is growing or shrinking?)
loops_filtered2 = loops_filtered.filter(growth=(-0.5, 0.5))

# Filter by loop area
loops_filtered3 = loops_filtered2.filter(max_area=0.001)

loops_filtered3.write_line_set(os.path.join("data", "plot_loops", "plot_2023-04-11_308_exact_r0.005_n3.ply"))

# Filter by loop hue
green_leaves = loops_filtered3.filter(hue=(0.1, 0.6))

green_leaves.write_line_set(os.path.join("data", "plot_loops", "leaves_2023-04-11_308_exact_r0.005_n3.ply"))

#green_leaves_mesh = green_leaves.connect_loops()

# View all loops
loops.view()
#loops_filtered.view()
#loops_filtered2.view()
#loops_filtered3.view()
green_leaves.view()
#green_leaves_mesh.view()

# TODO: show simple mesh calculations for leaf area/green leaf are

# Now that we have identified plants, we can also separate out plants and add these plant meshes to our plot
# TODO: Separate out plants