import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phenomesh.experiment_types.field_trial import Field, Plot
from phenomesh.observations import Observation
from phenomesh.geometry import Mesh
from phenomesh.segmentation.cross_sections import CrossSections

# Read in some mesh data
mesh = Mesh()
mesh_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                                          "examples", "meshes", "2023-04-11_308_decimated",
                                          "2023-04-11_308.ply")
mesh.read(mesh_path)

# Add the mesh data to an observation
observation = Observation(data=mesh,
                          year=2023,
                          month=4,
                          day=11)

# Now add this observation to a plot in a field. To do this we'll initialize a field and a plot
# Create a Field named field
field = Field()

# Create a plot named 102
field.plots["102"] = Plot()

# Add the observation to the plot. We'll name this observation "2023-04-11_mesh"
field.plots["102"].observations["2023-04-11_mesh"] = observation

# View the observation
field.plots["102"].observations["2023-04-11_mesh"].view()

# The next step is segmenting the plot mesh into individual plant meshes
# To do this we'll use a technique that finds cross-sections in a mesh with attributes consistent with what we would
# expect cross-sections from an actual plant would have
cross_sections = CrossSections()

# Find cross-sections in our mesh
cross_sections.find(source_mesh=mesh.data.to_legacy(), resolution=0.003, thickness=0.003,
                    samples=3, min_points=10, max_points=100, max_perimeter=None, max_area=None)

# Label cross-sections using hdbscan
cross_sections.label(leaf_size=40, min_cluster_size=200, min_samples=10, gen_min_span_tree=True)

cross_sections.print_stats()

# Rank cross-sections
cross_sections.rank(resolution=0.01)

cross_sections.print_stats()
cross_sections.view_centroids()
cross_sections.view_vertices()

cross_sections.filter(max_rank=0)

cross_sections.print_stats()
cross_sections.view_centroids()
cross_sections.view_vertices()

