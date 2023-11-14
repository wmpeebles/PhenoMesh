import open3d as o3d
import numpy as np
import networkx as nx
from numba import njit
from scipy.spatial import KDTree


class Loops:
    def __init__(self):
        self.mesh = o3d.t.geometry.TriangleMesh()
        self.line_set = o3d.t.geometry.LineSet()

        self.G = nx.Graph()
        self.connected_components = list()
        self.node_to_loop = {}
        self.loop_to_nodes = {}

        self.loop_perimeters = np.ndarray(shape=(0, 1))

    def get_loops_from_mesh(self, mesh, normals=None, samples=None):
        self.mesh = mesh

        point = self.mesh.get_center()

        if samples is not None:
            normals = sample_directions(samples=samples)

        for normal in normals:
            line_set = mesh.slice_plane(point=point, normal=normal, contour_values=np.linspace(-5, 5, num=250*10))
            self._add_line_set(line_set)
            self._find_loops()

        self._calculate_loop_perimeters()
        self._remove_neighboring_loops(distance=0.003)

    def _add_line_set(self, loops):
        if (len(self.line_set.point)) == 0:
            self.line_set = loops
        else:
            self._merge_onto_existing_loops(loops)

    def _merge_onto_existing_loops(self, loops):
        self.line_set.point.positions = self.line_set.point.positions.append(loops.point.positions, axis=0)
        self.line_set.point.normals = self.line_set.point.normals.append(loops.point.normals, axis=0)
        self.line_set.point.colors = self.line_set.point.colors.append(loops.point.colors, axis=0)
        self.line_set.line.indices = self.line_set.line.indices.append(loops.line.indices + self.line_set.line.indices.max() + 1, axis=0)

    def _find_loops(self):
        line_indices = self.line_set.line.indices.numpy()

        # Add edges (lines) to the graph
        self.G.add_edges_from(line_indices)

        # Create a mapping from node to loop index
        self.node_to_loop = {}
        self.loop_to_nodes = {}
        for contour_index, node_indices in enumerate(nx.connected_components(self.G)):
            self.loop_to_nodes[contour_index] = node_indices
            for node in node_indices:
                self.node_to_loop[node] = contour_index

        # Assign contour index to each line
        loop_indices = np.array([self.node_to_loop[line] for line in line_indices[:, 0:1].reshape((-1))])

        self.line_set.line["loops"] = o3d.core.Tensor(loop_indices.reshape(-1, 1))

    def _calculate_loop_perimeters(self):
        """
        Loop Properties include perimeter, area, and centroid
        :return:
        """
        max_loop_index = self.line_set.line["loops"].max()
        perimeters = []
        i = 0

        while i <= max_loop_index:
            perimeter = 0
            line_indices = self.line_set.line.indices[self.line_set.line.loops.reshape(-1) == i].numpy()

            for line in line_indices:
                p0 = self.line_set.point.positions[line[0]].numpy()
                p1 = self.line_set.point.positions[line[1]].numpy()
                #print(p0, p1)
                perimeter += self._euclidean_distance(p0, p1)
                #print(perimeter)
            i += 1
            perimeters.append(perimeter)
            #print(perimeter)

        self.loop_perimeters = np.asarray(perimeters).reshape(-1, 1)

    def _remove_neighboring_loops(self, distance):
        points = self.line_set.point.positions.numpy()
        perimeters = self.loop_perimeters.reshape(-1)

        # Flatten the points array to work with KDTree
        #flattened_points = np.vstack([points[node] for _, node in enumerate(self.loop_to_nodes)])
        #print(flattened_points)

        kdtree = KDTree(points)

        # List to keep track of loops to remove
        loops_to_remove = set()

        for loop_idx, loop_points_indices in self.loop_to_nodes.items():
            if loop_idx in loops_to_remove:
                continue

            for point_idx in loop_points_indices:
                point = points[point_idx]
                # Find points within the danger radius
                indices = kdtree.query_ball_point(point, distance)

                # Check if these points belong to a different loop
                for idx in indices:
                    other_loop_idx = self.find_loop_of_point(idx, self.loop_to_nodes)
                    if other_loop_idx != loop_idx and other_loop_idx not in loops_to_remove:
                        # Compare perimeters and decide which loop to remove
                        if perimeters[loop_idx] > perimeters[other_loop_idx]:
                            loops_to_remove.add(loop_idx)
                        else:
                            loops_to_remove.add(other_loop_idx)

        # Removing the loops with larger perimeters
        for loop_idx in loops_to_remove:
            del self.loop_to_nodes[loop_idx]

        print(list(self.loop_to_nodes.keys()))
        self.filter_line_set_by_loop_indices(loop_indices=np.asarray(list(self.loop_to_nodes.keys())))

    def find_loop_of_point(self, point_idx, edge_loop_to_points):
        for loop_idx, loop_points_indices in edge_loop_to_points.items():
            if point_idx in loop_points_indices:
                return loop_idx
        return None

    @staticmethod
    @njit
    def _euclidean_distance(point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)**0.5

    def filter(self, min_perimeter=None, max_perimeter=None, min_area=None, max_area=None):
        valid_indices = np.argwhere(self.loop_perimeters >= 0)[:, 0:1].reshape(-1)
        print(valid_indices)

        if min_perimeter is not None:
            valid_indices = np.intersect1d(valid_indices, np.argwhere(self.loop_perimeters >= min_perimeter)[:, 0:1].reshape(-1))
        if max_perimeter is not None:
            valid_indices = np.intersect1d(valid_indices, np.argwhere(self.loop_perimeters <= max_perimeter)[:, 0:1].reshape(-1))

        self.filter_line_set_by_loop_indices(valid_indices)

    def filter_line_set_by_loop_indices(self, loop_indices=np.array([])):
        filtered_line_indices = [i for i, loop_index in enumerate(self.line_set.line.loops.numpy().tolist()) if loop_index in loop_indices]
        self.line_set.line["indices"] = o3d.core.Tensor([self.line_set.line.indices.numpy()[i] for i in filtered_line_indices])

    def view(self, *args, **kwargs):
        o3d.visualization.draw(self.line_set,
                               show_skybox=False,
                               *args, **kwargs)


def sample_directions(samples=10):
    """
    Generates a list of normals evenly distributed over the surface of a sphere.

    :param samples: The number of samples (normals) to generate.
    :return: A list of 3D unit vectors (normals) uniformly distributed.
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return points


def calculate_contour_area(contour_points):
    """
    Calculates the area of a 3D contour by projecting it onto the XY plane.

    :param contour_points: A list of 3D points representing the contour.
    :return: The area of the contour.
    """
    # Project contour points onto the XY plane
    projected_points = [np.array([x, y]) for x, y, _ in contour_points]

    # Calculate area using the shoelace formula
    n = len(projected_points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += projected_points[i][0] * projected_points[j][1]
        area -= projected_points[j][0] * projected_points[i][1]
    area = abs(area) / 2.0

    return area


class ConnectedLoops:
    def __init__(self):
        pass
