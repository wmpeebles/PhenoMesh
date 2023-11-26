import xarray as xr
import numpy as np
import open3d as o3d
import networkx as nx
from numba import njit
from scipy.spatial import KDTree
import time
from tqdm import tqdm
import copy

class Loop:
    def __init__(self):
        self.line_set = o3d.t.geometry.LineSet()


class LoopSet:
    def __init__(self, mesh=o3d.t.geometry.TriangleMesh()):
        """
        A LoopSet is an extension of LineSet where lines forming loops are grouped together by connected components.
        Methods exist for generating a LoopSet from a mesh by slicing the mesh with planes, extracting contours,
        selecting contours which form closed loops, and adding the line indices from the loops to a TensorMap with
        default attribute lines shape (N, 1).
        Conditions can be imposed on loop sampling to ensure loops are
        """
        self.mesh = mesh
        self.bbox = self.mesh.get_oriented_bounding_box()

        self.line_set = o3d.t.geometry.LineSet()

        self.G = nx.Graph()
        self.connected_components = list()
        self.node_to_loop = {}
        self.loop_to_nodes = {}

        self.loop_labels = np.ndarray(shape=(0, 1))
        self.loop_perimeters = np.ndarray(shape=(0, 1))
        self.loop_fits = np.ndarray(shape=(0, 3))

        self.rot_z = None

    def rotate_mesh_around_z_axis(self, angle, direction="clockwise"):
        if direction == "clockwise":
            angle = 360 - angle
        center = self.mesh.get_center()

        # Convert angle to radians
        angle_radians = np.radians(angle)

        # Create the rotation matrix
        self.rot_z = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                        [np.sin(angle_radians), np.cos(angle_radians), 0],
                        [0, 0, 1]])

        # Apply the rotation
        self.mesh = self.mesh.rotate(self.rot_z, center=center)
        self.bbox = self.mesh.get_oriented_bounding_box()

    def sample_loops_from_mesh(self, decimate=None,
                               normals=None,
                               samples=None,
                               resolution=0.003):
        """

        :param mesh: An open3d.t.geometry.TriangleMesh mesh to sample loops from
        :param decimate: A fraction from 0-1 representing the decimation percentage
        :param normals:
        :param samples:
        :param resolution: The distance apart between planes used to slice the mesh for each sample.
        A smaller resolution means more loops will be sampled closer apart
        :return:
        """
        timer.reset()

        if decimate is not None:
            #TODO: Decimate does not keep original mesh colors?
            self.mesh = self.mesh.simplify_quadric_decimation(target_reduction=decimate)
            timer.log("Decimating mesh")

        if samples is not None:
            normals = sample_directions(samples=samples)
        else:
            normals = np.asarray(normals).reshape(-1, 3)

        point = self.mesh.get_center()

        for normal in tqdm(normals, f"Getting loops from mesh for normal"):
            bbox = self.bbox.to_legacy()

            rotation_axis = np.cross([0, 0, 1], normal)
            rotation_angle = np.arccos(np.dot([0, 0, 1], normal))
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            rotated_bbox = bbox.rotate(R=rot, center=bbox.center)
            rotated_bbox_box_points = np.asarray(rotated_bbox.get_box_points())

            min_z = np.min(rotated_bbox_box_points[:, 2])
            max_z = np.max(rotated_bbox_box_points[:, 2])
            contour_values = np.arange(start=min_z, stop=max_z, step=resolution)-point.numpy()[2]

            line_set = self.mesh.slice_plane(point=point, normal=normal, contour_values=contour_values)
            line_set.point["loop_normals"] = o3d.core.Tensor(np.repeat([normal], len(line_set.point.positions), axis=0).reshape(-1, 3))
            self._add_line_set(line_set)

        self._find_loops()
        timer.log("Finding loops")
        # TODO: This will not work if mesh does not have normals
        #self._calculate_point_loop_fit()
        #timer.log("Calculating point loop fit")
        #self._calculate_loop_fit()
        #timer.log("Calculating loop fit")
        #TODO: This function takes too long
        #self._calculate_loop_perimeters()
        #timer.log("Calculating loop perimeters")

    def _add_line_set(self, new_line_set):
        if self._line_set_is_empty():
            self.line_set = new_line_set
        else:
            try:
                self._merge_onto_existing_line_set(new_line_set)
            except Exception as e:
                print(f"Could not add line_set {e}")

    def _line_set_is_empty(self):
        if (len(self.line_set.point)) == 0:
            return True
        return False

    def _merge_onto_existing_line_set(self, new_line_set):
        self.line_set.point.positions = self.line_set.point.positions.append(new_line_set.point.positions, axis=0)
        self.line_set.point.loop_normals = self.line_set.point.loop_normals.append(new_line_set.point.loop_normals, axis=0)
        self.line_set.point.normals = self.line_set.point.normals.append(new_line_set.point.normals, axis=0)
        try:
            self.line_set.point.colors = self.line_set.point.colors.append(new_line_set.point.colors, axis=0)
        except KeyError:
            # Line set points do not have colors
            pass
        self.line_set.line.indices = self.line_set.line.indices.append(new_line_set.line.indices + self.line_set.line.indices.max() + 1, axis=0)

    def _create_point_to_loop_mapping(self):
        # Using NetworkX to find connected components
        G = nx.Graph()
        G.add_edges_from(self.line_set.line.indices.numpy())
        self.loop_to_nodes = {i: set(comp) for i, comp in enumerate(nx.connected_components(G))}
        self.node_to_loop = {node: loop_idx for loop_idx, nodes in self.loop_to_nodes.items() for node in nodes}

    def _update_point_to_loop_mapping(self):
        # This step intending to be run after filtering
        self._create_point_to_loop_mapping()
        loop_labels = list(self.loop_labels.flatten())
        loop_to_nodes = {}  # Initialize outside the loop
        for loop_label in loop_labels:
            # Problem here is point indices are not updated after points were filtered in the mapping
            #loop_to_nodes[label] = set(self.loop_to_nodes[label])
            mask = np.isin(self.line_set.point.loops.numpy(), loop_label)
            point_labels = self.line_set.point.labels.numpy()[mask]
            #print(point_labels, point_labels.shape)
            loop_to_nodes[loop_label] = set(point_labels)

        self.loop_to_nodes = loop_to_nodes
        self.node_to_loop = {node: loop_idx for loop_idx, nodes in self.loop_to_nodes.items() for node in nodes}

    def _find_loops(self):
        self._create_point_to_loop_mapping()
        line_indices = self.line_set.line.indices.numpy()

        self.loop_labels = np.asarray(list(self.loop_to_nodes.keys())).reshape(-1, 1)

        # Add loop indices to lines
        line_loop_indices = np.array([self.node_to_loop[line[0]] for line in line_indices])
        self.line_set.line["loops"] = o3d.core.Tensor(line_loop_indices.reshape(-1, 1))
        self.line_set.line["labels"] = np.arange(self.line_set.line.indices.numpy().shape[0]).reshape(-1, 1)

        # Add loop indices to points
        point_loop_indices = np.array([self.node_to_loop[idx] for idx, point in enumerate(self.line_set.point.positions.numpy())])
        self.line_set.point["loops"] = o3d.core.Tensor(point_loop_indices.reshape(-1, 1))
        self.line_set.point["labels"] = np.arange(self.line_set.point.positions.numpy().shape[0]).reshape(-1, 1)

    def filter(self, max_std_normal=0.1, max_std_difference=0.1, min_angle=0.3, min_perimeter=None, max_perimeter=None, min_area=None, max_area=None, distance=None):
        """

        :param max_std_normal:
        :param max_std_difference:
        :param min_perimeter:
        :param max_perimeter:
        :param min_area:
        :param max_area:
        :param distance: Minimum distance between any loop and its neighbor. Loops with points closer than this
        distance will be filtered out. Set to None to disable filtering. Ideal value should be roughly 1/2 to 2/3
        smaller than resolution.
        :return:
        """
        timer.reset()
        valid_indices = np.argwhere(self.loop_labels[:, 0] >= 0).reshape(-1)
        #print(valid_indices)

        if max_std_normal is not None:
            valid_indices = np.intersect1d(valid_indices, np.argwhere(self.loop_fits[:, 0] <= max_std_normal)[:, 0:1].reshape(-1))
        if max_std_difference is not None:
            valid_indices = np.intersect1d(valid_indices, np.argwhere(self.loop_fits[:, 1] <= max_std_difference)[:, 0:1].reshape(-1))
        if min_angle is not None:
            valid_indices = np.intersect1d(valid_indices, np.argwhere(self.loop_fits[:, 2] >= min_angle)[:, 0:1].reshape(-1))
        if min_perimeter is not None:
            valid_indices = np.intersect1d(valid_indices, np.argwhere(self.loop_perimeters >= min_perimeter)[:, 0:1].reshape(-1))
        if max_perimeter is not None:
            valid_indices = np.intersect1d(valid_indices, np.argwhere(self.loop_perimeters <= max_perimeter)[:, 0:1].reshape(-1))

        timer.log("Getting valid loop indices from filter criteria")
        #print(valid_indices.shape)
        self._filter_loops(labels=valid_indices)

        timer.log("Filtering line_set by loop indices")

        if distance is not None:
            self._remove_neighboring_loops(distance=distance)
            timer.log(f"Removing neighboring loops with distance: {distance}")

    def _filter_loops(self):
        # Return new loops opject

    def _filter_loops_old(self, labels=np.array([])):
        labels = labels.reshape(-1)
        loop_mask = np.isin(self.loop_labels, labels)
        loop_mask_t = o3d.core.Tensor(loop_mask)
        loop_mask_3d = np.column_stack((loop_mask, loop_mask, loop_mask))
        loop_mask_3d_t = o3d.core.Tensor(loop_mask_3d)

        line_mask = np.isin(self.line_set.line.loops.numpy(), labels)
        print(line_mask.shape)
        line_mask_t = o3d.core.Tensor(line_mask.reshape(-1, 1))
        line_mask_2d = np.column_stack((line_mask, line_mask))
        line_mask_2d_t = o3d.core.Tensor(line_mask_2d.reshape(-1, 2))

        point_mask = np.isin(self.line_set.point.loops.numpy(), labels)
        point_mask_t = o3d.core.Tensor(point_mask)
        point_mask_3d = np.column_stack((point_mask, point_mask, point_mask))
        point_mask_3d_t = o3d.core.Tensor(point_mask_3d.reshape(-1, 3))

        self.line_set.line["labels"] = self.line_set.line.labels[line_mask_t].reshape((-1, 1))
        self.line_set.line["loops"] = self.line_set.line.loops[line_mask_t].reshape((-1, 1))

        self.line_set.point["labels"] = self.line_set.point.labels[point_mask_t].reshape((-1, 1))
        self.line_set.point["positions"] = self.line_set.point.positions[point_mask_3d_t].reshape((-1, 3))
        self.line_set.point["colors"] = self.line_set.point.colors[point_mask_3d_t].reshape((-1, 3))
        self.line_set.point["normals"] = self.line_set.point.normals[point_mask_3d_t].reshape((-1, 3))
        self.line_set.point["loops"] = self.line_set.point.loops[point_mask_t].reshape((-1, 1))
        self.line_set.point["loop_normals"] = self.line_set.point.loop_normals[point_mask_3d_t].reshape((-1, 3))
        self.line_set.point["loop_fit"] = self.line_set.point.loop_fit[point_mask_3d_t].reshape((-1, 3))

        # Update indices so they point to the correct points since points have been filtered
        self.line_set.line["indices"] = o3d.core.Tensor(np.searchsorted(self.line_set.point.labels.numpy().flatten(),
                                             self.line_set.line.indices[line_mask_2d_t].reshape((-1, 2)).numpy()))

        self.loop_labels = self.loop_labels[loop_mask].reshape(-1, 1)
        self.loop_fits = self.loop_fits[loop_mask_3d].reshape(-1, 3)

        point_loops = np.unique(self.line_set.point.loops.numpy().flatten())
        line_loops = np.unique(self.line_set.line.loops.numpy().flatten())
        loops = np.unique(self.loop_labels.flatten())

        print(point_loops, line_loops, loops)
        assert(point_loops.all() == line_loops.all())
        assert(line_loops.all() == loops.all())

        if self.loop_perimeters.shape[0] == loop_mask.shape:
            self.loop_perimeters = self.loop_perimeters[loop_mask]

        self._update_point_to_loop_mapping()

    def _calculate_point_loop_fit(self):
        # Ensure the inputs are numpy arrays
        normals1 = self.line_set.point.normals.numpy()
        normals2 = self.line_set.point.loop_normals.numpy()
        #print(normals1, normals2)

        # Calculate dot products
        dot_products = np.einsum('ij,ij->i', normals1, normals2)

        # Clip values to avoid errors due to numerical precision
        dot_products = np.clip(dot_products, -1.0, 1.0)

        # Calculate angles in radians
        angles_radians = np.arccos(dot_products)

        # Convert angles to degrees and normalize to [0, 1]
        angles_degrees = np.degrees(angles_radians) / 180.0

        # Calculate the smallest angle when the second normal is flipped
        flipped_angles_degrees = np.degrees(np.minimum(angles_radians, np.pi - angles_radians)) / 180.0

        loop_fit = np.column_stack((angles_degrees, flipped_angles_degrees))
        #print(loop_fit)

        # Combine and return the results
        self.line_set.point["loop_fit"] = o3d.core.Tensor(loop_fit)

    def _calculate_loop_perimeters(self):
        loop_labels = self.loop_labels.flatten()
        line_loops = self.line_set.line.loops.numpy().flatten()
        line_indices = self.line_set.line.indices.numpy()
        point_positions = self.line_set.point.positions.numpy()

        perimeters = []

        for loop_label in loop_labels:
            # Selecting lines belonging to the current loop
            loop_lines = line_indices[line_loops == loop_label]

            # Calculating perimeter using numpy vectorization
            p0 = point_positions[loop_lines[:, 0]]
            p1 = point_positions[loop_lines[:, 1]]
            perimeter = np.sum(np.linalg.norm(p0 - p1, axis=1))
            perimeters.append(perimeter)

        self.loop_perimeters = np.array(perimeters).reshape(-1, 1)

    def _calculate_loop_fit(self):
        #points_loop_fit = self.line_set.point.loop_fit.numpy()

        loop_fits = []

        for loop_label in self.loop_labels.flatten():
            #loop_point_indices = np.array(list(self.loop_to_nodes[loop_label]))
            #loop_point_loop_fits = points_loop_fit[loop_point_indices]

            mask = self.line_set.point.loops.numpy() == loop_label
            loop_point_loop_fits = self.line_set.point.loop_fit.numpy()[np.column_stack((mask, mask))].reshape(-1, 2)

            #print(loop_point_loop_fits)

            std_dev_loop = np.std(loop_point_loop_fits, axis=0)
            mean_loop = np.mean(loop_point_loop_fits, axis=0)


            std_dev_difference = std_dev_loop[0] - std_dev_loop[1]
            #print(std_dev_loop, round(abs(std_dev_difference), 6))

            # Measure variance(max angle) - loops with small variance for max angle mean loop is good fit

            # Measure variance(max angle) - variance(min angle). If high, indicates loop fit could be better if
            # loop was rotated at center

            # Measure average(min angle) - 0.5 means loop is neither growing nor shrinking. Closer to 0 means greater
            # growing or shrinking (shrink or growth can be determined by looking at max angle)
            #print(mean_loop[1])
            loop_fits.append((std_dev_loop[0], std_dev_difference, mean_loop[1]))

        self.loop_fits = np.array(loop_fits).reshape(-1, 3)
        #print(self.loop_fits)

    def _remove_neighboring_loops(self, distance):
        self._update_point_to_loop_mapping()

        points = self.line_set.point.positions.numpy()
        point_loops = self.line_set.point.loops.numpy().flatten()
        loop_labels = self.loop_labels.flatten()

        # Building a KDTree for efficient spatial queries
        kdtree = KDTree(points)

        # Set for storing loops to remove
        loops_to_remove = set()

        # Process each loop only once
        for loop_label in loop_labels:
            if loop_label in loops_to_remove:
                continue

            #loop_points_indices = np.array(list(self.loop_to_nodes[loop_label]))
            loop_points_indices = self.line_set.point.loops.numpy().flatten()
            loop_points = points[loop_points_indices]

            # Query KDTree for all points within 'distance' for all points in the loop
            indices_within_distance = kdtree.query_ball_point(loop_points, distance)

            # Flattening and unique filtering of indices
            indices_within_distance = set(np.unique(np.hstack(indices_within_distance)))

            # Remove indices belonging to the current loop
            indices_within_distance -= set(loop_points_indices)

            # Identifying loops to remove based on proximity and perimeter comparison
            for point_idx in indices_within_distance:
                #print(point_idx)
                other_loop_label = point_loops[point_idx]
                #print(other_loop_label)
                if other_loop_label != loop_label:
                    current_angle = self.loop_fits[np.where(loop_labels == loop_label)[0][0]][2]
                    other_angle = self.loop_fits[np.where(loop_labels == other_loop_label)[0][0]][2]
                    if current_angle < other_angle:
                        loops_to_remove.add(loop_label)
                    else:
                        loops_to_remove.add(other_loop_label)

        loop_to_nodes = copy.deepcopy(self.loop_to_nodes)

        # Removing the identified loops
        for loop_label in loops_to_remove:
            del loop_to_nodes[loop_label]

        # Update line set to reflect the removals
        self._filter_loops(labels=np.array(list(loop_to_nodes.keys())))

    @staticmethod
    @njit
    def _euclidean_distance(point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)**0.5



    def view_mesh_2(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.add_geometry(self.mesh.to_legacy())
        vis.add_geometry(self.bbox.to_legacy())

        o3d.visualization.gui.Application.instance.initialize()
        window = o3d.visualization.gui.Application.instance.create_window("scenewidget", 1024, 1024)
        widget = o3d.visualization.gui.SceneWidget()
        widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                      [10, 10, 10])
        fov_degs = 60
        widget.setup_camera(fov_degs, bbox, [0, 0, 0])
        widget.scene.camera.look_at([0, 0, 0],
                                    [0, 50, 50],
                                    [0, -1, 0])
        widget.scene.camera.set_projection(o3d.visualization.rendering.Camera.Projection.Ortho, -10, 10, -10, 10, 1, 10)

        vis.run()

    def view_mesh(self):
        o3d.visualization.draw_geometries([self.mesh.to_legacy(), self.bbox.to_legacy()])

    def view(self, *args, **kwargs):
        o3d.visualization.draw(self.line_set,
                               show_skybox=False,
                               *args, **kwargs)

    def read(self, path):
        pass
    def write(self, path):
        pass


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


class Time:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.messages = []

    def log(self, message="Time elapsed"):
        now = time.time()
        difference = now - self.last_time
        message = f"{message}: {round(difference, 2)} seconds"
        print(message)
        self.messages.append(message)
        self.last_time = now

    def reset(self):
        self.last_time = time.time()

    def elapsed_total(self, message="Time elapsed in total"):
        self.log(message=message)
        for message in self.messages:
            print(message)


timer = Time()
