import open3d as o3d
import numpy as np
from scipy.ndimage import label
from scipy.spatial import ConvexHull
from tqdm import tqdm
from matplotlib import pyplot as plt
import hdbscan
import copy


class MeshVolume:
    def __init__(self, mesh=o3d.t.geometry.TriangleMesh(),
                 rot=np.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]]),
                 resolution=0.003):
        """
        A MeshVolume represents a volumetric image produced from a mesh for the purpose of slicing
        """
        self.mesh = mesh
        self.rot = rot
        self.resolution = resolution

        self.n_labels = 0

        self.centroids = None
        self.vertices = None

    def rotate_mesh(self, rot):
        mesh = copy.deepcopy(self.mesh)
        mesh = mesh.rotate(rot, center=(0, 0, 0))
        return mesh

    def mesh_should_be_rotated(self):
        if self.rot.all() == np.array([[1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., 1.]]).all():
            return False
        return True

    def find_cross_sections(self, thickness=0.003, min_points=10, max_points=250, max_perimeter=None, max_area=None):
        """

        :return: centroids_pcd, vertices_pcd
        """
        mesh = self.mesh
        if self.mesh_should_be_rotated():
            mesh = self.rotate_mesh(self.rot)

        min_coord = np.min(np.asarray(mesh.vertices), axis=0)
        voxel_grid = mesh_to_voxel_grid(mesh, voxel_size=self.resolution)
        volume = voxel_grid_to_volume(voxel_grid)

        centroids_list = []
        vertices_list = []

        #normals_map = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}

        for axis in ['x', 'y', 'z']:
            centroids_array, vertices_array = self.find_cross_sections_along_axis(volume, axis, min_points, max_points,
                                                                                  max_perimeter, max_area)
            if centroids_array.any():  # Check if any cross-sections were found by checking centroids
                centroids_pcd = self.add_centroids_to_point_cloud(centroids=centroids_array, min_coord=min_coord)
                vertices_pcd = self.add_vertices_to_point_cloud(vertices=vertices_array, min_coord=min_coord)
                centroids_list.append(centroids_pcd)
                vertices_list.append(vertices_pcd)

        centroids_pcd = merge_point_clouds_from_list(centroids_list)
        vertices_pcd = merge_point_clouds_from_list(vertices_list)

        if self.mesh_should_be_rotated():
            centroids_pcd = centroids_pcd.rotate(R=o3d.core.Tensor(np.transpose(self.rot)),
                                                 center=o3d.core.Tensor(np.array([0, 0, 0])))
            vertices_pcd = vertices_pcd.rotate(R=o3d.core.Tensor(np.transpose(self.rot)),
                                               center=o3d.core.Tensor(np.array([0, 0, 0])))

        return centroids_pcd, vertices_pcd

    def find_cross_sections_along_axis(self, volume, axis, min_points, max_points, max_perimeter, max_area):
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        n_slices = volume.shape[axis_index]

        vertices = np.array([]).reshape((0, 4))
        centroids = np.array([]).reshape((0, 5))

        for i in tqdm(range(n_slices), desc=f'Finding cross-sections along {axis}'):
            if axis == 'x':
                current_slice = volume[i, :, :]
            elif axis == 'y':
                current_slice = volume[:, i, :]
            else:
                current_slice = volume[:, :, i]

            slice_labeled, n_slice_labels = self.find_cross_sections_in_slice(current_slice)
            for slice_label in range(1, n_slice_labels + 1):
                cross_section_index = self.n_labels + slice_label
                cross_section = np.argwhere(slice_labeled == slice_label)

                n_vertices = len(cross_section)

                if min_points is not None:
                    if n_vertices < min_points:
                        continue
                if max_points is not None:
                    if n_vertices > max_points:
                        continue

                try:
                    hull = ConvexHull(cross_section)

                    # The area and perimeter (length) of the convex hull
                    area = hull.volume
                    perimeter = hull.area

                    if max_area is not None:
                        if area > max_area:
                            continue

                    if max_perimeter is not None:
                        if perimeter > max_perimeter:
                            continue
                except:
                    continue  # Element is probably linear

                cross_section_vertices = np.array([]).reshape((0, 4))

                for vertex in cross_section:
                    if axis == 'x':
                        cross_section_vertex = np.array([cross_section_index, i, vertex[0], vertex[1]])
                    elif axis == 'y':
                        cross_section_vertex = np.array([cross_section_index, vertex[0], i, vertex[1]])
                    else:
                        cross_section_vertex = np.array([cross_section_index, vertex[0], vertex[1], i])
                    cross_section_vertices = np.vstack((cross_section_vertices, cross_section_vertex))

                vertices = np.vstack((vertices, cross_section_vertices))

                if axis == 'x':
                    _, _, y_sum, z_sum = cross_section_vertices.sum(axis=0)
                    cross_section_centroid = np.array([cross_section_index,
                                                       i, y_sum / n_vertices, z_sum / n_vertices, n_vertices])
                elif axis == 'y':
                    _, x_sum, _, z_sum = cross_section_vertices.sum(axis=0)
                    cross_section_centroid = np.array([cross_section_index,
                                                       int(x_sum / n_vertices), i, int(z_sum / n_vertices), n_vertices])
                else:
                    _, x_sum, y_sum, _ = cross_section_vertices.sum(axis=0)
                    cross_section_centroid = np.array([cross_section_index,
                                                       int(x_sum / n_vertices), int(y_sum / n_vertices), i, n_vertices])

                centroids = np.vstack((centroids, cross_section_centroid))

            self.n_labels = self.n_labels + n_slice_labels

        return centroids, vertices

    @staticmethod
    def find_cross_sections_in_slice(slice_):
        # TODO: Eliminate compute_centroids and compute_vertices by combining functionality together in this function
        # Define a structuring element that includes diagonal connections
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]

        slice_labeled, n_slice_labels = label(slice_, structure=s)

        return slice_labeled, n_slice_labels

    def add_centroids_to_point_cloud(self, centroids, min_coord):
        point_cross_sections = centroids[:, 0:1]
        point_positions = centroids[:, 1:4] * self.resolution + min_coord  # scales and translates vertex coordinated
        point_n_vertices = centroids[:, 4:5]

        point_cloud = o3d.t.geometry.PointCloud()
        point_cloud.point["cross_sections"] = o3d.core.Tensor(point_cross_sections.astype(np.uint8))
        point_cloud.point["positions"] = o3d.core.Tensor(point_positions.astype(np.float64))
        point_cloud.point["colors"] = o3d.core.Tensor(np.repeat([[0.0, 0.0, 0.0]], len(point_cross_sections), axis=0))
        point_cloud.point["n_vertices"] = o3d.core.Tensor(point_n_vertices.astype(np.uint8))

        return point_cloud

    def add_vertices_to_point_cloud(self, vertices, min_coord):
        point_cross_sections = vertices[:, 0:1]
        point_positions = vertices[:, 1:4] * self.resolution + min_coord  # scales and translates vertex coordinated

        point_cloud = o3d.t.geometry.PointCloud()
        point_cloud.point["cross_sections"] = o3d.core.Tensor(point_cross_sections.astype(np.uint8))
        point_cloud.point["positions"] = o3d.core.Tensor(point_positions.astype(np.float64))
        point_cloud.point["colors"] = o3d.core.Tensor(np.repeat([[0.0, 0.0, 0.0]], len(point_cross_sections), axis=0))

        return point_cloud


class CrossSections:
    def __init__(self):
        self.source_mesh = None
        self.centroids = None
        self.vertices = None
        self.result_mesh = None

    def find(self, source_mesh=None, resolution=0.003, thickness=0.003, n_samples=1, min_points=10, max_points=250,
             max_perimeter=None, max_area=None):
        """
        Find cross-sections in a source_mesh. Includes parameters for initial filtering to reduce computation time.
        Cross-sections found using this method are stored in two point clouds:

        self.centroids: an open3d.t.geometry.point_cloud object which stores the centroids of each cross-section.
        Centroids have positions (3d), normals (oriented perpendicular to the cross-section plane), sizes (number
        of points in the cross-section), perimeters (perimeter of 2d convex hull placed on cross-section), areas
        (area of 2d convex hull placed on cross-section).

        self.vertices: an open3d.t.geometry.point_cloud object which stores the cross-section points/vertices.
        Vertices have positions (3d), normals (reflecting the average normal of points used to generate the
        voxel grid for finding cross-sections), colors (reflecting the average color of the points used to
        generate the voxel grid for finding cross-sections), and centroids which refers to the index of the
        corresponding centroid in self.centroids.

        :param source_mesh: mesh in which to find cross-sections
        :param resolution: resolution of voxel_grid used to find cross-sections
        :param thickness: cross-section thickness, must be a multiple of resolution. Default is equal to resolution.
        :param n_samples: number of voxel_grids to find cross-sections. For samples > 1,
        each voxel grid will be rotated to optimize voxel_grid slicing direction. Each voxel grid will be sliced
        along each axis for a total of 3 slicing directions per sample. A larger number of samples will result
        in higher accuracy for cross-section detection, but will take more resources to compute.
        :param min_points: Minimum number of points allowed for a cross-section
        :param max_points: Maximum number of points allowed for a cross-section
        :param max_perimeter: Maximum perimeter allowed for a cross-section
        :param max_area: Maximum area allowed for a cross-section.
        :return: None
        """
        print(f"Finding cross-sections in mesh")
        if source_mesh is not None:
            self.source_mesh = source_mesh

        rot_matrices = find_optimal_rotation_matrices(n_samples=n_samples)

        centroids = []
        vertices = []

        for idx, rot in enumerate(rot_matrices):
            print(f"Finding cross-sections for sample {idx + 1}/{n_samples}")
            mesh_volume = MeshVolume(mesh=self.source_mesh, rot=rot, resolution=resolution)
            centroids_pcd, vertices_pcd = mesh_volume.find_cross_sections(thickness=thickness,
                                                                          min_points=min_points,
                                                                          max_points=max_points,
                                                                          max_perimeter=max_perimeter,
                                                                          max_area=max_area)
            centroids.append(centroids_pcd)
            vertices.append(vertices_pcd)

        self.centroids = merge_point_clouds_from_list(centroids)
        self.vertices = merge_point_clouds_from_list(vertices)

        return self.centroids, self.vertices

    def rank(self, resolution=0.003):
        """
        Cross-section ranking method. Idea is to identify the best-fit cross-sections at a particular location
        by grouping nearby cross-sections together (by centroid) and ranking these cross-sections
        according to attribute(s) which are associated with best-fit. Cross-sections which fit the best generally
        minimize the area, and the normals of the points in the cross-section have a consistent angle
        to the cross-section plane.

        This method adds a rank attribute to points in self.centroids, where rank of 0 corresponding to best fit,
        1 is second-best, 2, and so on.

        :param resolution: resolution of voxel_grid in which to group cross-sections by centroids
        :return: None
        """
        print(f"Ranking cross-section centroids by size, grouping using resolution={resolution}")
        centroids_ranker = Ranker(self.centroids)
        self.centroids.point["n_vertices_rank"] = centroids_ranker.group_points_and_rank(numeric_attribute="n_vertices",
                                                                                         voxel_size=resolution,
                                                                                         rank_order='ascend')

        # Min size_rank should be 0
        #print(self.centroids.point["size_ranks"].min())
        # Max size_rank is unbounded
        #print(self.centroids.point["size_ranks"].max())

    def filter(self, max_rank=0,
               min_points=None, max_points=None,
               max_perimeter=None,
               max_area=None):
        """
        Cross-section filtering method. Filtering is primarily based on rank, but can also be performed
        for other attributes of centroids.

        Filtering trims the self.centroids point cloud and removes points in the self.vertices point cloud
        which no longer point to a valid centroid.

        :param max_rank: Maximum rank
        :param min_points: Minimum number of points allowed for a cross-section
        :param max_points: Maximum number of points allowed for a cross-section
        :param max_perimeter: Maximum perimeter allowed for a cross-section
        :param max_area: Maximum area allowed for a cross-section.
        :return: None
        """
        print(f"Filtering cross-sections using rank={max_rank}")
        centroids_filter = Filter(point_cloud=self.centroids)
        centroids_filter.filter_by_n_vertices_rank(rank_val=max_rank)
        self.centroids = centroids_filter.point_cloud

    def remove_intersecting(self, resolution=0.003, retain=0.95):
        """
        Remove cross-sections until no cross-section intersects another
        This method is similar to the filter method in that additional cross-sections are removed,
        and is similar to the rank method in that a resolution parameter uses a voxel grid to determine
        if the vertices of multiple cross-sections are occupying the same space.

        If there are multiple vertices occupying the same space, then the best-fit cross-section should be
        retained. Similar to the rank method which determines best-fit on neighboring centroids using angle
        to cross-section plane, a best-fit could be computed for neighboring vertices. I'm not yet sure how
        this can be done because it would be complicated. Alternatively, we could just remove cross-sections
        at random until a maximum of 1 cross-section vertex occupies a voxel in the voxel-grid, however
        removing cross-sections at random could cause several voxels that previously had vertices in them
        to be removed entirely. In an ideal world, all occupied voxels at the start would have exactly
        1 vertex occupying each voxel after removing intersecting cross-sections, however it's reasonable
        to expect some amount to be dropped. The retain parameter is added to specify the desired fraction
        of voxels containing at least one cross-section vertex after all intersecting cross-sections are
        removed. Perhaps the first cross-sections to be removed are ones which do not result in any dropped
        voxels. Then pairs of cross-sections are evaluated such that the removed cross-section results in
        minimal dropped voxels.

        This method trims self.centroids and self.vertices similar to the self.filter method.

        :return: None
        """

    def connect(self):
        """
        After finding, ranking, filtering, and removing intersecting cross-sections, cross-sections can be connected together
        to form a network, representing the surface of the source_mesh in a structured manner.

        This method creates a networkx network object where nodes reflect cross-sections (centroids and vertices)
        and edges reflecting the connections between neighboring cross-sections. Connections between nearby points
        are determined by finding nodes with close-by location, similar angle, and similar area.

        The networkx object should really only store cross-section indices as nodes and the indices of
        connected cross-sections (connected nodes) as edges. At this time, the network should not store any other
        properties.

        ????Is this necessary? Edge data should also be stored as a property of points in self.centroids using a property
        called "neighbor". If [0, 1] and [0, 2] are two edges, then centroid with index 0 would have a
        neighbors list of [1, 2].
        :return: None
        """
        pass

    def to_mesh(self):
        """
        At long last, the to_mesh method creates a self.result_mesh object. It connects the vertices within
        cross-section to create edge loops, and connects the vertices between neighboring cross-sections.
        :return: None
        """

    def separate_groups(self):
        """
        :return:
        """
        pass

    def label(self, leaf_size=40, min_cluster_size=200, min_samples=10, gen_min_span_tree=True):
        """
        Adds labels and colors to centroids/vertices point clouds
        :return: None
        """
        print("Labeling cross-sections and adding colors to point cloud")
        points = self.centroids.point["positions"].numpy()

        clusterer = hdbscan.HDBSCAN(algorithm='best',
                                    cluster_selection_method='leaf',
                                    leaf_size=leaf_size,
                                    min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    gen_min_span_tree=gen_min_span_tree)

        labels = clusterer.fit_predict(points)
        self.centroids.point["labels"] = o3d.core.Tensor(np.asarray(labels))

        max_label = labels.max()
        print(f"Point cloud has {max_label + 1} labeled cross-sections")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        self.centroids.point["colors"] = o3d.core.Tensor(colors[:, :3])

    def view_centroids(self):
        """
        View cross-section centroids as an open3d point cloud. Be sure to label cross-sections first.
        :return:
        """
        o3d.visualization.draw_geometries([self.centroids.to_legacy()])

    def view_vertices(self):
        o3d.visualization.draw_geometries([self.vertices.to_legacy()])

    def print_stats(self):
        print("Cross-section stats:")
        print(f"Centroids: {self.centroids.point.positions.shape[0]} points")
        print(f"Vertices: {self.vertices.point.positions.shape[0]} points")

    def read(self, cs_dir):
        """
        Read cross-section geometry objects from cs_dir. See self.write for
        :param cs_dir:
        :return:
        """
        pass

    def write(self, cs_dir, write_write_mesh=False):
        pass


class Ranker:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud

    def group_points_and_rank(self, numeric_attribute, voxel_size, rank_order):
        """
        Group points by voxel size and rank them based on a numeric attribute.

        Parameters:
        point_cloud (o3d.geometry.PointCloud): The original point cloud.
        numeric_attribute (np.ndarray): The custom numeric attribute corresponding to each point.
        voxel_size (float): The size of the voxel grid.
        rank_order (str): The order for ranking ('ascend' or 'descend').

        Returns:
        np.ndarray: An array containing the rank for each point in the original point cloud.
        """
        numeric_attribute = self.point_cloud.point[f"{numeric_attribute}"].numpy()

        # Convert point cloud to numpy array
        points = self.point_cloud.point["positions"].numpy()

        # Round points to the nearest voxel_size
        rounded_points = np.round(points / voxel_size) * voxel_size

        # print(rounded_points)

        # Create a dictionary to store groups of points
        groups = {}

        for i, point in enumerate(rounded_points):
            key = tuple(point)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        # print(groups)

        # Initialize rank array
        rank_array = np.zeros(len(points), dtype=int)

        # Sort and rank points in each group
        for group in groups.values():
            sorted_indices = np.argsort(numeric_attribute[group])

            if rank_order == 'descend':
                sorted_indices = sorted_indices[::-1]

            for rank, index in enumerate(sorted_indices):
                original_index = group[index[0]]
                rank_array[original_index] = rank + 1

            # print(rank_array)

        return o3d.core.Tensor(rank_array.reshape((-1, 1)))


class Filter:
    def __init__(self, point_cloud):
        """
        Cross-section filter class. Designed to be used when finding cross-sections as it prevents
        unneccesary computation and for filtering cross-sections after generation.
        :param point_cloud: Point cloud object to perform filtering on
        """
        self.point_cloud = point_cloud

    def apply_mask(self, mask):
        filtered_point_cloud = o3d.t.geometry.PointCloud(device=self.point_cloud.device)

        filtered_point_cloud.point["positions"] = self.point_cloud.point["positions"][mask].reshape((-1, 3))
        #filtered_point_cloud.point["normals"] = self.point_cloud.point["normals"][mask].reshape((-1, 3))
        filtered_point_cloud.point["colors"] = self.point_cloud.point["colors"][mask].reshape((-1, 3))

        filtered_point_cloud.point["labels"] = self.point_cloud.point["labels"][mask]
        filtered_point_cloud.point["n_vertices_rank"] = self.point_cloud.point["n_vertices_rank"][mask].reshape((-1, 1))
        filtered_point_cloud.point["n_vertices"] = self.point_cloud.point["n_vertices"][mask].reshape((-1, 1))

        return filtered_point_cloud

    def filter_by_n_vertices_rank(self, rank_val=0):
        """
        Filters point cloud by size_rank
        :param rank_val: integer of rank value points in filtered point cloud should have
        :return:
        """
        mask = o3d.core.Tensor(np.argwhere(self.point_cloud.point["n_vertices_rank"].numpy() == rank_val))
        filtered_point_cloud = self.apply_mask(mask)
        self.point_cloud = filtered_point_cloud


class FieldObj:
    def __init__(self, photo_mesh: o3d.geometry.TriangleMesh = None) -> None:
        """
        Class for representing a field object (plot, plant within plot,
        leaf within plant, stake within plot, etc.) using multiple types of geometry formats
        :param photo_mesh: Triangle mesh produced from photogrammetry
        """
        self.photo_mesh = photo_mesh
        self.cross_sections = CrossSections()

    def segment_mesh(self):
        self.cross_sections.find(source_mesh=self.photo_mesh, resolution=0.003, thickness=0.003,
                                 n_samples=1, min_points=10, max_points=100, max_perimeter=None, max_area=None)
        self.cross_sections.label(leaf_size=40, min_cluster_size=200, min_samples=10, gen_min_span_tree=True)
        self.cross_sections.rank(resolution=0.003)
        self.cross_sections.print_stats()
        self.view_segments()
        self.cross_sections.filter(max_rank=0)
        self.cross_sections.print_stats()
        self.view_segments()
        self.cross_sections.view_vertices()
        #self.cross_sections.filter()

    def view_segments(self):
        self.cross_sections.view_centroids()

    def view_mesh(self):
        o3d.visualization.draw_geometries([self.photo_mesh])


def mesh_to_voxel_grid(mesh, voxel_size=0.001):
    # Convert triangle mesh to voxel model
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    return voxel_grid


def voxel_grid_to_volume(voxel_grid):
    # Convert voxel grid to volume (3D binary image)
    # Get all voxels in the voxel grid
    voxels = [v.grid_index for v in voxel_grid.get_voxels()]

    # Find the dimensions of the voxel grid
    max_dim = np.max(voxels, axis=0) + 1  # Adding 1 to make sure the max index is included

    # Create an empty 3D binary image
    volume = np.zeros(max_dim, dtype=bool)

    # Mark the voxels in the binary image
    for v in voxels:
        volume[v[0], v[1], v[2]] = True

    return volume


def voxel_grid_to_point_cloud(voxel_grid):
    # Get voxel size and origin
    voxel_size = voxel_grid.resolution
    origin = voxel_grid.origin

    # Convert VoxelGrid to PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        [voxel.grid_index * voxel_size + origin for voxel in voxel_grid.get_voxels()])

    # Optionally, you could also assign colors based on voxel colors
    point_cloud.colors = o3d.utility.Vector3dVector([voxel.color for voxel in voxel_grid.get_voxels()])

    return point_cloud


def point_cloud_to_voxel_grid(point_cloud, voxel_size):
    # Convert the point cloud to a voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)

    return voxel_grid


def find_optimal_rotation_matrices(n_samples):
    """
    Finds optimal rotation matrices for the number of samples
    :return: rot_matrices
    """
    # TODO: Fix so rotation matrices are more uniform
    rot_matrices = []

    min_angle = 0
    max_angle = 90

    current_angle = min_angle
    angle_amt = 90/n_samples

    while current_angle < max_angle:
        rot_matrices.append(generate_rotation_matrix(current_angle, current_angle))
        current_angle += angle_amt

    return rot_matrices


def generate_rotation_matrix(x_angle_deg, y_angle_deg):
    # Convert angles to radians
    x_angle_rad = np.radians(x_angle_deg)
    y_angle_rad = np.radians(y_angle_deg)

    # Generate X-axis rotation matrix
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle_rad), -np.sin(x_angle_rad)],
        [0, np.sin(x_angle_rad), np.cos(x_angle_rad)]
    ])

    # Generate Y-axis rotation matrix
    Ry = np.array([
        [np.cos(y_angle_rad), 0, np.sin(y_angle_rad)],
        [0, 1, 0],
        [-np.sin(y_angle_rad), 0, np.cos(y_angle_rad)]
    ])

    # Combine the rotations
    R = np.dot(Ry, Rx)

    return R


def merge_point_clouds_from_list(point_cloud_list):
    merged_point_cloud = None
    for pcd in point_cloud_list:
        if merged_point_cloud is None:
            merged_point_cloud = pcd
        else:
            merged_point_cloud += pcd

    return merged_point_cloud