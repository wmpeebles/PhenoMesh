import numpy as np
import open3d as o3d
import copy
from tqdm import tqdm
import cv2

class Slicer:
    def __init__(self, mesh, bbox, point, normals, resolution):
        self.mesh = mesh
        self.bbox = bbox
        self.point = point
        self.normals = normals
        self.resolution = resolution

    def _prepare_for_slicing(self):
        pass

    def slice_plane_o3d(self):
        results = []
        #for normal in tqdm(self.normals, "Slicing planes from mesh"):
        for normal in self.normals:
            point_positions, point_normals, point_colors, line_indices = _slice_plane_o3d(self.mesh, self.bbox,
                                                                                          self.point, normal,
                                                                                          self.resolution)
            results.append((point_positions, point_normals, point_colors, line_indices))
        return results

    def slice_plane_voxel_approximation(self):
        results = []
        for normal in tqdm(self.normals, "Slicing planes from mesh"):
            point_positions, point_normals, point_colors, line_indices = _slice_plane_voxel_approximation(copy.deepcopy(self.mesh),
                                                                                                          copy.deepcopy(self.bbox),
                                                                                                          normal,
                                                                                                          self.resolution)
            results.append((point_positions, point_normals, point_colors, line_indices))
        return results


def _slice_plane_o3d(mesh, bbox, point, normal, resolution):
    """
    Slices planes from mesh using built-in o3d method, however this is slow and does not seem to be compatible
    with multiprocessing
    :param mesh:
    :param bbox:
    :param point:
    :param normal:
    :param resolution:
    :return:
    """
    rot = _get_rotation_matrix_from_normal(normal)
    rotated_bbox = bbox.rotate(R=rot, center=bbox.center)
    rotated_bbox_box_points = np.asarray(rotated_bbox.get_box_points())

    min_z = np.min(rotated_bbox_box_points[:, 2])
    max_z = np.max(rotated_bbox_box_points[:, 2])
    contour_values = np.arange(start=min_z, stop=max_z, step=resolution) - point[2]

    t_mesh = o3d.t.geometry.TriangleMesh().from_legacy(mesh).clone()
    line_set = t_mesh.slice_plane(point=point, normal=normal,
                                  contour_values=contour_values)  # .rotate(R=o3d.core.Tensor(rot), center=point)

    point_positions = line_set.point.positions.numpy()  # Shape (N, 3)
    point_normals = line_set.point.normals.numpy()  # Shape (N, 3)
    point_colors = line_set.point.colors.numpy()  # Shape (N, 3)
    line_indices = line_set.line.indices.numpy()  # Shape (N, 2)

    return point_positions, point_normals, point_colors, line_indices


def _slice_plane_voxel_approximation(mesh, bbox, normal, resolution):
    """
    Slices planes from mesh using built-in o3d method, however this is slow and does not seem to be compatible
    with multiprocessing
    :param mesh:
    :param bbox:
    :param point:
    :param normal:
    :param resolution:
    :return:
    """
    rot = _get_rotation_matrix_from_normal(normal)
    rotated_mesh = copy.deepcopy(mesh).rotate(R=rot, center=bbox.center)
    #rotated_bbox = rotated_mesh.get_oriented_bounding_box()
    #min_coord_orig = np.min(rotated_bbox.get_box_points(), axis=0)
    min_coord_orig = np.min(np.asarray(rotated_mesh.vertices), axis=0)
    min_coord = np.asarray((min_coord_orig[1], min_coord_orig[0], min_coord_orig[2]-resolution/2))
    #print(min_coord)

    voxel_grid = mesh_to_voxel_grid(rotated_mesh, voxel_size=resolution)
    volume = voxel_grid_to_volume(voxel_grid)

    line_set = None

    # Slice the volume only along the z axis
    n_slices = volume.shape[2]
    #for i in tqdm(range(n_slices), "Slicing planes using voxel approximation of mesh"):
    for i in range(n_slices):
        z_slice = volume[:, :, i]
        points, lines = _connect_points(z_slice, i)
        if len(points) < 3:
            continue
        s_line_set = o3d.geometry.LineSet()
        s_line_set.points = o3d.utility.Vector3dVector(points)
        s_line_set.lines = o3d.utility.Vector2iVector(lines)
        # Scale and transform points here
        old_points = np.asarray(s_line_set.points)
        #print(old_points)
        new_points = old_points * resolution + min_coord
        #print(new_points)
        #x = np.copy(new_points[:, 1:2])
        #y = np.copy(new_points[:, 0:1])
        #new_points[:, 0:1] = x
        #new_points[:, 1:2] = y
        new_points = np.column_stack((new_points[:, 1:2], new_points[:, 0:1], new_points[:, 2:3]))
        #print(new_points)
        s_line_set.points = o3d.utility.Vector3dVector(new_points)
        s_line_set = s_line_set.rotate(R=np.transpose(rot), center=bbox.center)
        #print(rot)
        # quit()

        #print(np.asarray(s_line_set.points))
        #quit()

        if line_set is None:
            line_set = s_line_set
        else:
            line_set += s_line_set

    point_positions = np.asarray(line_set.points)  # Shape (N, 3)
    point_colors, point_normals = _add_colors_and_normals(mesh, line_set)
    line_indices = np.asarray(line_set.lines)  # Shape (N, 2)

    #print(point_positions.shape, point_normals.shape, point_colors.shape, line_indices.shape)

    return point_positions, point_normals, point_colors, line_indices


def _add_colors_and_normals(mesh, line_set):
    # Check if mesh has vertex normals and colors
    if not mesh.has_vertex_normals() or not mesh.has_vertex_colors():
        raise ValueError("Mesh must have vertex normals and colors")

    # Create a KDTree for the mesh vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Initialize arrays for colors and normals
    colors = np.zeros_like(line_set.points)
    normals = np.zeros_like(line_set.points)

    # Iterate over each point in the line_set
    for i, point in enumerate(line_set.points):
        # Find the nearest vertex in the mesh
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)

        # Copy the color and normal from the nearest mesh vertex
        colors[i] = mesh.vertex_colors[idx[0]]
        normals[i] = mesh.vertex_normals[idx[0]]

    return colors, normals


def _connect_points(slice, z):
    # Find contours for each label
    contours, _ = cv2.findContours(slice.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lists to store lineset data
    points = []
    lines = []

    # Process each contour
    point_idx = 0
    for contour in contours:
        if len(contour) < 3:
            # Ignore contours with fewer than 3 points
            continue
        # TODO: implement less of a hacky solution than this, perhaps use max_perimeter to filter prior to loop creation
        if len(contour) > 50:
            continue
        # Approximate contour to polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Add points and lines
        for pt in approx:
            points.append([pt[0][0], pt[0][1], z])  # Z-coordinate is 0 for 2D slice

        for i in range(len(approx)):
            start_point = point_idx + i
            end_point = point_idx + (i + 1) % len(approx)
            lines.append([start_point, end_point])

        point_idx += len(approx)

    return points, lines


def _get_rotation_matrix_from_normal(normal):
    rotation_axis = np.cross([0, 0, 1], normal)
    rotation_angle = np.arccos(np.dot([0, 0, 1], normal))
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    return rot


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
