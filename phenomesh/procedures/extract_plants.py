"""
Goal here is to extract plants from a mesh that includes non-plant objects, such as soil and stakes
"""
import copy

import open3d as o3d
import numpy as np


class GeometryFilter:
    def __init__(self, mesh: o3d.geometry.TriangleMesh(), max_angle=-0.75, within_radius=0.045):
        f"""
        This class attempts to find plants by using the geometric properties of the mesh to remove
        non-plant objects, such as soil, that do not form a curved surface. The completeness of the curve
        is defined by min_angle and the size is defined by within_radius. The intent of the function
        is as follows:
        
        In mesh {mesh}, remove all points if none of the surrounding points within {within_radius} units
        form a angle greater than {max_angle} units.
        
        This function should work with incomplete curved surfaces. For example, if only 75% of the 
        perimeter of a circle is present, setting max_angle to -0.75 should allow this perimeter to be
        detected. This is useful in cases where occlusion exists, however an angle closer to 0 can cause
        increasingly planar surfaces to not be filtered out.
        
        :param mesh: 
        :param max_angle: This value should be between [-1,0], corresponding to [-180°,0°], but
        can be between [-1, 1] if desired. Angle here is defined as the 
        dot product between the normals of two points in the mesh. 
        :param within_radius: 
        """
        self.mesh = copy.deepcopy(mesh)
        self.max_angle = max_angle
        self.within_radius = within_radius

    def filter_mesh(self):
        positions = np.asarray(self.mesh.vertices)
        kdtree = self._get_kdtree(self.mesh)
        point_angles = self._find_angles(kdtree, positions)
        mesh = self._remove_invalid_points(self.mesh, point_angles, self.max_angle)
        return mesh

    def _get_kdtree(self, mesh):
        pcd = self._mesh_to_pcd(mesh)
        return o3d.geometry.KDTreeFlann(pcd)

    @staticmethod
    def _mesh_to_pcd(mesh):
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.normals = mesh.vertex_normals
        pcd.colors = mesh.vertex_colors
        return pcd

    def _find_angles(self, kdtree, vertices):
        point_angles = np.zeros(shape=(vertices.shape[0],))
        for idx, point in enumerate(vertices):
            [k, idxs, d] = kdtree.search_radius_vector_3d(point, self.within_radius)
            min_point_angle = self._get_min_normal_differences(idx, idxs)
            point_angles[idx] = min_point_angle
        return point_angles

    def _get_min_normal_differences(self, idx, idxs: o3d.utility.IntVector()):
        idxs = np.asarray(idxs)
        all_normals = np.asarray(self.mesh.vertex_normals)
        normal = all_normals[idx]
        normals = all_normals[idxs]
        angles = self._calculate_angle_differences(normal, normals)
        min_angle = np.min(angles)
        return min_angle

    @staticmethod
    def _calculate_angle_differences(normal, normals):
        if normals.shape[0] == 0:
            return 2  # code for invalid angle
        return np.dot(normal, np.transpose(normals))

    @staticmethod
    def _remove_invalid_points(mesh, point_angles, max_angle):
        invalid_indices = np.argwhere(point_angles > max_angle)
        # TODO: Add ability to keep
        mesh.remove_vertices_by_index(invalid_indices)
        return mesh
