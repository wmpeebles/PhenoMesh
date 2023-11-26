import os
import sys
import open3d as o3d
import pytest

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from phenomesh.io import import_mesh, mesh_importer
from phenomesh.data import mesh_paths, get_mesh

for mesh_name, mesh_path in mesh_paths.items():
    def test_mesh_path_exists():
        assert os.path.exists(mesh_path) is True, f"mesh_path: {mesh_path}"

    #importer = mesh_importer.MeshImporter(mesh_path=mesh_path)
    #importer.import_mesh()

    mesh = import_mesh(mesh_path)

    def test_mesh_imported_successfully():
        assert isinstance(mesh, o3d.geometry.TriangleMesh)

    def test_mesh_has_textures():
        assert mesh.has_textures() is True

    def test_mesh_has_vertices():
        assert mesh.has_vertices() is True

    def test_mesh_has_triangles():
        assert mesh.has_triangles() is True

    def test_mesh_has_triangle_uvs():
        assert mesh.has_triangle_uvs() is True

    def test_mesh_has_vertex_colors():
        assert mesh.has_vertex_colors() is True

    def test_mesh_has_adjacency_list():
        assert mesh.has_adjacency_list() is True

    def test_mesh_has_triangle_material_ids():
        assert mesh.has_triangle_material_ids() is True

    def test_mesh_has_triangle_normals():
        assert mesh.has_triangle_normals() is True

    def test_mesh_has_vertex_normals():
        assert mesh.has_vertex_normals() is True

    def test_mesh_is_not_empty():
        assert mesh.is_empty() is False

    def test_mesh_is_not_intersecting():
        assert mesh.is_intersecting() is False

    def test_mesh_is_orientable():
        assert mesh.is_orientable() is True

    def test_mesh_is_watertight():
        assert mesh.is_watertight() is True or False

    def test_mesh_is_edge_manifold():
        assert mesh.is_edge_manifold() is True

    #def test_mesh_is_not_self_intersecting():
    #    # This test runs a while
    #    assert mesh.is_self_intersecting() is False

    def test_mesh_is_vertex_manifold():
        assert mesh.is_vertex_manifold() is True
