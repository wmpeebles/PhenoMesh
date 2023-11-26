from .mesh_importer import MeshImporter


def import_mesh(mesh_path):
    importer = MeshImporter(mesh_path=mesh_path)
    importer.import_mesh()
    return importer.mesh
