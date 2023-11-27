import os
import open3d as o3d



class MeshImporter:
    """
    This importer is designed to be a generalized mesh importer, though there are some requirements:
    1. The mesh is in a directory of the same name (example: mymesh/mymesh.ply)
    2. The mesh is saved with extension .ply (Plan to add support for more formats later)
    3. The mesh directory can have one or more RGB textures with extension .png.
    If there are more than 1 texture images mapped to the object, then these should be ordered sequentially
    (Plan to add support for .tif and other formats, as well as other types of maps (normal, specularity, etc.))
    4. The mesh directory can have metadata stored in a .xml file
    """
    def __init__(self, mesh_path=None, rgb_texture_dir=None, metadata_dir=None):
        self.mesh_path = mesh_path
        self.mesh_dir, self.mesh_file = os.path.split(self.mesh_path)
        self.mesh_name, _ = os.path.splitext(self.mesh_file)

        self.rgb_texture_dir = self.mesh_dir if rgb_texture_dir is None else rgb_texture_dir
        self.rgb_texture_files = self.find_textures(texture_dir=self.rgb_texture_dir)


        self.metadata_dir = self.mesh_dir if metadata_dir is None else metadata_dir
        self.metadata_file = self.find_metadata(metadata_dir=self.metadata_dir)

        self.mesh: o3d.geometry.TriangleMesh() = self._import_mesh()

    @staticmethod
    def import_mesh(mesh_path=None, rgb_texture_dir=None, metadata_dir=None):
        importer = MeshImporter(mesh_path=mesh_path, rgb_texture_dir=rgb_texture_dir,
                                metadata_dir=metadata_dir)
        return importer.mesh

    def _import_mesh(self):
        mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(filename=self.mesh_path, print_progress=False)

        if mesh.has_triangle_uvs() is False:
            # Add these from file
            pass

        if mesh.has_triangle_material_ids() is False:
            # Add these too? I'm not sure
            pass

        if mesh.has_vertex_normals() is False:
            mesh.compute_vertex_normals()

        if mesh.has_triangle_normals() is False:
            mesh.compute_triangle_normals()

        if self.rgb_texture_files is not None:
            # See http://www.open3d.org/docs/release/python_example/visualization/index.html#textured-model-py
            pass

        if self.metadata_file is not None:
            pass


        # TODO: Normals are ignored when non-decimated mesh from Metashape is imported
        return mesh


    def find_textures(self, texture_dir=None, prefix="mesh_name", sep=".", first_texture_number=1001, ext=".png"):
        """
        Finds texture files and appends them to self.texture_files.
        Assumes textures have a similar name as the ply file
        Using the default parameters, the texture file name of the first texture
        will look like this: ""
        :param texture_dir: If None, uses the same directory as the ply file
        :param prefix: If None, uses the ply file name as the prefix for texture file name
        :param sep: Separates the texture file name from the texture number
        :param first_texture_number: The number of the first texture file. If mesh only has 1 texture file.
        :param ext: The file extension of the texture file(s)
        :return:
        """
        prefix = self.mesh_name if prefix is "mesh_name" else ""

        texture_number = first_texture_number
        texture_files = None

        while os.path.exists(os.path.join(self.rgb_texture_dir, f'{prefix}{sep}{texture_number}.{ext}')):
            texture_files = [] if texture_files is None else texture_files
            texture_files.append(f'{self.mesh_name}.{texture_number}.{ext}')
            texture_number += 1

        return texture_files

    def find_metadata(self, metadata_dir=None):
        """
        Metashape offers the option to save a metadata.xml file
        :return:
        """
        metadata_dir = self.mesh_dir if metadata_dir is None else metadata_dir

        metadata_path = os.path.join(metadata_dir, "metadata.xml")
        if os.path.exists(metadata_path):
            return metadata_path
        return None
