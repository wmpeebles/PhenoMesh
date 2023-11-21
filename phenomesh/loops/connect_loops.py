import open3d as o3d
import bpy
import numpy as np


def connect_loops_blender(line_set, type='CLOSED', interpolation='PATH', number_cuts=5, smoothness=2):
    vertices = np.asarray(line_set.points)
    edges = np.asarray(line_set.lines)

    # Clear existing objects in the scene (optional)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Create a new mesh and a new object
    mesh_data = bpy.data.meshes.new('edge_loops_mesh')
    object_data = bpy.data.objects.new('My_Object', mesh_data)

    # Link the object to the scene
    bpy.context.collection.objects.link(object_data)

    # Change the mode to 'EDIT' to modify the mesh
    bpy.context.view_layer.objects.active = object_data
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.object.mode_set(mode='OBJECT')
    # Add vertices and edges to the mesh
    mesh_data.from_pydata(vertices, edges, [])

    # Update mesh with new data
    mesh_data.update()

    # Change to Object Mode to use bpy.ops.mesh.bridge_edge_loops
    bpy.ops.object.mode_set(mode='OBJECT')

    # Select all edges to bridge edge loops
    bpy.ops.object.select_all(action='SELECT')

    # Switch back to Edit Mode to perform the bridge operation
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Bridge the edge loops
    bpy.ops.mesh.bridge_edge_loops(type=type, interpolation=interpolation, number_cuts=number_cuts, smoothness=smoothness)

    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    verts = np.array([vert.co for vert in mesh_data.vertices])
    faces = np.array([face.vertices[:] for face in mesh_data.polygons])

    #print(verts.shape)
    #print(np.asarray(line_set.colors).shape)
    #print(faces.shape)

    mesh = o3d.geometry.TetraMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.tetras = o3d.utility.Vector4iVector(faces)

    return mesh


def add_colors_and_normals(orig_mesh, tetra_mesh):
    #print(orig_mesh)
    #print(tetra_mesh)
    # Check if mesh has vertex normals and colors
    if not orig_mesh.has_vertex_normals() or not orig_mesh.has_vertex_colors():
        raise ValueError("Mesh must have vertex normals and colors")

    # Create a KDTree for the mesh vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = orig_mesh.vertices
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Initialize arrays for colors and normals
    colors = np.zeros_like(tetra_mesh.vertices)
    normals = np.zeros_like(tetra_mesh.vertices)

    # Iterate over each point in the line_set
    for i, point in enumerate(tetra_mesh.vertices):
        # Find the nearest vertex in the mesh
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)

        # Copy the color and normal from the nearest mesh vertex
        colors[i] = orig_mesh.vertex_colors[idx[0]]
        normals[i] = orig_mesh.vertex_normals[idx[0]]

    tetra_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    tetra_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    #print(tetra_mesh.has_vertex_colors())
    #print(tetra_mesh.has_vertex_normals())

    #print(tetra_mesh)

    return tetra_mesh


if __name__ == "__main__":
    # Create two edge loops as example
    vertices = [(1, 1, 0), (1, -1, 0), (-1, -1, 0), (-1, 1, 0), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    o3d.visualization.draw_geometries([line_set])

    mesh = connect_loops_blender(line_set)

    print(mesh)

    o3d.visualization.draw_geometries([mesh])