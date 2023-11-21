import copy
import numpy as np
import open3d as o3d
import networkx as nx
from tqdm import tqdm
from numba import njit
from sklearn.decomposition import PCA
import pickle

from .mesh_slicing import Slicer
from .connect_loops import connect_loops_blender, add_colors_and_normals


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class Point:
    def __init__(self, position: {}, normal=None, color=None, uv=None):
        self.position = HashableDict(position)
        if normal is not None:
            self.normal = HashableDict(normal)
        if color is not None:
            self.color = HashableDict(color)
        if uv is not None:
            self.uv = HashableDict(uv)

    def vars(self):
        return HashableDict(self.__dict__)


class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
        # Calculating line length during initialization
        self.length = self.calculate_line_length(self.point1.position,
                                                 self.point2.position)
        self.color = self.calculate_line_color(point1.color, point2.color)

    def calculate_line_length(self, p1, p2):
        if len(p1.keys()) == 3 and len(p2.keys()) == 3:
            return self._calculate_3d_line_length(p1['x'], p1['y'], p1['z'], p2['x'], p2['y'], p2['z'])

    @staticmethod
    @njit
    def _calculate_3d_line_length(x1, y1, z1, x2, y2, z2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5

    @staticmethod
    def calculate_line_color(rgb1, rgb2):
        if rgb1 is not None or rgb2 is not None:
            r = (rgb1['r'] + rgb2['r']) / 2
            g = (rgb1['g'] + rgb2['g']) / 2
            b = (rgb1['b'] + rgb2['b']) / 2
            return r, g, b
        return None


class Loop:
    def __init__(self, lines, normal=None):
        self.lines = lines

        # Calculating loop color based on the colors of points making up the lines
        self.color, self.hue = self.calculate_average_color_based_on_points()
        self.center = self.calculate_center()
        self.perimeter = self.calculate_perimeter()
        self.area = self.calculate_area_shoelace()
        self.normal = self.calculate_normal(normal)
        self.skew, self.growth = self.calculate_fit()
        self.is_valid = None

        self.G = nx.Graph()

    def calculate_normal(self, normal):
        if normal is not None:
            return normal
        else:
            points = set([line.point1 for line in self.lines])
            if len(points) == 1:
                raise Exception("Line only has 1 unique point")
            #print(points)
            positions = [[point.position['x'], point.position['y'], point.position['z']] for point in points]
            pca = PCA(n_components=2)
            #print(positions)
            pca.fit(np.asarray(positions).reshape(-1, 3))
            normal = pca.components_[-1]
            return {'nx': normal[0], 'ny': normal[1], 'nz': normal[2]}

    def calculate_center(self):
        points = set([line.point1 for line in self.lines])
        positions = [[point.position['x'], point.position['y'], point.position['z']] for point in points]
        center_p = np.mean(np.asarray(positions).reshape(-1, 3), axis=0)
        center = {'x': center_p[0], 'y': center_p[1], 'z': center_p[2]}
        return center

    def calculate_perimeter(self):
        perimeter = sum(line.length for line in self.lines)
        return perimeter

    def calculate_average_color_based_on_points(self):
        point_colors = []
        for line in self.lines:
            if line.point1.color is not None or line.point2.color is not None:
                point_colors.append(list(line.point1.color.values()))
                #point_colors.append(list(line.point2.color.values())) # Causes duplicate measurements
        if point_colors:
            colors_array = np.array(point_colors)
            colors_list = colors_array.mean(axis=0).tolist()
            colors = {'x': colors_list[0], 'y': colors_list[1], 'z': colors_list[2]}
            hue = self.calculate_average_hue(colors_array)
            return colors, hue
        return None

    @staticmethod
    @njit
    def calculate_average_hue(point_colors):
        r_sum = 0.0
        g_sum = 0.0
        b_sum = 0.0

        for color in point_colors:
            r_sum += color[0]
            g_sum += color[1]
            b_sum += color[2]

        n = len(point_colors)
        r = r_sum / n
        g = g_sum / n
        b = b_sum / n
        h, _, _ = rgb_to_hsv(r, g, b)
        return h

    # Area calculation using the Shoelace formula
    def calculate_area_shoelace(self):
        # TODO: Fix this, not working correctly
        points = [line.point1 for line in self.lines]
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i].position['x'] * points[j].position['y']
            area -= points[j].position['x'] * points[i].position['y']
        area = abs(area) / 2.0
        return area

    def calculate_fit(self):
        # Collecting normals of points in the loop
        points = set([line.point1 for line in self.lines])
        point_normals = [[point.normal['nx'], point.normal['ny'], point.normal['nz']] for point in points]

        # Calculate the normalized difference in point normals with the plane normal
        # The resulting array has 2 columns
        # The first column has a value representing the angle between the point normal and plane normal
        # The second column has a value representing the minimum angle between the point normal and plane normal
        # As such, normals in the first column can range from [0, 1], but only [0, 0.5] in the second column
        point_normal_differences = calculate_point_loop_fit(point_normals, list(self.normal.values()))

        # Calculate loop skew
        point_std_nd = np.std(point_normal_differences, axis=0)
        point_skew = point_std_nd[0] - point_std_nd[1]
        loop_skew = np.mean(point_skew)
        #print(loop_skew)

        # Calculate loop growth (normalized to [-1, 1]
        # Relative to loop normal,
        # values of [0, 0.5) represent a shrinking loop
        # 0.5 represents a non-shrinking loop
        # values of (0.5, 1] represent a growing loop
        mean_nd = np.mean(point_normal_differences, axis=0)
        loop_growth = (0.5 - mean_nd[0])*2
        #print(loop_growth)

        # Other ideas
        # Measure variance(max angle) - loops with small variance for max angle mean loop is good fit
        return loop_skew, loop_growth

    @staticmethod
    def from_points(points):
        """
        Create loop from ordered list of points
        Faster than creating lines first and then creating
        :return: Loop
        """
        lines = [Line(points[i], points[i + 1]) for i in range(len(points) - 1)] + [Line(points[-1], points[0])]
        loop = Loop.from_lines(lines)
        return loop

    @staticmethod
    def from_lines(lines, allow_invalid_loop=False):
        loop = Loop(lines)

        for idx, line in enumerate(lines):
            n1, n2 = line.point1.vars(), line.point2.vars()
            loop.G.add_edge(n1, n2)
            loop.G.edges[n1, n2]['length'] = line.length
            loop.G.edges[n1, n2]['color'] = line.color

        loop.check_validity()

        if loop.is_valid:
            return loop
        elif allow_invalid_loop:
            return loop
        else:
            raise Exception("Loop is not valid; lines do not form closed polygon")

    #@njit
    def check_validity(self):
        # A component is a loop if each node has exactly two neighbors (closed polygon)
        self.is_valid = all(len(list(self.G.neighbors(node))) == 2 for node in self.G.nodes())
        return self.is_valid

    def vars(self):
        #print(self.__dict__)
        return HashableDict(self.__dict__)


class LoopSet:
    def __init__(self):
        self.mesh: o3d.geometry.TriangleMesh = None
        self.loops = []  # TODO: Remove this; loops should be accessed from
        self.bbox: o3d.geometry.OrientedBoundingBox() = None
        self.G = nx.Graph()

    def filter(self, **kwargs):
        """

        :param kwargs:
        Accepted kwargs: "max_skew", "max_area"
        :return:
        """
        self.filter_kwargs = kwargs
        loop_set = LoopSet()

        #print(len(self.G.nodes))
        loop_set.G = nx.subgraph_view(self.G, filter_node=self._filter_nodes)
        #print(len(loop_set.G.nodes))

        loops = [lines for lines in [node['G'].edges for node in loop_set.G.nodes()]]

        loops2 = []
        for loop in loops:
            lines2 = []
            for line in loop:
                line2 = []
                for point in line:
                    position = point['position']
                    normal = point['normal']
                    color = point['color']
                    line2.append(Point(position=position, normal=normal, color=color))
                line3 = Line(line2[0], line2[1])
                lines2.append(line3)
            loops2.append(Loop.from_lines(lines2))

        loop_set.loops = loops2
        #print(len(loops))
        loop_set.bbox = self.bbox
        loop_set.mesh = self.mesh

        return loop_set

    def _filter_nodes(self, nodes):
        keep_loop: bool = True

        for kwarg in self.filter_kwargs.items():
            if kwarg[0] == "max_skew":
                keep_loop = nodes['skew'] <= kwarg[1]
                if not keep_loop:
                    return False
            if kwarg[0] == "max_area":
                keep_loop = nodes['area'] <= kwarg[1]
                if not keep_loop:
                    return False
            if kwarg[0] == "growth":
                keep_loop = nodes['growth'] >= kwarg[1][0]
                if not keep_loop:
                    return False
                keep_loop = nodes['growth'] <= kwarg[1][1]
                if not keep_loop:
                    return False
            if kwarg[0] == "hue":
                keep_loop = nodes['hue'] >= kwarg[1][0]
                if not keep_loop:
                    return False
                keep_loop = nodes['hue'] <= kwarg[1][1]
                if not keep_loop:
                    return False
        return True

    def from_mesh(self, mesh: o3d.geometry.TriangleMesh = None, normals=None, resolution=0.003, method="approximate"):
        #print(mesh)
        self.mesh = copy.deepcopy(mesh)
        #print(self.mesh)
        self.bbox = mesh.get_oriented_bounding_box()
        point = mesh.get_center()

        if method == "exact":
            for normal in tqdm(normals):
                results = Slicer(copy.deepcopy(mesh),
                                 copy.deepcopy(self.bbox),
                                 point, [normal], resolution).slice_plane_o3d()

                point_positions, point_normals, point_colors, line_indices = results[0]

                # Process the data to create Points and then Lines and Loops
                points = [Point(position={'x': position[0], 'y': position[1], 'z': position[2]},
                                normal={'nx': normal[0], 'ny': normal[1], 'nz': normal[2]},
                                color={'r': color[0], 'g': color[1], 'b': color[2]})
                          for position, normal, color in zip(point_positions, point_normals, point_colors)]

                # Create Lines from Points using line_indices
                lines = [Line(points[i], points[j]) for i, j in line_indices]

                self.loops += self._find_loops(points, lines)
            self._to_graph()
        elif method == "approximate":
            results = Slicer(copy.deepcopy(mesh),
                             copy.deepcopy(self.bbox),
                             point, normals, resolution).slice_plane_voxel_approximation()

            for result in tqdm(results, "Creating loops from points and lines"):
                point_positions, point_normals, point_colors, line_indices = result

                # Process the data to create Points and then Lines and Loops
                points = [Point(position={'x': position[0], 'y': position[1], 'z': position[2]},
                                normal={'nx': normal[0], 'ny': normal[1], 'nz': normal[2]},
                                color={'r': color[0], 'g': color[1], 'b': color[2]})
                          for position, normal, color in zip(point_positions, point_normals, point_colors)]

                # Create Lines from Points using line_indices
                lines = [Line(points[i], points[j]) for i, j in line_indices]

                self.loops += self._find_loops(points, lines)
            self._to_graph()
        else:
            raise Exception(f"Slicing method '{method}' is not valid. Choose one of 'exact' or 'approximate'")

    def _find_loops(self, points, lines):
        G = nx.Graph()
        # Add edges to the graph
        for line in lines:
            start_idx = points.index(line.point1)
            end_idx = points.index(line.point2)
            G.add_edge(start_idx, end_idx)

        loops = []
        loops_added = 0
        loops_not_added = 0

        for loop_point_indices in nx.simple_cycles(G):
            loop_points = [points[i] for i in loop_point_indices]
            try:
                loop = Loop.from_points(loop_points)
                loops.append(loop)
                loops_added += 1
            except Exception as e:
                loops_not_added += 1

        #print(f"{loops_added}/{loops_added+loops_not_added} valid loops found")

        return loops

    def _create_lines_from_points(self, points):
        # Create lines from points in the correct order to form a loop
        lines = []
        for i in range(len(points)):
            line = Line(points[i], points[(i + 1) % len(points)])  # Loop back to the first point at the end
            lines.append(line)
        return lines

    def _loop_to_lineset(self, loop):
        # Convert a Loop object to an Open3D LineSet object
        line_set = o3d.geometry.LineSet()

        # Set points
        points = [[point.position['x'], point.position['y'], point.position['z']] for line in loop.lines for point in [line.point1, line.point2]]
        line_set.points = o3d.utility.Vector3dVector(points)

        # Set lines
        lines = [[i, i + 1] for i in range(0, len(points), 2)]
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # TODO: Make functional
        # Optional: Set color for lines
        #colors = [[0.5, 1, 0.5] for _ in range(len(lines))]
        #line_set.colors = o3d.utility.Vector3dVector(colors)
        colors = [list(loop.color.values()) for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def _to_graph(self):
        for idx, loop in enumerate(self.loops):
            #print(loop.G)
            # TODO: Would be better to add loop: {nodes: _, edges: _} than 'loop'
            #  'loop': HashableDict({'nodes': loop.G.nodes(), 'edges': loop.G.edges()})
            self.G.add_node(HashableDict({'G': loop.G,
                                          'center': HashableDict(loop.center),
                                          'color': HashableDict(loop.color),
                                          'hue': loop.hue,
                                          'perimeter': loop.perimeter,
                                          'area': loop.area,
                                          'normal': HashableDict(loop.normal),
                                          'skew': loop.skew,
                                          'growth': loop.growth,
                                          'is_valid': loop.is_valid}))
            #print(self.G.nodes)
            #quit()

    def add_loop(self, loop):
        self.loops.append(loop)
        self._to_graph()

    def view(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        line_sets = None

        for loop in self.loops:
            line_set = self._loop_to_lineset(loop)
            #vis.add_geometry(line_set)
            if line_sets is None:
                line_sets = line_set
            else:
                line_sets += line_set

        vis.add_geometry(line_sets)

        if self.bbox is None:
            self.bbox = line_sets.get_oriented_bounding_box()

        vis.add_geometry(self.bbox)

        opt = vis.get_render_option()
        opt.line_width = 50.
        opt.point_show_normal = False
        opt.background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()

    def connect_loops(self):
        line_sets = None

        for loop in self.loops:
            line_set = self._loop_to_lineset(loop)
            # vis.add_geometry(line_set)
            if line_sets is None:
                line_sets = line_set
            else:
                line_sets += line_set

        loop_mesh = LoopMesh()
        loop_mesh.tetra_mesh = connect_loops_blender(line_sets)
        #print(self.mesh)
        loop_mesh.tetra_mesh = add_colors_and_normals(self.mesh, loop_mesh.tetra_mesh)
        loop_mesh.bbox = self.bbox
        return loop_mesh

    def read(self, path):
        with open(path, 'rb') as f:
            self.G = pickle.load(f)

    def write_line_set(self, path):
        line_sets = None

        for loop in self.loops:
            line_set = self._loop_to_lineset(loop)
            # vis.add_geometry(line_set)
            if line_sets is None:
                line_sets = line_set
            else:
                line_sets += line_set

        o3d.io.write_line_set(path, line_sets)

    def write(self, path):
        #nx.write_gml(self.G, path=path, stringizer=nx.readwrite.gml.literal_stringizer(self.G)) # TODO: Won't work because of graph inside graph
        #with open(path, 'wb') as f:
        #    pickle.dump(copy.deepcopy(self.G), f, pickle.HIGHEST_PROTOCOL)
        #    # TODO: get TypeError: cannot pickle 'open3d.cuda.pybind.geometry.OrientedBoundingBox' object
        pass


class LoopMesh:
    def __init__(self):
        self.orig_mesh = o3d.geometry.TriangleMesh()
        self.tetra_mesh = o3d.geometry.TetraMesh()
        self.bbox = None

    def view(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.add_geometry(self.tetra_mesh)

        #if self.bbox is None:
        #    self.bbox = self.tetra_mesh.get_oriented_bounding_box(robust=True)

        vis.add_geometry(self.bbox)

        opt = vis.get_render_option()
        opt.line_width = 50.
        opt.mesh_show_wireframe = True
        opt.background_color = np.asarray([1, 1, 1])
        vis.run()
        vis.destroy_window()

    def write(self):
        o3d.io.write_line_set(self.tetra_mesh.extract_triangle_mesh())


class LoopTree:
    def __init__(self):
        """
        Placeholder for building trees of loops from a loop set, specifically adding edges based on
        loops that are close by, have similar normals, and similar area and perimeter. Functionality
        will be like Bridge Edge Loops tool in Blender
        """
        pass


def calculate_point_loop_fit(point_normals, loop_normal):
    # Ensure the inputs are numpy arrays
    point_normals = np.array(point_normals)
    loop_normal = np.array(loop_normal)

    # Calculate dot products
    dot_products = np.dot(point_normals, loop_normal)

    # Clip values to avoid errors due to numerical precision
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Calculate angles in radians
    angles_radians = np.arccos(dot_products)

    # Convert angles to degrees and normalize to [0, 1]
    angles_degrees = np.degrees(angles_radians) / 180.0

    # Calculate the smallest angle when the second normal is flipped
    flipped_angles_degrees = np.degrees(np.minimum(angles_radians, np.pi - angles_radians)) / 180.0

    loop_fit = np.column_stack((angles_degrees, flipped_angles_degrees))

    # Return the results
    return loop_fit


def sample_normals(divisions):
    normals = []

    # Angle increments based on the number of divisions
    theta_step = 2 * np.pi / divisions
    phi_step = np.pi / 2 / divisions

    for i in range(divisions):
        for j in range(divisions):
            # Compute the angles
            theta = theta_step * i
            phi = phi_step * (j + 0.5)

            # Convert spherical coordinates to Cartesian coordinates
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)

            # The Cartesian coordinates are the normal vector
            normal = np.array([x, y, z])
            normals.append(normal)

    return normals


@njit
def rgb_to_hsv(r, g, b):
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c

    if max_c == min_c:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif max_c == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    if max_c == 0:
        s = 0
    else:
        s = (diff / max_c)

    v = max_c

    # Normalize h to 0-1 range (like OpenCV's 0-179 range divided by 180)
    h = h / 360.0

    return h, s, v
