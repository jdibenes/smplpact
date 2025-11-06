#==============================================================================
# SMPL Painting And Charting Tools
#==============================================================================

import os
import time
import math
import collections
import operator
import numpy as np
import cv2
import torch
import pyrender
import trimesh.visual
import trimesh.exchange.obj
import smplx
import roma

from PIL import Image, ImageFont, ImageDraw


#------------------------------------------------------------------------------
# File
#------------------------------------------------------------------------------

def scan_path(base_path, files_sort=False, files_key=None, files_reverse=False, folders_sort=False, folders_key=None, folders_reverse=False):
    items = os.listdir(base_path)
    paths = [os.path.join(base_path, item) for item in items]
    files = [path for path in paths if (os.path.isfile(path))]
    folders = [path for path in paths if (os.path.isdir(path))]
    if (files_sort):
        files.sort(key=files_key, reverse=files_reverse)
    if (folders_sort):
        folders.sort(key=folders_key, reverse=folders_reverse)
    return (files, folders) # tuple return


#------------------------------------------------------------------------------
# Math
#------------------------------------------------------------------------------

# TODO: handle m == 0
def math_normalize(a):
    m = np.linalg.norm(a)
    return (a / m, m) # tuple return


def math_transform_points(points, pose, inverse=False):
    return ((points @ pose[:3, :3]) + pose[3:4, :3]) if (not inverse) else ((points - pose[3:4, :3]) @ pose[:3, :3].T)


def math_transform_bearings(bearings, pose, inverse=False):
    return (bearings @ pose[:3, :3]) if (not inverse) else (bearings @ pose[:3, :3].T)


#------------------------------------------------------------------------------
# Geometry Solvers
#------------------------------------------------------------------------------

# TODO: handle error for singular matrix
def geometry_solve_basis(vas, vbs, vad, vbd):
    return np.linalg.inv(np.vstack((vas, vbs, np.cross(vas, vbs)))) @ np.vstack((vad, vbd, np.cross(vad, vbd)))


def geometry_solve_fov_z(w, h, fx, fy, cx, cy, x, y, z, center, points):
    dp = (points - center)
    dx = dp @ x.T
    dy = dp @ y.T
    dz = dp @ z.T
    ix = fx * dx
    iy = fy * dy
    xp = dz + (ix / (w - cx))
    xn = dz + (ix / (0 - cx))
    yp = dz + (iy / (h - cy))
    yn = dz + (iy / (0 - cy))
    nz = np.max(np.hstack((xp, xn, yp, yn)))
    return nz


def geometry_fov_to_f(fov, pixels):
    f = (pixels / 2) / np.tan(fov / 2)
    return f


def geometry_f_to_fov(f, pixels):
    fov = 2 * np.arctan((pixels / 2) / f)
    return fov


def geometry_distance_point_line(line_point, line_direction, point):
    offset = point - line_point
    ny = offset @ line_direction.T
    xz = offset - ny * line_direction
    return np.linalg.norm(xz)


def geometry_distance_point_segment(line_start, line_end, point):
    line_direction, line_length = math_normalize(line_end - line_start)
    offset = point - line_start
    ny = offset @ line_direction.T
    if (ny <= 0):
        return np.linalg.norm(offset)
    offset = point - line_end
    ny = offset @ line_direction.T
    if (ny >= 0):
        return np.linalg.norm(offset)
    xz = offset - ny * line_direction
    return np.linalg.norm(xz)


#------------------------------------------------------------------------------
# Texture Processing
#------------------------------------------------------------------------------

def texture_load_image(filename_image, load_alpha=True, alpha=255):
    rgb = cv2.imread(filename_image, cv2.IMREAD_COLOR_RGB)
    raw = cv2.imread(filename_image, cv2.IMREAD_UNCHANGED)
    a = raw[:, :, 3] if ((load_alpha) and (raw.shape[2] == 4)) else (np.ones((rgb.shape[0], rgb.shape[1], 1), rgb.dtype) * alpha)
    return np.dstack((rgb, a))


def texture_load_uv(filename_uv):
    with open(filename_uv, 'r') as obj_file:
        obj_mesh_a = trimesh.exchange.obj.load_obj(file_obj=obj_file, maintain_order=True)
    with open(filename_uv, 'r') as obj_file:
        obj_mesh_b = trimesh.exchange.obj.load_obj(file_obj=obj_file)
    mesh_vertices_a = obj_mesh_a['geometry'][filename_uv]['vertices']
    mesh_vertices_b = obj_mesh_b['geometry'][filename_uv]['vertices']
    mesh_faces_a = obj_mesh_a['geometry'][filename_uv]['faces']
    mesh_faces_b = obj_mesh_b['geometry'][filename_uv]['faces']
    mesh_uv_a = obj_mesh_a['geometry'][filename_uv]['visual'].uv
    mesh_uv_b = obj_mesh_b['geometry'][filename_uv]['visual'].uv
    uv_transform = np.zeros(mesh_vertices_b.shape[0], np.int64)
    for face_index in range(0, mesh_faces_b.shape[0]):
        for vertex_index in range(0, 3):
            uv_transform[mesh_faces_b[face_index, vertex_index]] = mesh_faces_a[face_index, vertex_index]
    return (mesh_vertices_a, mesh_vertices_b, mesh_faces_a, mesh_faces_b, mesh_uv_a, mesh_uv_b, uv_transform) # tuple return


def texture_load_font(font_name, font_size):
    return ImageFont.truetype(font_name, font_size)


def texture_stack(textures, fill_color, spacing, vertical):
    axis_a, axis_b = (0, 1) if (vertical) else (1, 0)
    pad = np.zeros((len(textures), 4), np.int32)
    d = np.array([texture.shape[axis_b] for texture in textures], np.int32)
    fill_d = np.max(d) - d
    pad[0:, 2 * axis_b + 0] = fill_d // 2
    pad[0:, 2 * axis_b + 1] = fill_d - pad[:, 2 * axis_b + 0]
    pad[1:, 2 * axis_a + 0] = spacing
    return np.concatenate([cv2.copyMakeBorder(texture, pad[i, 0], pad[i, 1], pad[i, 2], pad[i, 3], cv2.BORDER_CONSTANT, value=fill_color) for i, texture in enumerate(textures)], axis_a)


def texture_pad(texture, pad_w, pad_h, fill_color):
    h, w = texture.shape[0:2]
    pad_x = math.ceil(w * pad_w)
    pad_y = math.ceil(h * pad_h)
    return cv2.copyMakeBorder(texture, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=fill_color)


def texture_create_text(text, font, font_color, bg_color=(255, 255, 255, 255), stroke_width=0):
    bbox = font.getbbox(text, stroke_width=stroke_width)
    image = Image.new('RGBA', (bbox[2], bbox[3]), bg_color)
    ImageDraw.Draw(image).text((0, 0), text, font_color, font, stroke_width=stroke_width)
    return np.array(image.crop(bbox))


def texture_create_multiline_text(text_list, font, font_color, bg_color=(255, 255, 255, 255), stroke_width=0, spacing=4):
    return texture_stack([texture_create_text(text, font, font_color, bg_color, stroke_width) for text in text_list], bg_color, spacing, True)


def texture_create_visual(uv, texture):
    return trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(texture))


def texture_uv_to_uvx(uv, image_shape):
    u = uv[:, 0:1] * (image_shape[1] - 1)
    v = (1 - uv[:, 1:2]) * (image_shape[0] - 1)
    return np.hstack((u, v))


def texture_uvx_invert(uvx, image_shape, axis):
    uvx[:, axis] = (image_shape[1 - axis] - 1) - uvx[:, axis]
    return uvx


# TODO: ignores last row and column to simplify bilinear interpolation
def texture_test_inside(texture, x, y):
    return (x >= 0) & (y >= 0) & (x < (texture.shape[1] - 1)) & (y < (texture.shape[0] - 1))


def texture_read(texture, x, y):
    xf = np.floor(x)
    yf = np.floor(y)

    a1 = (x - xf)[:, np.newaxis]
    b1 = (y - yf)[:, np.newaxis]
    a0 = 1 - a1
    b0 = 1 - b1

    x0 = xf.astype(np.int32)
    y0 = yf.astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    return b0 * (a0 * texture[y0, x0, :] + a1 * texture[y0, x1, :]) + b1 * (a0 * texture[y1, x0, :] + a1 * texture[y1, x1, :])


def texture_alpha_blend(texture_1a, texture_a, alpha):
    return (1 - alpha) * texture_1a + alpha * texture_a


def texture_alpha_remap(alpha, src, dst):
    ah = alpha >= src[1]
    al = alpha < src[1]
    alpha[ah] = np.interp(alpha[ah], [src[1], src[2]], [dst[1], dst[1]])
    alpha[al] = np.interp(alpha[al], [src[0], src[1]], [dst[0], dst[1]])
    return alpha


def texture_processor(simplex_uvx, callback, tolerance=0):
    # uvx : [u * (w - 1), (1 - v) * (h - 1)]
    u = simplex_uvx[:, 0]
    v = simplex_uvx[:, 1]

    left = math.floor(np.min(u))
    right = math.ceil(np.max(u))
    top = math.floor(np.min(v))
    bottom = math.ceil(np.max(v))

    if ((left < right) and (top < bottom)):
        box = np.mgrid[left:right, top:bottom].T.reshape((-1, 2))
        anchor = simplex_uvx[2:3, :]
        ab = (box - anchor) @ np.linalg.inv(simplex_uvx[0:2, :] - anchor)
        abc = np.hstack((ab, 1 - ab[:, 0:1] - ab[:, 1:2]))
        mask = np.logical_and.reduce(abc >= -tolerance, axis=1)
        if (np.any(mask)):
            callback(box[mask, :], abc[mask, :])


#------------------------------------------------------------------------------
# Mesh Processing
#------------------------------------------------------------------------------

def mesh_create(vertices, faces, visual=None):
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def mesh_expand(mesh, uv_transform, faces_extended, visual=None):
    return mesh_create(mesh.vertices.view(np.ndarray)[uv_transform, :], faces_extended, visual)


def mesh_faces_of_vertices(mesh, vertex_indices):
    vertex_faces = mesh.vertex_faces
    face_indices = set()
    for vertex_index in vertex_indices:
        face_indices.update(vertex_faces[vertex_index, :])
    face_indices.discard(-1)
    return face_indices


def mesh_vertices_of_faces(mesh, face_indices):
    faces = mesh.faces.view(np.ndarray)
    vertex_indices = set()
    for face_index in face_indices:
        vertex_indices.update(faces[face_index])
    return vertex_indices


def mesh_raycast(mesh, origin, direction):
    point, rid, tid = mesh.ray.intersects_location(origin, direction, multiple_hits=False)
    return (point, tid[0]) if (len(rid) > 0) else (None, None) # tuple return


def mesh_closest(mesh, origin):
    point, distance, tid = mesh.nearest.on_surface(origin)
    return (point, tid[0], distance[0]) if (len(tid) > 0) else (None, None, None) # tuple return


def mesh_align_prior(mesh, face_index, align_axis, align_axis_fallback, tolerance=0):
    align_normal = mesh.face_normals[np.newaxis, face_index, :]
    align_prior, nap = math_normalize(align_axis - (align_normal @ align_axis.T) * align_normal)
    return align_prior if (nap > tolerance) else math_normalize(align_axis_fallback - (align_normal @ align_axis_fallback.T) * align_normal)[0]


def mesh_snap_to_vertex(mesh, point, face_index):
    vertex_indices = mesh.faces.view(np.ndarray)[face_index]
    vertices = mesh.vertices.view(np.ndarray)[vertex_indices, :]
    distances = np.linalg.norm(point - vertices, axis=1)
    return np.argmin(distances)


def mesh_select_vertices(mesh, origin_vertex_index, radius, level):
    vertices  = mesh.vertices.view(np.ndarray)
    neighbors = mesh.vertex_neighbors
    buffer    = collections.deque()
    distances = {origin_vertex_index : 0}

    buffer.append((origin_vertex_index, 0, 0))

    while (len(buffer) > 0):
        vertex_index, vertex_distance, vertex_level = buffer.popleft()
        vertex_xyz = vertices[vertex_index, :]
        for neighbor_index in neighbors[vertex_index]:
            neighbor_xyz = vertices[neighbor_index, :]
            neighbor_distance = vertex_distance + np.linalg.norm(neighbor_xyz - vertex_xyz)
            neighbor_level = vertex_level + 1
            if ((neighbor_distance <= radius) and (neighbor_level <= level) and (neighbor_distance < distances.get(neighbor_index, np.Inf))):          
                buffer.append((neighbor_index, neighbor_distance, neighbor_level))
                distances[neighbor_index] = neighbor_distance
    
    return distances


def mesh_select_complete_faces(mesh, vertex_indices):
    vertex_faces = mesh.vertex_faces
    faces = mesh.faces.view(np.ndarray)
    face_indices_seen = set()
    face_indices_complete = set()
    vertex_indices_complete = set()
    
    for vertex_index in vertex_indices:
        for face_index in vertex_faces[vertex_index, :]:
            if (face_index < 0):
                break
            if (face_index not in face_indices_seen):
                face_indices_seen.add(face_index)
                face_vertices = faces[face_index]
                keep = all([face_vertex in vertex_indices for face_vertex in face_vertices])
                if (keep):
                    face_indices_complete.add(face_index)
                    vertex_indices_complete.update(face_vertices)
    
    return (face_indices_complete, vertex_indices_complete) # tuple return


# TODO: this is slow
def mesh_to_renderer(mesh):
    return pyrender.Mesh.from_trimesh(mesh)


class mesh_neighborhood_builder:
    def __init__(self, mesh):
        self._mesh = mesh
        self._mesh_faces = self._mesh.faces.view(np.ndarray)
        self._mesh_vertex_faces = self._mesh.vertex_faces
        self._seen_face = set()
        self._seen_vertex = set()
        self._iterations = 0

    def fetch(self, expand_faces, ignore_faces):
        result = set()
        self._seen_face.update(expand_faces)
        self._seen_face.update(ignore_faces)
        for face_anchor in expand_faces:
            for vertex_index in self._mesh_faces[face_anchor]:
                if (vertex_index not in self._seen_vertex):
                    self._seen_vertex.add(vertex_index)
                    for face_index in self._mesh_vertex_faces[vertex_index]:
                        if (face_index < 0):
                            break
                        if (face_index not in self._seen_face):
                            result.add(face_index)
        self._iterations += 1
        return result
    
    def level(self):
        return self._iterations


class mesh_neighborhood_processor_command:
    CONTINUE = 0
    IGNORE = 1
    EXPAND = 2


class mesh_neighborhood_processor:
    def __init__(self, mesh, faces, callback):
        self._mnb = mesh_neighborhood_builder(mesh)
        self._faces = faces
        self._expand_faces = set()
        self._ignore_faces = set()
        self._callback = callback
        self._done = False

    def invoke(self, max_iterations):
        tpdl = []
        for _ in range(0, max_iterations):
            if (len(self._expand_faces) > 0):
                self._faces = self._mnb.fetch(self._expand_faces, self._ignore_faces)
                self._expand_faces.clear()
                self._ignore_faces.clear()
            for face_anchor in self._faces:
                result = self._callback(face_anchor, self._mnb.level())
                code = result[0]
                if (code == mesh_neighborhood_processor_command.EXPAND):
                    self._expand_faces.add(face_anchor)
                elif (code == mesh_neighborhood_processor_command.IGNORE):
                    self._ignore_faces.add(face_anchor)
                tpdl.append(result)
            if (len(self._expand_faces) < 1):
                self._done = True
                break
        return tpdl

    def invoke_timeslice(self, timeout, steps=1):
        tpdl = []
        start = time.perf_counter()
        while (not self.done()):
            result = self.invoke(steps)
            tpdl.extend(result)
            if ((time.perf_counter() - start) >= timeout):
                break
        return tpdl

    def status(self):
        return self._mnb.level()

    def done(self):
        return self._done


class mesh_list_processor:
    def __init__(self, face_list, callback):
        self._face_list = face_list
        self._face_index = 0
        self._face_count = len(face_list)
        self._callback = callback        
        self._done = False

    def invoke(self, max_iterations):
        tpdl = []
        for _ in range(0, max_iterations):
            if (self._face_index < self._face_count):
                result = self._callback(self._face_list[self._face_index], -1)
                self._face_index += 1
                tpdl.append(result)
            if (self._face_index >= self._face_count):
                self._done = True
                break
        return tpdl
        
    def invoke_timeslice(self, timeout, steps=1):
        tpdl = []
        start = time.perf_counter()
        while (not self.done()):
            result = self.invoke(steps)
            tpdl.extend(result)
            if ((time.perf_counter() - start) >= timeout):
                break
        return tpdl

    def status(self):
        return (self._face_index, self._face_count) # tuple return

    def done(self):
        return self._done


#------------------------------------------------------------------------------
# Mesh Painting
#------------------------------------------------------------------------------

class mesh_neighborhood_operation_color:
    def __init__(self, mesh_vertices, mesh_faces, mesh_uvx, target, tolerance=0):
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._mesh_uvx = mesh_uvx
        self._target = target
        self._tolerance = tolerance

    def paint(self, face_index, level):
        vertex_indices = self._mesh_faces[face_index]
        self._level = level
        self._result = (mesh_neighborhood_processor_command.IGNORE, None)
        texture_processor(self._mesh_uvx[vertex_indices, :], self._paint_uv, self._tolerance)
        return self._result
    
    def _paint_uv(self, pixels, weights):
        self._result = self._target(pixels, self._level)


# TODO: THIS DISTANCE IS NOT GEODESIC
class mesh_neighborhood_operation_brush:
    def __init__(self, mesh_vertices, mesh_faces, mesh_uvx, origin, target, tolerance=0):
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._mesh_uvx = mesh_uvx
        self._origin = origin
        self._target = target
        self._tolerance = tolerance

    def paint(self, face_index, level):
        vertex_indices = self._mesh_faces[face_index]
        self._simplex_3d = self._mesh_vertices[vertex_indices, :]
        self._level = level
        self._result = (mesh_neighborhood_processor_command.IGNORE, None)
        texture_processor(self._mesh_uvx[vertex_indices, :], self._paint_uv, self._tolerance)
        return self._result
    
    def _paint_uv(self, pixels, weights):
        distances = np.linalg.norm((weights @ self._simplex_3d) - self._origin, axis=1)
        self._result = self._target(pixels, distances, self._level)


class mesh_neighborhood_operation_decal:
    def __init__(self, mesh_vertices, mesh_faces, mesh_face_normals, mesh_uvx, uv_transform, origin, target, tolerance=0):
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._mesh_face_normals = mesh_face_normals
        self._mesh_uvx = mesh_uvx
        self._uv_transform = uv_transform
        self._origin = origin
        self._target = target
        self._tolerance = tolerance

    def _bootstrap(self):
        value = self._target(self._mesh_vertices, self._face_normal, self._origin, self._vertex_indices_b, self._vertex_indices_a, None, None, self._level)
        self._result = (value, None, None)
        return value == mesh_neighborhood_processor_command.EXPAND

    def paint(self, face_index, level):
        self._face_normal = self._mesh_face_normals[np.newaxis, face_index, :]
        self._vertex_indices_b = self._mesh_faces[face_index]
        self._vertex_indices_a = self._uv_transform[self._vertex_indices_b]
        self._level = level
        if (not self._bootstrap()):
            return self._result
        self._result = (mesh_neighborhood_processor_command.IGNORE, None, None)
        texture_processor(self._mesh_uvx[self._vertex_indices_b, :], self._paint_uv, self._tolerance)
        return self._result

    def _paint_uv(self, pixels, weights):
        self._result = self._target(self._mesh_vertices, self._face_normal, self._origin, self._vertex_indices_b, self._vertex_indices_a, pixels, weights, self._level)        


class paint_color_solid:
    def __init__(self, color, stop_level, render_buffer):
        self._color = color
        self._stop_level = stop_level
        self._render_buffer = render_buffer

    def paint(self, pixels, level):
        selection = pixels
        pixels_painted = selection.shape[0]
        if (pixels_painted > 0):
            self._render_buffer[selection[:, 1], selection[:, 0], :] = self._color
        cc = level < self._stop_level
        command = mesh_neighborhood_processor_command.EXPAND if (cc) else mesh_neighborhood_processor_command.IGNORE
        return (command, selection) # tuple return


class paint_brush_solid:
    def __init__(self, size, color, render_buffer, fill_test=0.0):
        self._size = size
        self._color = color
        self._render_buffer = render_buffer
        self._fill_test = fill_test

    def paint(self, pixels, distances, level):
        mask = distances < self._size
        selection = pixels[mask, :]
        pixels_painted = selection.shape[0]
        if (pixels_painted > 0):
            self._render_buffer[selection[:, 1], selection[:, 0], :] = self._color
        cc = pixels_painted > int(self._fill_test * pixels.shape[0])
        command = mesh_neighborhood_processor_command.EXPAND if (cc) else mesh_neighborhood_processor_command.IGNORE
        return (command, selection) # tuple return


class paint_brush_gradient:
    def __init__(self, size, color_center, color_edge, hardness, render_buffer, fill_test=0.0):
        self._size = size
        self._color_center = color_center
        self._color_edge = color_edge
        self._render_buffer = render_buffer
        self._fill_test = fill_test
        self._src = [0, hardness, 1]
        self._dst = [0, 0.5, 1]

    def paint(self, pixels, distances, level):
        mask = distances < self._size
        selection = pixels[mask, :]
        pixels_painted = selection.shape[0]
        if (pixels_painted > 0):            
            self._render_buffer[selection[:, 1], selection[:, 0], :] = texture_alpha_blend(self._color_center, self._color_edge, texture_alpha_remap(distances[mask, np.newaxis] / self._size, self._src, self._dst))
        cc = pixels_painted > int(self._fill_test * pixels.shape[0])
        command = mesh_neighborhood_processor_command.EXPAND if (cc) else mesh_neighborhood_processor_command.IGNORE
        return (command, selection) # tuple return


# TODO: THIS UNWRAPPING METHOD IS AFFECTED BY THE ORDER IN WHICH FACES ARE PROCESSED
class paint_decal_solid:
    def __init__(self, align_prior, angle, scale, image_buffer, render_buffer, double_cover_test=True, fill_test=0.0, tolerance=0):
        self._align_prior = align_prior
        self._angle = angle
        self._scale = scale
        self._image_buffer = image_buffer
        self._render_buffer = render_buffer
        self._double_cover_test = double_cover_test
        self._fill_test = fill_test
        self._tolerance = tolerance
        self._simplices = []
        self._simplices_map = []

    def _push_simplex(self, simplex):
        self._simplices.append(simplex)
        self._simplices_map.append(None)

    def _test_simplex(self, point, i):
        simplex = self._simplices[i]
        simplex_map = self._simplices_map[i]
        anchor = simplex[2:3, :]
        if (simplex_map is None):
            simplex_map = np.linalg.inv(simplex[0:2, :] - anchor)
            self._simplices_map[i] = simplex_map
        ab = (point - anchor) @ simplex_map
        abc = np.hstack((ab, 1 - ab[:, 0:1] - ab[:, 1:2]))
        return np.all(abc > -self._tolerance)

    def _test_double_cover(self, vxd):
        if (self._double_cover_test):
            for i in range(0, len(self._simplices)):
                double_cover = self._test_simplex(vxd[:, 0:2], len(self._simplices) - 1 - i)
                if (double_cover):
                    return True
        return False

    def _bootstrap(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        self._align_axis = np.array([[0, 1, 0]], face_normal.dtype)
        self._uvx_normal = np.array([[0, 0, 1]], face_normal.dtype)

        self._image_uvx = np.ones_like(mesh_vertices)

        vps = origin
        vxs = mesh_vertices[indices_vertices, :]

        vpd = np.array([[self._image_buffer.shape[1] // 2, self._image_buffer.shape[0] // 2, 0]], mesh_vertices.dtype)

        align_outward = geometry_solve_basis(self._align_prior, face_normal, self._align_axis * self._scale, self._uvx_normal)
        align_simplex = cv2.Rodrigues(self._uvx_normal * -self._angle)[0]
        
        vxd = (((vxs - vps) @ align_outward) @ align_simplex) + vpd
        vxd[:, 2] = 0

        self._push_simplex(vxd[:, 0:2])
        self._image_uvx[indices_uvx, :] = vxd

        return mesh_neighborhood_processor_command.EXPAND

    def _unwrap(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        unwrapped = self._image_uvx[indices_uvx, 2] == 0
        unwrapped_count = unwrapped.sum()

        if (unwrapped_count <= 1):
            return mesh_neighborhood_processor_command.CONTINUE
        
        if (unwrapped_count >= 3):
            return mesh_neighborhood_processor_command.EXPAND

        unwrapped_indices = [1, 2, 0] if (not unwrapped[0]) else [2, 0, 1] if (not unwrapped[1]) else [0, 1, 2]

        vips_a, viqs_a, vixs_a = indices_uvx[unwrapped_indices]
        vips_b, viqs_b, vixs_b = indices_vertices[unwrapped_indices]

        vps = mesh_vertices[np.newaxis, vips_b, :]
        vqs = mesh_vertices[np.newaxis, viqs_b, :]
        vxs = mesh_vertices[np.newaxis, vixs_b, :]

        vpd = self._image_uvx[np.newaxis, vips_a, :]
        vqd = self._image_uvx[np.newaxis, viqs_a, :]

        align_outward = geometry_solve_basis(vqs - vps, face_normal, vqd - vpd, self._uvx_normal)

        vxd = ((vxs - vps) @ align_outward) + vpd
        vxd[:, 2] = 0

        double_cover = self._test_double_cover(vxd)

        if (double_cover):
            return mesh_neighborhood_processor_command.IGNORE

        self._push_simplex(np.vstack((vxd[:, 0:2], vqd[:, 0:2], vpd[:, 0:2])))
        self._image_uvx[np.newaxis, vixs_a, :] = vxd

        return mesh_neighborhood_processor_command.EXPAND

    def _blit(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        pixels_src = texture_uvx_invert(weights_src @ self._image_uvx[indices_uvx, 0:2], self._image_buffer.shape, 1)
        mask = texture_test_inside(self._image_buffer, pixels_src[:, 0], pixels_src[:, 1])
        pixels_painted = np.count_nonzero(mask)
        dst = pixels_dst[mask, :]
        src = pixels_src[mask, :]
        if (pixels_painted > 0):
            self._render_buffer[dst[:, 1], dst[:, 0], :] = texture_read(self._image_buffer, src[:, 0], src[:, 1])
        cc = pixels_painted > int(self._fill_test * pixels_dst.shape[0])
        command = mesh_neighborhood_processor_command.EXPAND if (cc) else mesh_neighborhood_processor_command.IGNORE
        return (command, dst, src) # tuple return

    def paint(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        call = self._blit if ((pixels_dst is not None) and (weights_src is not None)) else self._unwrap if (level > 0) else self._bootstrap
        return call(mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level)


def painter_create_color(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, color, tolerance=0, fixed=False):
    mno = mesh_neighborhood_operation_color(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_uvx, color, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint) if (not fixed) else mesh_list_processor(face_index, mno.paint)
    return mnp


def painter_create_brush(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, brush, tolerance=0):
    mno = mesh_neighborhood_operation_brush(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_uvx, origin, brush, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


def painter_create_decal(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, decal, tolerance=0):
    mno = mesh_neighborhood_operation_decal(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_b.face_normals, mesh_uvx, uv_transform, origin, decal, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


class mesh_paint_descriptor:
    def __init__(self, resource_id, layer_id, task_id):
        self.resource_id = resource_id
        self.layer_id = layer_id
        self.task_id = task_id


class mesh_paint_result:
    def __init__(self, done, status, data, layer_id):
        self.done = done
        self.status = status
        self.data = data
        self.layer_id = layer_id


class mesh_paint:
    def __init__(self, uvx, render_target, uv_transform, background):
        self._uvx = uvx
        self._render_target = render_target
        self._uv_transform = uv_transform
        self._background = background
        self._layers = dict()
        self._layer_enable = dict()
        self._colors = dict()
        self._textures = dict()
        self._brushes = dict()
        self._decals = dict()
        self._tasks = dict()

    def set_background(self, background):
        self._background = background

    def layer_create(self, layer_id):
        self._layers[layer_id] = np.zeros_like(self._render_target)
        self._layer_enable[layer_id] = False

    def layer_enable(self, layer_id, enable):
        self._layer_enable[layer_id] = enable

    def layer_clear(self, layer_id, color=0):
        self._layers[layer_id][:, :, :] = color

    def layer_erase(self, layer_id, pixels, color=0):
        self._layers[layer_id][pixels[:, 1], pixels[:, 0], :] = color

    def layer_delete(self, layer_id):
        self._layers.pop(layer_id)
        self._layer_enable.pop(layer_id)

    def texture_attach(self, texture_id, texture):
        self._textures[texture_id] = texture

    def texture_detach(self, texture_id):
        self._textures.pop(texture_id)

    def color_create_solid(self, color_id, color, stop_level, layer_id):
        self._colors[color_id] = paint_color_solid(color, stop_level, self._layers[layer_id])

    def color_delete(self, color_id):
        self._colors.pop(color_id)

    def brush_create_solid(self, brush_id, size, color, layer_id, fill_test=0.0):
        self._brushes[brush_id] = paint_brush_solid(size, color, self._layers[layer_id], fill_test)

    def brush_create_gradient(self, brush_id, size, color_center, color_edge, hardness, layer_id, fill_test=0.0):
        self._brushes[brush_id] = paint_brush_gradient(size, color_center, color_edge, hardness, self._layers[layer_id], fill_test)

    def brush_delete(self, brush_id):
        self._brushes.pop(brush_id)

    def decal_create_solid(self, decal_id, align_prior, angle, scale, texture_id, layer_id, double_cover_test=True, fill_test=0.0, tolerance=0):
        self._decals[decal_id] = paint_decal_solid(align_prior, angle, scale, self._textures[texture_id], self._layers[layer_id], double_cover_test, fill_test, tolerance)

    def decal_delete(self, decal_id):
        self._decals.pop(decal_id)

    def task_create_paint_color(self, task_id, mesh_a, mesh_b, face_index, origin, color_idx, tolerance=0, fixed=False):
        o = self._colors[color_idx].paint
        self._tasks[task_id] = painter_create_color(mesh_a, mesh_b, self._uvx, self._uv_transform, face_index, origin, o, tolerance, fixed)

    def task_create_paint_brush(self, task_id, mesh_a, mesh_b, face_index, origin, brush_idx, tolerance=0):
        o = self._brushes[brush_idx].paint
        self._tasks[task_id] = painter_create_brush(mesh_a, mesh_b, self._uvx, self._uv_transform, face_index, origin, o, tolerance)
        
    def task_create_paint_decal(self, task_id, mesh_a, mesh_b, face_index, origin, decal_idx, tolerance=0):
        o = self._decals[decal_idx].paint
        self._tasks[task_id] = painter_create_decal(mesh_a, mesh_b, self._uvx, self._uv_transform, face_index, origin, o, tolerance)

    def task_execute(self, task_id, timeout, steps=1):
        return self._tasks[task_id].invoke_timeslice(timeout, steps)
    
    def task_status(self, task_id):
        return self._tasks[task_id].status()

    def task_done(self, task_id):
        return self._tasks[task_id].done()
    
    def task_delete(self, task_id):
        self._tasks.pop(task_id)

    def clear(self, enabled_only=False, color=0):
        for key in self._layers.keys():
            if ((not enabled_only) or self._layer_enable[key]):
                self.layer_clear(key, color)

    def flush(self, force_alpha=None, stencil_layer=None):
        self._render_target[:, :, :] = self._background
        for key in sorted(self._layers.keys()):
            if (self._layer_enable[key]):
                self._render_target[:, :, :] = np.array(Image.alpha_composite(Image.fromarray(self._render_target), Image.fromarray(self._layers[key])))
        if (force_alpha is not None):
            self._render_target[:, :, 3] = force_alpha
        if (stencil_layer is not None):
            self._render_target[:, :, 3] = self._layers[stencil_layer][:, :, 3]


class mesh_paint_single_pass(mesh_paint):
    def __init__(self, uvx, render_target, uv_transform, background):
        super().__init__(uvx, render_target, uv_transform, background)

    def paint_color_solid(self, mesh_a, mesh_b, face_index, point, color, stop_level, tolerance=0, fixed=False, layer_id=0, timeout=0.05, steps=1):
        d = mesh_paint_descriptor(0, layer_id, 0)
        self.color_create_solid(d.resource_id, color, stop_level, d.layer_id)
        self.task_create_paint_color(d.task_id, mesh_a, mesh_b, face_index, point, d.resource_id, tolerance, fixed)
        data = self.task_execute(d.task_id, timeout, steps)
        done = self.task_done(d.task_id)
        status = self.task_status(d.task_id)
        self.task_delete(d.task_id)
        self.color_delete(d.resource_id)
        return mesh_paint_result(done, status, data, layer_id)
    
    def paint_brush_solid(self, mesh_a, mesh_b, face_index, point, size, color, fill_test=0.0, tolerance=0, layer_id=0, timeout=0.05, steps=1):
        d = mesh_paint_descriptor(0, layer_id, 1)
        self.brush_create_solid(d.resource_id, size, color, d.layer_id, fill_test)
        self.task_create_paint_brush(d.task_id, mesh_a, mesh_b, face_index, point, d.resource_id, tolerance)
        data = self.task_execute(d.task_id, timeout, steps)
        done = self.task_done(d.task_id)
        status = self.task_status(d.task_id)
        self.task_delete(d.task_id)
        self.brush_delete(d.resource_id)
        return mesh_paint_result(done, status, data, layer_id)
    
    def paint_brush_gradient(self, mesh_a, mesh_b, face_index, point, size, color_center, color_edge, hardness, fill_test=0.0, tolerance=0, layer_id=0, timeout=0.05, steps=1):
        d = mesh_paint_descriptor(1, layer_id, 2)
        self.brush_create_gradient(d.resource_id, size, color_center, color_edge, hardness, d.layer_id, fill_test)
        self.task_create_paint_brush(d.task_id, mesh_a, mesh_b, face_index, point, d.resource_id, tolerance)
        data = self.task_execute(d.task_id, timeout, steps)
        done = self.task_done(d.task_id)
        status = self.task_status(d.task_id)
        self.task_delete(d.task_id)
        self.brush_delete(d.resource_id)
        return mesh_paint_result(done, status, data, layer_id)
    
    def paint_decal_solid(self, mesh_a, mesh_b, face_index, point, decal, align_prior, angle, scale, double_cover_test=True, fill_test=0.0, tolerance_decal=0, tolerance_paint=0, layer_id=0, timeout=0.05, steps=1):
        d = mesh_paint_descriptor(0, layer_id, 3)
        self.texture_attach(d.resource_id, decal)
        self.decal_create_solid(d.resource_id, align_prior, angle, scale, d.resource_id, d.layer_id, double_cover_test, fill_test, tolerance_decal)
        self.task_create_paint_decal(d.task_id, mesh_a, mesh_b, face_index, point, d.resource_id, tolerance_paint)
        data = self.task_execute(d.task_id, timeout, steps)
        done = self.task_done(d.task_id)
        status = self.task_status(d.task_id)
        self.task_delete(d.task_id)
        self.decal_delete(d.resource_id)
        self.texture_detach(d.resource_id)
        return mesh_paint_result(done, status, data, layer_id)


#------------------------------------------------------------------------------
# Mesh Chart
#------------------------------------------------------------------------------

class mesh_chart_point:
    def __init__(self, point, face_index, position, direction, orientation):
        self.point = point
        self.face_index = face_index
        self.position = position
        self.direction = direction
        self.orientation = orientation


class mesh_chart_local:
    def __init__(self, p1, p2, offset, nx, ny, nz, xz, nxz):
        self.p1 = p1
        self.p2 = p2
        self.offset = offset
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.xz = xz
        self.nxz = nxz


class mesh_chart_frame:
    def __init__(self, left, up, front, center, length, points):
        self.left = left
        self.up = up
        self.front = front
        self.center = center
        self.length = length
        self.points = points

    def _decompose(self, point):
        offset = point - self.center
        ny = offset @ self.up.T
        xz = offset - ny * self.up
        nxz = np.linalg.norm(xz)
        nx = self.left @ xz.T
        nz = self.front @ xz.T
        return (offset, nx, ny, nz, xz, nxz) # tuple return

    def from_cylindrical(self, mesh, displacement, yaw):
        position = self.center + displacement * self.up
        orientation = cv2.Rodrigues(self.up * -yaw)[0]
        direction = self.front @ orientation
        point, face_index = mesh_raycast(mesh, position, direction)
        return mesh_chart_point(point, face_index, position, direction, orientation)
    
    def from_spherical(self, mesh, yaw, pitch):
        position = self.center
        orientation = cv2.Rodrigues(self.left * pitch)[0] @ cv2.Rodrigues(self.up * -yaw)[0]
        direction = self.front @ orientation
        point, face_index = mesh_raycast(mesh, position, direction)
        return mesh_chart_point(point, face_index, position, direction, orientation)
    
    def to_cylindrical(self, point):
        offset, nx, ny, nz, xz, nxz = self._decompose(point)
        displacement = ny
        yaw = np.arctan2(nx, nz)
        return mesh_chart_local(displacement, yaw, offset, nx, ny, nz, xz, nxz)

    def to_spherical(self, point):
        offset, nx, ny, nz, xz, nxz = self._decompose(point)
        yaw = np.arctan2(nx, nz)
        pitch = np.arctan2(ny, nxz)
        return mesh_chart_local(yaw, pitch, offset, nx, ny, nz, xz, nxz)

    def to_pose(self):
        pose = np.eye(4, dtype=self.center.dtype)
        pose[0:1, :3] = self.left
        pose[1:2, :3] = self.up
        pose[2:3, :3] = self.front
        pose[3:4, :3] = self.center
        return pose


class mesh_chart:
    def __init__(self, mesh):
        self._mesh = mesh
        self._cache = dict()

    def create_frame(self, region):
        frame = self._cache.get(region, None)
        if (frame is None):
            frame = operator.methodcaller('_create_frame_' + region)(self)
            self._cache[region] = frame
        return frame
    
    def from_cylindrical(self, frame, displacement, yaw):
        return frame.from_cylindrical(self._mesh, displacement, yaw)
    
    def from_spherical(self, frame, yaw, pitch):
        return frame.from_spherical(self._mesh, yaw, pitch)

    def to_cylindrical(self, frame, point):
        return frame.to_cylindrical(point)

    def to_spherical(self, frame, point):
        return frame.to_spherical(point)
    
    def to_pose(self, frame):
        return frame.to_pose()


#------------------------------------------------------------------------------
# SMPL Chart
#------------------------------------------------------------------------------

def smpl_camera_align_It(K_smpl, K_dst, points_world):
    n = points_world.shape[0]
    K = K_smpl @ np.linalg.inv(K_dst)
    b = points_world @ (K - np.eye(3, dtype=points_world.dtype))
    a = (points_world / points_world[:, 2:3]) @ K
    f = np.ones((n, 1), dtype=points_world.dtype)
    z = np.zeros((n, 1), dtype=points_world.dtype)
    e = np.vstack((np.hstack((f, z, -a[:, 0:1])), np.hstack((z, f, -a[:, 1:2]))))
    s = np.vstack((b[:, 0:1], b[:, 1:2]))
    t, res, rank, sv = np.linalg.lstsq(e, s)
    return (np.eye(3, dtype=points_world.dtype), t.T) # tuple return


def smpl_camera_align_Rt(K_smpl, K_dst, points_world):
    u = np.ascontiguousarray(((points_world / points_world[:, 2:3]) @ K_smpl)[:, 0:2])
    ok, r, t = cv2.solvePnP(points_world, u, K_dst.T, None, flags=cv2.SOLVEPNP_SQPNP)
    R = cv2.Rodrigues(r)[0]
    return (R.T, t.T) # tuple return


def smpl_camera_align_dz(K_smpl, K_dst, points_world):
    K = K_smpl @ np.linalg.inv(K_dst)
    s = (K[0, 0] + K[1, 1]) / 2
    return ((1 / s) * K, 0) # tuple return


class smpl_joints:
    pelvis = 0
    left_hip = 1
    right_hip = 2
    spine_1 = 3
    left_knee = 4
    right_knee = 5
    spine_2 = 6
    left_ankle = 7
    right_ankle = 8
    spine_3 = 9
    left_foot = 10
    right_foot = 11
    neck = 12
    left_collar = 13
    right_collar = 14
    head = 15
    left_shoulder = 16
    right_shoulder = 17
    left_elbow = 18
    right_elbow = 19
    left_wrist = 20
    right_wrist = 21
    left_hand = 22
    right_hand = 23


class smpl_joints_x1:
    nose = 24
    right_eye = 25
    left_eye = 26
    right_ear = 27
    left_ear = 28
    left_big_toe = 29
    left_small_toe = 30
    left_heel = 31
    right_big_toe = 32
    right_small_toe = 33
    right_heel = 34
    left_thumb = 35
    left_index = 36
    left_middle = 37
    left_ring = 38
    left_pinky = 39
    right_thumb = 40
    right_index = 41
    right_middle = 42
    right_ring = 43
    right_pinky = 44


class smpl_joints_x2:
    right_hip = 45
    left_hip = 46
    lsp_neck = 47
    lsp_top_of_head = 48
    mpii_pelvis = 49
    mpii_thorax = 50
    h36m_spine = 51
    h36m_jaw = 52
    h36m_head = 53


class smpl_joints_openpose:
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    MidHip = 8
    RHip = 9
    RKnee = 10
    RAnkle = 11
    LHip = 12
    LKnee = 13
    LAnkle = 14
    REye = 15
    LEye = 16
    REar = 17
    LEar = 18
    LBigToe = 19
    LSmallToe = 20
    LHeel = 21
    RBigToe = 22
    RSmallToe = 23
    RHeel = 24


class smpl_mesh_chart_openpose(mesh_chart):
    def __init__(self, mesh, joints):
        super().__init__(mesh)
        self._joints = joints

    def _template_frame_foot(self, bigtoe, smalltoe, ankle, heel):
        left  = np.cross(ankle - heel, bigtoe - ankle)
        front = np.cross(left, ankle - smalltoe)
        up    = np.cross(front, left)

        left  = math_normalize(left)[0]
        front = math_normalize(front)[0]
        up    = math_normalize(up)[0]

        center = (ankle + smalltoe) * 0.5
        length = np.linalg.norm(ankle - smalltoe)
        points = np.vstack((bigtoe, smalltoe, ankle, heel))

        return mesh_chart_frame(left, up, front, center, length, points)
    
    def _create_frame_foot_left(self):
        bigtoe   = self._joints[np.newaxis, smpl_joints_openpose.LBigToe, :]
        smalltoe = self._joints[np.newaxis, smpl_joints_openpose.LSmallToe, :]
        ankle    = self._joints[np.newaxis, smpl_joints_openpose.LAnkle, :]
        heel     = self._joints[np.newaxis, smpl_joints_openpose.LHeel, :]

        return self._template_frame_foot(bigtoe, smalltoe, ankle, heel)
    
    def _create_frame_foot_right(self):
        bigtoe   = self._joints[np.newaxis, smpl_joints_openpose.RBigToe, :]
        smalltoe = self._joints[np.newaxis, smpl_joints_openpose.RSmallToe, :]
        ankle    = self._joints[np.newaxis, smpl_joints_openpose.RAnkle, :]
        heel     = self._joints[np.newaxis, smpl_joints_openpose.RHeel, :]

        return self._template_frame_foot(bigtoe, smalltoe, ankle, heel)

    def _template_frame_lower_leg(self, bigtoe, ankle, knee):
        up    = knee - ankle
        left  = np.cross(up, bigtoe - ankle)
        front = np.cross(left, up)

        up    = math_normalize(up)[0]
        left  = math_normalize(left)[0]
        front = math_normalize(front)[0]

        center = (ankle + knee) * 0.5
        length = np.linalg.norm(ankle - knee)
        points = np.vstack((bigtoe, ankle, knee))

        return mesh_chart_frame(left, up, front, center, length, points)

    def _create_frame_lower_leg_left(self):
        bigtoe = self._joints[np.newaxis, smpl_joints_openpose.LBigToe, :]
        ankle  = self._joints[np.newaxis, smpl_joints_openpose.LAnkle, :]
        knee   = self._joints[np.newaxis, smpl_joints_openpose.LKnee, :]

        return self._template_frame_lower_leg(bigtoe, ankle, knee)
    
    def _create_frame_lower_leg_right(self):
        bigtoe = self._joints[np.newaxis, smpl_joints_openpose.RBigToe, :]
        ankle  = self._joints[np.newaxis, smpl_joints_openpose.RAnkle, :]
        knee   = self._joints[np.newaxis, smpl_joints_openpose.RKnee, :]

        return self._template_frame_lower_leg(bigtoe, ankle, knee)

    def _template_frame_thigh(self, ankle, knee, hip):
        up    = hip - knee
        left  = np.cross(up, knee - ankle)
        front = np.cross(left, up)

        up    = math_normalize(up)[0]
        left  = math_normalize(left)[0]
        front = math_normalize(front)[0]

        center = (hip + knee) * 0.5
        length = np.linalg.norm(hip - knee)
        points = np.vstack((knee, hip))

        return mesh_chart_frame(left, up, front, center, length, points)
    
    def _create_frame_thigh_left(self):
        ankle = self._joints[np.newaxis, smpl_joints_openpose.LAnkle, :]
        knee  = self._joints[np.newaxis, smpl_joints_openpose.LKnee, :]
        hip   = self._joints[np.newaxis, smpl_joints_openpose.LHip, :]

        return self._template_frame_thigh(ankle, knee, hip)

    def _create_frame_thigh_right(self):
        ankle = self._joints[np.newaxis, smpl_joints_openpose.RAnkle, :]
        knee  = self._joints[np.newaxis, smpl_joints_openpose.RKnee, :]
        hip   = self._joints[np.newaxis, smpl_joints_openpose.RHip, :]

        return self._template_frame_thigh(ankle, knee, hip)
    
    def _template_frame_body(self, lhip, mhip, rhip, neck):
        left  = lhip - rhip
        front = np.cross(left, neck - mhip)
        up    = np.cross(front, left)

        up    = math_normalize(up)[0]
        left  = math_normalize(left)[0]
        front = math_normalize(front)[0]

        center = (mhip + neck) * 0.5
        length = np.linalg.norm(mhip - neck)
        points = np.vstack((lhip, mhip, rhip, neck))
        
        return mesh_chart_frame(left, up, front, center, length, points)
    
    def _create_frame_body_center(self):
        lhip = self._joints[np.newaxis, smpl_joints_openpose.LHip, :]
        mhip = self._joints[np.newaxis, smpl_joints_openpose.MidHip, :]
        rhip = self._joints[np.newaxis, smpl_joints_openpose.RHip, :]
        neck = self._joints[np.newaxis, smpl_joints_openpose.Neck, :]

        return self._template_frame_body(lhip, mhip, rhip, neck)

    def _template_frame_head(self, lear, rear, neck, nose):
        left  = lear - rear
        up    = lear - neck
        front = np.cross(left, up)
        up    = np.cross(front, left)

        left  = math_normalize(left)[0]
        up    = math_normalize(up)[0]
        front = math_normalize(front)[0]

        center = (nose + lear + rear) / 3
        length = np.linalg.norm(neck - nose)
        points = np.vstack((lear, rear, neck, nose))

        return mesh_chart_frame(left, up, front, center, length, points)
    
    def _create_frame_head_center(self):
        lear = self._joints[np.newaxis, smpl_joints_openpose.LEar, :]
        rear = self._joints[np.newaxis, smpl_joints_openpose.REar, :]
        neck = self._joints[np.newaxis, smpl_joints_openpose.Neck, :]
        nose = self._joints[np.newaxis, smpl_joints_openpose.Nose, :]

        return self._template_frame_head(lear, rear, neck, nose)
    
    def _template_frame_upper_arm(self, wrist, elbow, shoulder):
        up    = shoulder - elbow
        left  = np.cross(up, wrist - elbow)
        front = np.cross(left, up)

        left  = math_normalize(left)[0]
        up    = math_normalize(up)[0]
        front = math_normalize(front)[0]

        center = (elbow + shoulder) * 0.5
        length = np.linalg.norm(elbow - shoulder)
        points = np.vstack((shoulder, elbow))

        return mesh_chart_frame(left, up, front, center, length, points)

    def _create_frame_upper_arm_left(self):
        wrist    = self._joints[np.newaxis, smpl_joints_openpose.LWrist, :]
        elbow    = self._joints[np.newaxis, smpl_joints_openpose.LElbow, :]
        shoulder = self._joints[np.newaxis, smpl_joints_openpose.LShoulder, :]

        return self._template_frame_upper_arm(wrist, elbow, shoulder)

    def _create_frame_upper_arm_right(self):
        wrist    = self._joints[np.newaxis, smpl_joints_openpose.RWrist, :]
        elbow    = self._joints[np.newaxis, smpl_joints_openpose.RElbow, :]
        shoulder = self._joints[np.newaxis, smpl_joints_openpose.RShoulder, :]

        return self._template_frame_upper_arm(wrist, elbow, shoulder)

    def _template_frame_lower_arm(self, wrist, elbow, shoulder):
        up = elbow - wrist
        left = np.cross(up, shoulder - elbow)
        front = np.cross(left, up)

        left  = math_normalize(left)[0]
        up    = math_normalize(up)[0]
        front = math_normalize(front)[0]

        center = (elbow + wrist) * 0.5
        length = np.linalg.norm(elbow - wrist)
        points = np.vstack((wrist, elbow))

        return mesh_chart_frame(left, up, front, center, length, points)

    def _create_frame_lower_arm_left(self):
        wrist    = self._joints[np.newaxis, smpl_joints_openpose.LWrist, :]
        elbow    = self._joints[np.newaxis, smpl_joints_openpose.LElbow, :]
        shoulder = self._joints[np.newaxis, smpl_joints_openpose.LShoulder, :]

        return self._template_frame_lower_arm(wrist, elbow, shoulder)

    def _create_frame_lower_arm_right(self):
        wrist    = self._joints[np.newaxis, smpl_joints_openpose.RWrist, :]
        elbow    = self._joints[np.newaxis, smpl_joints_openpose.RElbow, :]
        shoulder = self._joints[np.newaxis, smpl_joints_openpose.RShoulder, :]

        return self._template_frame_lower_arm(wrist, elbow, shoulder)


class smpl_model_result:
    def __init__(self, vertices, faces, joints):
        self.vertices = vertices
        self.faces = faces
        self.joints = joints


class smpl_model:
    _SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

    def __init__(self, model_path, num_betas, device):
        self._smpl_model = smplx.SMPLLayer(model_path=model_path, num_betas=num_betas).to(device)
        self._smpl_to_open_pose = torch.tensor(smpl_model._SMPL_TO_OPENPOSE, dtype=torch.long, device=device)

    def to_mesh(self, smpl_params, openpose_joints=True):
        smpl_output = self._smpl_model(**smpl_params)
        vertices = smpl_output.vertices
        joints = smpl_output.joints.index_select(1, self._smpl_to_open_pose) if (openpose_joints) else smpl_output.joints
        return smpl_model_result(vertices.cpu().numpy(), self._smpl_model.faces, joints.cpu().numpy())
    

class smpl_model_filtered(smpl_model):
    def __init__(self, model_path, num_betas, device, weight=0.10):
        super().__init__(model_path, num_betas, device)
        self._alpha = torch.scalar_tensor(weight, dtype=torch.float32, device=device)
        self._l = False
        self._g = None
        self._p = None
        self._b = None
        self._t = None

    def set_weight(self, weight):
        self._alpha = torch.scalar_tensor(weight, dtype=self._alpha.dtype, device=self._alpha.device)        

    def to_mesh(self, smpl_params, openpose_joints=True):
        global_orient = smpl_params['global_orient'][0:1] # n, 1, 3, 3
        body_pose = smpl_params['body_pose'][0:1] # n 23 3 3
        betas = smpl_params['betas'][0:1] # n 10
        transl = smpl_params['transl'][0:1] # n 3

        if (not self._l):
            self._g = global_orient
            self._p = body_pose
            self._b = betas
            self._t = transl
            self._l = True
            self._prev_p = body_pose
            #self._dg = 0.5
            #print(self._g.shape)
            #print(self._p.shape)
            #print(self._t.shape)
            #print(self._b.shape)
            #print('END DEF')
        else:
            dg = torch.min((np.pi - roma.rotmat_geodesic_distance(self._prev_p, body_pose)) / np.pi)
            dg = dg * dg
            self._prev_p = body_pose
            #print(global_orient.shape)
            #print(body_pose.shape)
            #print(transl.shape)
            #print(betas.shape)
            #self._dg = self._dg + self._alpha * ( - self._dg)
            
            #print(dg)






            self._g = roma.rotmat_slerp(self._g, global_orient, self._alpha)
            self._p = roma.rotmat_slerp(self._p, body_pose, dg * self._alpha) #self._alpha)
            print(self._p.shape)
            self._t = self._t + self._alpha * (transl - self._t)
            self._b = self._b + self._alpha * (betas - self._b) # linear??
            #print(self._g.shape)
            #print(self._p.shape)
            #print(self._t.shape)
            #print(self._b.shape)


        smpl_params['global_orient'] = self._g
        smpl_params['body_pose'] = self._p
        smpl_params['betas'] = self._b
        smpl_params['transl'] = self._t

        return super().to_mesh(smpl_params, openpose_joints)



#------------------------------------------------------------------------------
# Rendering Components
#------------------------------------------------------------------------------

def renderer_create_settings_offscreen(width, height, point_size=1):
    s = dict()
    s['viewport_width'] = width
    s['viewport_height'] = height
    s['point_size'] = point_size
    return s


def renderer_create_settings_scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.0, 0.0, 0.0), name='scene'):
    s = dict()
    s['bg_color'] = bg_color
    s['ambient_light'] = ambient_light
    s['name'] = name
    return s


def renderer_create_settings_camera(fx, fy, cx, cy, znear=0.05, zfar=100, name='camera'):
    s = dict()
    s['fx'] = fx
    s['fy'] = fy
    s['cx'] = cx
    s['cy'] = cy
    s['znear'] = znear
    s['zfar'] = zfar
    s['name'] = name
    return s


def renderer_create_settings_lamp(color=(1.0, 1.0, 1.0), intensity=3.0, name='lamp'):
    s = dict()
    s['color'] = color
    s['intensity'] = intensity
    s['name'] = name
    return s


def renderer_create_settings_camera_transform(center=(0, 0, 0), yaw=0, pitch=0, distance=1, min_pitch=-75, max_pitch=75, znear=0.05, zfar=100):
    s = dict()
    s['center'] = np.array(center, np.float32)
    s['yaw'] = yaw
    s['pitch'] = pitch
    s['distance'] = distance
    s['min_pitch'] = min_pitch
    s['max_pitch'] = max_pitch
    s['znear'] = znear
    s['zfar'] = zfar
    return s


class renderer_camera_transform_parameters:
    def __init__(self, yaw, pitch, distance, center, tc, ry, rx, tz):
        self.yaw = yaw
        self.pitch = pitch
        self.distance = distance
        self.center = center
        self.tc = tc
        self.ry = ry
        self.rx = rx
        self.tz = tz


class renderer_camera_transform:
    def __init__(self, center, yaw, pitch, distance, min_pitch, max_pitch, znear, zfar):
        self._min_pitch = min_pitch
        self._max_pitch = max_pitch
        self._znear = znear
        self._zfar = zfar
        self._tz = np.eye(4, dtype=center.dtype)
        self._tc = np.eye(4, dtype=center.dtype)
        self._global_pose = np.eye(4, dtype=center.dtype)
        self._global_x = self._global_pose[:3, 0]
        self._global_y = self._global_pose[:3, 1]
        self._global_z = self._global_pose[:3, 2]

        self.set_center(center)
        self.set_yaw(yaw)
        self.set_pitch(pitch)
        self.set_distance(distance)

    def get_yaw(self):
        return self._yaw

    def set_yaw(self, value):
        self._yaw = value
        self._ry = trimesh.transformations.rotation_matrix(np.radians(self._yaw), self._global_y)
        self._dirty = True

    def update_yaw(self, delta):
        self.set_yaw(self._yaw + delta)

    def get_pitch(self):
        return self._pitch

    def set_pitch(self, value):
        self._pitch = np.clip(value, self._min_pitch, self._max_pitch)
        self._rx = trimesh.transformations.rotation_matrix(np.radians(self._pitch), self._global_x)
        self._dirty = True

    def update_pitch(self, delta):
        self.set_pitch(self._pitch + delta)

    def get_distance(self):
        return self._distance

    def set_distance(self, value):
        self._distance = np.clip(value, self._znear, self._zfar)
        self._tz[2, 3] = self._distance
        self._dirty = True

    def update_distance(self, delta):
        self.set_distance(self._distance + delta)

    def get_center(self):
        return self._center
    
    def set_center(self, value):
        self._center = value
        self._tc[:3, 3] = self._center
        self._dirty = True

    def update_center(self, delta):
        self.set_center(self._center + delta)

    def adjust_parameters(self, yaw=None, pitch=None, distance=None, center=None, relative=True):
        if (relative):
            if (yaw is not None):
                self.update_yaw(yaw)
            if (pitch is not None):
                self.update_pitch(pitch)
            if (distance is not None):
                self.update_distance(distance)
            if (center is not None):
                self.update_center(center)
        else:
            if (yaw is not None):
                self.set_yaw(yaw)
            if (pitch is not None):
                self.set_pitch(pitch)
            if (distance is not None):
                self.set_distance(distance)
            if (center is not None):
                self.set_center(center)

    def get_matrix_center(self):
        return self._tc

    def get_matrix_yaw(self):
        return self._ry
    
    def get_matrix_pitch(self):
        return self._rx
    
    def get_matrix_distance(self):
        return self._tz
    
    def get_parameters(self):
        return renderer_camera_transform_parameters(self._yaw, self._pitch, self._distance, self._center, self._tc, self._ry, self._rx, self._tz)

    def _update(self):
        if (self._dirty):
            self._local_pose = self._tc @ self._ry @ self._rx @ self._tz
            self._local_x = self._local_pose[:3, 0]
            self._local_y = self._local_pose[:3, 1]
            self._local_z = self._local_pose[:3, 2]
            self._plane_z = np.cross(self._local_x, self._global_y)
            self._plane_pose = np.column_stack((self._local_x, self._global_y, self._plane_z))
            self._dirty = False

    def get_transform_local(self):
        self._update()
        return self._local_pose

    def get_transform_plane(self):
        self._update()
        return self._plane_pose
    
    def move_center(self, delta_xyz, plane=True):
        self._update()
        pose = self._local_pose if (not plane) else self._plane_pose
        center = delta_xyz[0] * pose[:3, 0] + delta_xyz[1] * pose[:3, 1] + delta_xyz[2] * pose[:3, 2]
        self.update_center(center)


class renderer_mesh_identifier:
    def __init__(self, group, name, kind):
        self.group = group
        self.name = name
        self.kind = kind


class renderer_scene_control:
    def __init__(self, settings_offscreen, settings_scene, settings_camera, settings_camera_transform, settings_lamp):
        self._renderer = pyrender.OffscreenRenderer(**settings_offscreen)
        self._scene = pyrender.Scene(**settings_scene)
        self._camera = pyrender.IntrinsicsCamera(**settings_camera)
        self._camera_transform = renderer_camera_transform(**settings_camera_transform)
        self._light = pyrender.DirectionalLight(**settings_lamp)
        self._groups = dict()
        self._camera_pose = self._camera_transform.get_transform_local()

        self._node_camera = self._scene.add(self._camera, 'internal@main@camera', self._camera_pose)
        self._node_light = self._scene.add(self._light, 'internal@main@lamp', self._camera_pose)

        self._kf = np.array([[self._camera.fx, self._camera.fy]], self._camera_pose.dtype)
        self._kc = np.array([[self._camera.cx, self._camera.cy]], self._camera_pose.dtype)

    def _camera_set_pose(self, camera_pose):
        self._camera_pose = camera_pose

        self._scene.set_pose(self._node_camera, self._camera_pose)
        self._scene.set_pose(self._node_light, self._camera_pose)

    def _camera_update_pose(self):
        pose = self._camera_transform.get_transform_local()
        self._camera_set_pose(pose)

    def camera_get_pose(self):
        return self._camera_pose

    def camera_get_projection_matrix(self):
        return self._camera.get_projection_matrix(self._renderer.viewport_width, self._renderer.viewport_height)
    
    def camera_get_transform_plane(self):
        return self._camera_transform.get_transform_plane()
    
    def camera_get_transform_local(self):
        return self._camera_transform.get_transform_local()
    
    def camera_get_parameters(self):
        return self._camera_transform.get_parameters()
    
    def camera_adjust_parameters(self, yaw=None, pitch=None, distance=None, center=None, relative=True):
        self._camera_transform.adjust_parameters(yaw, pitch, distance, center, relative)
        self._camera_update_pose()

    def camera_move_center(self, delta_xyz, plane=True):
        self._camera_transform.move_center(delta_xyz, plane)
        self._camera_update_pose()

    def camera_solve_fov_z(self, center, points, plane=False):
        pose = self._camera_transform.get_transform_local() if (not plane) else self._camera_transform.get_transform_plane()
        x = pose[:3, 0:1].T
        y = pose[:3, 1:2].T
        z = pose[:3, 2:3].T
        wz = geometry_solve_fov_z(self._renderer.viewport_width, self._renderer.viewport_height, self._camera.fx, self._camera.fy, self._camera.cx, self._camera.cy, x, y, z, center, points)
        return wz

    def camera_project_points(self, points, convention=(1, -1, -1)):
        q = math_transform_points(points, self._camera_pose.T, True)
        c = np.column_stack((convention[0] * q[:, 0], convention[1] * q[:, 1], convention[2] * q[:, 2]))
        r = (c[:, 0:2] / c[:, 2:3]) * self._kf + self._kc
        return (r, c, q) # tuple return

    def render(self):
        color, depth = self._renderer.render(self._scene, pyrender.RenderFlags.RGBA)
        return (color, depth) # tuple return

    def group_item_add(self, group, name, item, pose=None):
        nodes = self._groups.get(group, None)
        if (nodes is None):
            nodes = dict()
            self._groups[group] = nodes
        previous = nodes.get(name, None)
        if (previous is not None):
            self._scene.remove_node(previous)
        nodes[name] = self._scene.add(item, 'external@' + group + '@' + name, pose)
        return renderer_mesh_identifier(group, name, 'external')

    def group_item_remove(self, item_id):
        nodes = self._groups.get(item_id.group, None)
        if (nodes is not None):
            item = nodes.get(item_id.name, None)
            if (item is not None):
                self._scene.remove_node(item)

    def group_item_set_pose(self, item_id, pose):
        nodes = self._groups.get(item_id.group, None)
        if (nodes is not None):
            item = nodes.get(item_id.name, None)
            if (item is not None):
                self._scene.set_pose(item, pose)

    def group_item_get_pose(self, item_id):
        nodes = self._groups.get(item_id.group, None)
        if (nodes is not None):
            item = nodes.get(item_id.name, None)
            if (item is not None):
                return self._scene.get_pose(item)
        return None

    def group_clear(self, group):
        nodes = self._groups.pop(group, None)
        if (nodes is not None):
            for name, item in nodes.items():
                self._scene.remove_node(item)

    def clear(self):
        for nodes in self._groups.values():
            for name, item in nodes.items():
                self._scene.remove_node(item)
        self._groups.clear()











class renderer_mesh_control:
    def __init__(self, filename_uv, texture_shape):
        self._mesh_a_vertices, self._mesh_b_vertices, self._mesh_a_faces, self._mesh_b_faces, self._mesh_a_uv, self._mesh_b_uv, self._uv_transform = texture_load_uv(filename_uv)
        self._mesh_a_uvx = texture_uv_to_uvx(self._mesh_a_uv, texture_shape)
        self._mesh_b_uvx = texture_uv_to_uvx(self._mesh_b_uv, texture_shape)
        self._texture_shape = texture_shape
        self._meshes = dict()
        self._cswvfx = dict()

    def _mesh_add(self, group, name, mesh_a, mesh_b, chart, pose):
        g = self._meshes.get(group, None)
        if (g is None):
            g = dict()
            self._meshes[group] = g
        g[name] = [mesh_a, mesh_b, chart, pose]

    def _tvfx_add(self, group, name, texture):
        g = self._cswvfx.get(group, None)
        if (g is None):
            g = dict()
            self._cswvfx[group] = g
        u = g.get(name, None)
        if (u is None):
            target = np.zeros_like(texture)
            visual = texture_create_visual(self._mesh_b_uv, target)
            effect = mesh_paint_single_pass(self._mesh_b_uvx, target, self._uv_transform, texture)
            effect.layer_create(0)
            effect.layer_enable(0, True)
            g[name] = [visual, effect]
        else:
            visual, effect = u
            effect.set_background(texture)

    def mesh_add_smpl(self, group, name, mesh, joints, texture, pose):
        self._tvfx_add(group, name, texture)
        visual, effect = self._cswvfx[group][name]
        mesh_a = mesh
        mesh_b = mesh_expand(mesh_a, self._uv_transform, self._mesh_b_faces, visual)
        mesh_c = smpl_mesh_chart_openpose(mesh_a, joints)
        self._mesh_add(group, name, mesh_a, mesh_b, mesh_c, pose)
        return renderer_mesh_identifier(group, name, 'smpl')

    def mesh_add_user(self, group, name, mesh, pose):
        self._mesh_add(group, name, mesh, None, None, pose)
        return renderer_mesh_identifier(group, name, 'user')
    
    def mesh_remove_item(self, mesh_id):
        self._meshes[mesh_id.group].pop(mesh_id.name)

    def mesh_remove_group(self, group):
        self._meshes.pop(group)

    def mesh_remove_all(self):
        self._meshes.clear()
    
    def mesh_get_base(self, mesh_id):
        return self._meshes[mesh_id.group][mesh_id.name][0]
    
    def mesh_get_full(self, mesh_id):
        return self._meshes[mesh_id.group][mesh_id.name][1]

    def mesh_set_pose(self, mesh_id, pose):
        self._meshes[mesh_id.group][mesh_id.name][3] = pose

    def mesh_get_pose(self, mesh_id):
        return self._meshes[mesh_id.group][mesh_id.name][3]
    
    def mesh_operation_raycast(self, mesh_id, origin, direction):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        local_origin = math_transform_points(origin, pose.T, True)
        local_direction = math_transform_bearings(direction, pose.T, True)
        point, face_index = mesh_raycast(mesh_a, local_origin, local_direction)
        return mesh_chart_point(point, face_index, local_origin, local_direction, None)

    def mesh_operation_closest(self, mesh_id, origin):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        local_origin = math_transform_points(origin, pose.T, True)
        point, face_index, _, = mesh_closest(mesh_a, local_origin)
        return mesh_chart_point(point, face_index, local_origin, None, None)

    def smpl_chart_create_frame(self, mesh_id, region):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        return chart.create_frame(region)
    
    def smpl_chart_from_cylindrical(self, mesh_id, frame, displacement, yaw): 
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        return chart.from_cylindrical(frame, displacement, yaw)
    
    def smpl_chart_from_spherical(self, mesh_id, frame, yaw, pitch):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        return chart.from_spherical(frame, yaw, pitch)
    
    def smpl_chart_to_cylindrical(self, mesh_id, frame, point):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        local_point = math_transform_points(point, pose.T, True)
        return chart.to_cylindrical(frame, local_point)
    
    def smpl_chart_to_spherical(self, mesh_id, frame, point):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        local_point = math_transform_points(point, pose.T, True)
        return chart.to_spherical(frame, local_point)
    
    def smpl_chart_to_pose(self, mesh_id, frame):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        return chart.to_pose(frame)
    
    def smpl_paint_set_background(self, mesh_id, background):
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.set_background(background)
    
    def smpl_paint_layer_create(self, mesh_id, layer_id):
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.layer_create(layer_id)
    
    def smpl_paint_layer_enable(self, mesh_id, layer_id, enable):
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.layer_enable(layer_id, enable)

    def smpl_paint_layer_clear(self, mesh_id, layer_id, color=0):
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.layer_clear(layer_id, color)

    def smpl_paint_layer_erase(self, mesh_id, layer_id, data, color=0):
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        pixels = np.vstack([p[1] for p in data if p[1] is not None])
        effect.layer_erase(layer_id, pixels, color)

    def smpl_paint_layer_delete(self, mesh_id, layer_id):
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.layer_delete(layer_id)

    def smpl_paint_color_solid(self, mesh_id, anchor, color, stop_level, tolerance=0, fixed=False, layer_id=0, timeout=0.05, steps=1):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        face_index, point = (anchor.face_index, anchor.point) if (not fixed) else (anchor, None)
        return effect.paint_color_solid(mesh_a, mesh_b, face_index, point, color, stop_level, tolerance, fixed, layer_id, timeout, steps)
    
    def smpl_paint_brush_solid(self, mesh_id, anchor, size, color, fill_test=0.0, tolerance=0, layer_id=0, timeout=0.05, steps=1):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        return effect.paint_brush_solid(mesh_a, mesh_b, anchor.face_index, anchor.point, size, color, fill_test, tolerance, layer_id, timeout, steps)
    
    def smpl_paint_brush_gradient(self, mesh_id, anchor, size, color_center, color_edge, hardness, fill_test=0.0, tolerance=0, layer_id=0, timeout=0.05, steps=1):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        return effect.paint_brush_gradient(mesh_a, mesh_b, anchor.face_index, anchor.point, size, color_center, color_edge, hardness, fill_test, tolerance, layer_id, timeout, steps)
    
    def smpl_paint_decal_solid(self, mesh_id, anchor, decal, align_prior, angle, scale, double_cover_test=True, fill_test=0.0, tolerance_decal=0, tolerance_paint=0, layer_id=0, timeout=0.05, steps=1):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        return effect.paint_decal_solid(mesh_a, mesh_b, anchor.face_index, anchor.point, decal, align_prior, angle, scale, double_cover_test, fill_test, tolerance_decal, tolerance_paint, layer_id, timeout, steps)
    
    def smpl_paint_decal_align_prior(self, mesh_id, anchor, align_axis, align_axis_fallback, tolerance=0):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        return mesh_align_prior(mesh_a, anchor.face_index, align_axis, align_axis_fallback, tolerance)
    
    def smpl_paint_clear(self, mesh_id, enabled_only=False, color=0):
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.clear(enabled_only, color)
    
    def smpl_paint_flush(self, mesh_id, force_alpha=None, stencil_layer=None):
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.flush(force_alpha, stencil_layer)


#------------------------------------------------------------------------------
# Renderer
#------------------------------------------------------------------------------

class renderer:
    def __init__(self, settings_offscreen, settings_scene, settings_camera, settings_camera_transform, settings_lamp):
        self._scene_control = renderer_scene_control(settings_offscreen, settings_scene, settings_camera, settings_camera_transform, settings_lamp)

    def smpl_load_model(self, model_path, num_betas, device):
        self._smpl_control = smpl_model(model_path, num_betas, device)
    
    def smpl_load_uv(self, filename_uv, texture_shape):
        self._mesh_control = renderer_mesh_control(filename_uv, texture_shape)

    def smpl_get_mesh(self, smpl_params) -> smpl_model_result:
        return self._smpl_control.to_mesh(smpl_params)

    def camera_get_pose(self):
        return self._scene_control.camera_get_pose()

    def camera_get_projection_matrix(self):
        return self._scene_control.camera_get_projection_matrix()

    def camera_get_transform_local(self):
        return self._scene_control.camera_get_transform_local()

    def camera_get_transform_plane(self):
        return self._scene_control.camera_get_transform_plane()   
    
    def camera_get_parameters(self) -> renderer_camera_transform_parameters:
        return self._scene_control.camera_get_parameters()

    def camera_adjust_parameters(self, yaw=None, pitch=None, distance=None, center=None, relative=True):
        self._scene_control.camera_adjust_parameters(yaw, pitch, distance, center, relative)

    def camera_move_center(self, delta_xyz, plane=True):
        self._scene_control.camera_move_center(delta_xyz, plane)

    def camera_solve_fov_z(self, center, points, plane=False):
        return self._scene_control.camera_solve_fov_z(center, points, plane)
    
    def camera_project_points(self, points, convention=(1, -1, -1)):
        return self._scene_control.camera_project_points(points, convention)
    
    def scene_render(self):
        return self._scene_control.render()
    
    def mesh_add_smpl(self, group, name, mesh, joints, texture, pose) -> renderer_mesh_identifier:
        return self._mesh_control.mesh_add_smpl(group, name, mesh, joints, texture, pose)
    
    def mesh_add_user(self, group, name, mesh, pose) -> renderer_mesh_identifier:
        return self._mesh_control.mesh_add_user(group, name, mesh, pose)

    def mesh_set_pose(self, mesh_id, pose):
        self._mesh_control.mesh_set_pose(mesh_id, pose)
        self._scene_control.group_item_set_pose(mesh_id, pose)

    def mesh_get_pose(self, mesh_id):
        return self._mesh_control.mesh_get_pose(mesh_id)

    def mesh_present(self, mesh_id):
        mesh = self._mesh_control.mesh_get_full(mesh_id) if (mesh_id.kind == 'smpl') else self._mesh_control.mesh_get_base(mesh_id)
        pose = self._mesh_control.mesh_get_pose(mesh_id)
        item = mesh_to_renderer(mesh)
        self._scene_control.group_item_add(mesh_id.group, mesh_id.name, item, pose)

    def mesh_remove_item(self, mesh_id):
        self._mesh_control.mesh_remove_item(mesh_id)
        self._scene_control.group_item_remove(mesh_id)

    def mesh_remove_group(self, group):
        self._mesh_control.mesh_remove_group(group)
        self._scene_control.group_clear(group)

    def mesh_remove_all(self):
        self._mesh_control.mesh_remove_all()
        self._scene_control.clear()

    def mesh_operation_raycast(self, mesh_id, origin, direction) -> mesh_chart_point:
        return self._mesh_control.mesh_operation_raycast(mesh_id, origin, direction)

    def mesh_operation_closest(self, mesh_id, origin) -> mesh_chart_point:
        return self._mesh_control.mesh_operation_closest(mesh_id, origin)

    def smpl_chart_create_frame(self, mesh_id, region) -> mesh_chart_frame:
        return self._mesh_control.smpl_chart_create_frame(mesh_id, region)
    
    def smpl_chart_from_cylindrical(self, mesh_id, frame, displacement, yaw) -> mesh_chart_point:
        return self._mesh_control.smpl_chart_from_cylindrical(mesh_id, frame, displacement, yaw)
    
    def smpl_chart_from_spherical(self, mesh_id, frame, yaw, pitch) -> mesh_chart_point:
        return self._mesh_control.smpl_chart_from_spherical(mesh_id, frame, yaw, pitch)

    def smpl_chart_to_cylindrical(self, mesh_id, frame, point) -> mesh_chart_local:
        return self._mesh_control.smpl_chart_to_cylindrical(mesh_id, frame, point)
    
    def smpl_chart_to_spherical(self, mesh_id, frame, point) -> mesh_chart_local:
        return self._mesh_control.smpl_chart_to_spherical(mesh_id, frame, point)

    def smpl_chart_to_pose(self, mesh_id, frame):
        return self._mesh_control.smpl_chart_to_pose(mesh_id, frame)

    def smpl_paint_set_background(self, mesh_id, background):
        self._mesh_control.smpl_paint_set_background(mesh_id, background)
    
    def smpl_paint_layer_create(self, mesh_id, layer_id):
        self._mesh_control.smpl_paint_layer_create(mesh_id, layer_id)
    
    def smpl_paint_layer_enable(self, mesh_id, layer_id, enable):
        self._mesh_control.smpl_paint_layer_enable(mesh_id, layer_id, enable)
    
    def smpl_paint_layer_clear(self, mesh_id, layer_id, color=0):
        self._mesh_control.smpl_paint_layer_clear(mesh_id, layer_id, color)
    
    def smpl_paint_layer_erase(self, mesh_id, result, color=0):
        self._mesh_control.smpl_paint_layer_erase(mesh_id, result.layer_id, result.data, color)
    
    def smpl_paint_layer_delete(self, mesh_id, layer_id):
        self._mesh_control.smpl_paint_layer_delete(mesh_id, layer_id)
    
    def smpl_paint_color_solid(self, mesh_id, anchor, color, stop_level, tolerance=0, fixed=False, layer_id=0, timeout=0.05, steps=1) -> mesh_paint_result:
        return self._mesh_control.smpl_paint_color_solid(mesh_id, anchor, color, stop_level, tolerance, fixed, layer_id, timeout, steps)
    
    def smpl_paint_brush_solid(self, mesh_id, anchor, size, color, fill_test=0.0, tolerance=0, layer_id=0, timeout=0.05, steps=1) -> mesh_paint_result:
        return self._mesh_control.smpl_paint_brush_solid(mesh_id, anchor, size, color, fill_test, tolerance, layer_id, timeout, steps)
    
    def smpl_paint_brush_gradient(self, mesh_id, anchor, size, color_center, color_edge, hardness, fill_test=0.0, tolerance=0, layer_id=0, timeout=0.05, steps=1) -> mesh_paint_result:
        return self._mesh_control.smpl_paint_brush_gradient(mesh_id, anchor, size, color_center, color_edge, hardness, fill_test, tolerance, layer_id, timeout, steps)
    
    def smpl_paint_decal_solid(self, mesh_id, anchor, decal, align_prior, angle, scale, double_cover_test=True, fill_test=0.0, tolerance_decal=0, tolerance_paint=0, layer_id=0, timeout=0.05, steps=1) -> mesh_paint_result:
        return self._mesh_control.smpl_paint_decal_solid(mesh_id, anchor, decal, align_prior, angle, scale, double_cover_test, fill_test, tolerance_decal, tolerance_paint, layer_id, timeout, steps)
    
    def smpl_paint_decal_align_prior(self, mesh_id, anchor, align_axis, align_axis_fallback, tolerance=0) -> mesh_paint_result:
        return self._mesh_control.smpl_paint_decal_align_prior(mesh_id, anchor, align_axis, align_axis_fallback, tolerance)

    def smpl_paint_clear(self, mesh_id, enabled_only=False, color=0):
        self._mesh_control.smpl_paint_clear(mesh_id, enabled_only, color=0)

    def smpl_paint_flush(self, mesh_id, force_alpha=None, stencil_layer=None):
        self._mesh_control.smpl_paint_flush(mesh_id, force_alpha, stencil_layer)





def project_points(world_points, K):
    camera_points = world_points @ K
    camera_points = camera_points[:, 0:2] / camera_points[:, 2:3]
    return camera_points
