#==============================================================================
# SMPL Painting And Charting Tools
#==============================================================================

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

from PIL import Image, ImageFont, ImageDraw


#------------------------------------------------------------------------------
# Math
#------------------------------------------------------------------------------

# TODO: m == 0??
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

# TODO: error for singular matrix
def geometry_solve_basis(vas, vbs, vad, vbd):
    return np.linalg.inv(np.vstack((vas, vbs, np.cross(vas, vbs)))) @ np.vstack((vad, vbd, np.cross(vad, vbd)))


# TODO: cx, cy?
def geometry_solve_fov_z(width, height, fx, fy, cx, cy, x, y, z, center, points):
    dp = (points - center)
    dx = np.abs(dp @ x.T)
    dy = np.abs(dp @ y.T)
    dz = dp @ z.T
    wx = dz + ((2 * fx * dx) / width)
    wy = dz + ((2 * fy * dy) / height)
    wz = np.max(np.hstack((wy, wx)))
    return wz


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
    uv[:, 0] = uv[:, 0] * (image_shape[1] - 1)
    uv[:, 1] = (1 - uv[:, 1]) * (image_shape[0] - 1)
    return uv


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
        for _ in range(0, max_iterations):
            if (len(self._expand_faces) > 0):
                self._faces = self._mnb.fetch(self._expand_faces, self._ignore_faces)
                self._expand_faces.clear()
                self._ignore_faces.clear()
            for face_anchor in self._faces:
                code = self._callback(face_anchor, self._mnb.level())
                if (code == mesh_neighborhood_processor_command.EXPAND):
                    self._expand_faces.add(face_anchor)
                elif (code == mesh_neighborhood_processor_command.IGNORE):
                    self._ignore_faces.add(face_anchor)
            if (len(self._expand_faces) < 1):
                self._done = True
                break

    def invoke_timeslice(self, timeout, steps=1):
        start = time.perf_counter()
        while (not self.done()):
            self.invoke(steps)
            if ((time.perf_counter() - start) >= timeout):
                break

    def done(self):
        return self._done


class mesh_neighborhood_processor_list:
    def __init__(self, face_list, callback):
        self._face_list = face_list
        self._face_index = 0
        self._face_count = len(face_list)
        self._callback = callback        
        self._done = False

    def invoke(self, max_iterations):
        for _ in range(0, max_iterations):
            if (self._face_index >= self._face_count):
                self._done = True
                break
            self._callback(self._face_list[self._face_index], -1)
            self._face_index += 1
        
    def invoke_timeslice(self, timeout, steps=1):
        start = time.perf_counter()
        while (not self.done()):
            self.invoke(steps)
            if ((time.perf_counter() - start) >= timeout):
                break

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
        self._pixels_painted = 0
        texture_processor(self._mesh_uvx[vertex_indices, :], self._paint_uv, self._tolerance)
        return mesh_neighborhood_processor_command.EXPAND if (self._pixels_painted > 0) else mesh_neighborhood_processor_command.IGNORE
    
    def _paint_uv(self, pixels, weights):
        self._pixels_painted = self._target(pixels, self._level)


# TODO: THIS DISTANCE IS NOT GEODESIC
class mesh_neighborhood_operation_brush:
    def __init__(self, mesh_vertices, mesh_faces, mesh_uvx, origin, targets, tolerance=0):
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._mesh_uvx = mesh_uvx
        self._origin = origin
        self._targets = targets
        self._tolerance = tolerance

    def paint(self, face_index, level):
        vertex_indices = self._mesh_faces[face_index]
        self._simplex_3d = self._mesh_vertices[vertex_indices, :]
        self._level = level
        self._pixels_painted = 0
        texture_processor(self._mesh_uvx[vertex_indices, :], self._paint_uv, self._tolerance)
        return mesh_neighborhood_processor_command.EXPAND if (self._pixels_painted > 0) else mesh_neighborhood_processor_command.IGNORE
    
    def _paint_uv(self, pixels, weights):
        distances = np.linalg.norm((weights @ self._simplex_3d) - self._origin, axis=1)
        for target in self._targets:
            self._pixels_painted += target(pixels, distances, self._level)


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

    def paint(self, face_index, level):
        self._face_normal = self._mesh_face_normals[face_index:(face_index + 1), :]        
        self._vertex_indices_b = self._mesh_faces[face_index]
        self._vertex_indices_a = self._uv_transform[self._vertex_indices_b]
        self._level = level
        command = self._target(self._mesh_vertices, self._face_normal, self._origin, self._vertex_indices_b, self._vertex_indices_a, None, None, self._level)
        if (command != mesh_neighborhood_processor_command.EXPAND):
            return command
        self._pixels_painted = 0
        texture_processor(self._mesh_uvx[self._vertex_indices_b, :], self._paint_uv, self._tolerance)
        return mesh_neighborhood_processor_command.EXPAND if (self._pixels_painted > 0) else mesh_neighborhood_processor_command.IGNORE

    def _paint_uv(self, pixels, weights):
        self._pixels_painted = self._target(self._mesh_vertices, self._face_normal, self._origin, self._vertex_indices_b, self._vertex_indices_a, pixels, weights, self._level)


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
        return 1 if (level < self._stop_level) else 0


class paint_brush_solid:
    def __init__(self, size, color, render_buffer, fill_test=0.0):
        self._size = size
        self._color = color
        self._render_buffer = render_buffer
        self._fill_test = fill_test

    def paint(self, pixels, distances, level):
        mask = distances < self._size
        pixels_painted = np.count_nonzero(mask)
        if (pixels_painted > 0):
            selection = pixels[mask, :]
            self._render_buffer[selection[:, 1], selection[:, 0], :] = self._color
        return 1 if (pixels_painted > int(self._fill_test * pixels.shape[0])) else 0


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
        pixels_painted = np.count_nonzero(mask)
        if (pixels_painted > 0):
            selection = pixels[mask, :]
            self._render_buffer[selection[:, 1], selection[:, 0], :] = texture_alpha_blend(self._color_center, self._color_edge, texture_alpha_remap(distances[mask, np.newaxis] / self._size, self._src, self._dst))
        return 1 if (pixels_painted > int(self._fill_test * pixels.shape[0])) else 0


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

        self._image_uvx[indices_uvx, :] = vxd
        self._push_simplex(vxd[:, 0:2])

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

        vps = mesh_vertices[vips_b:(vips_b + 1), :]
        vqs = mesh_vertices[viqs_b:(viqs_b + 1), :]
        vxs = mesh_vertices[vixs_b:(vixs_b + 1), :]

        vpd = self._image_uvx[vips_a:(vips_a + 1), :]
        vqd = self._image_uvx[viqs_a:(viqs_a + 1), :]

        align_outward = geometry_solve_basis(vqs - vps, face_normal, vqd - vpd, self._uvx_normal)

        vxd = ((vxs - vps) @ align_outward) + vpd
        vxd[:, 2] = 0

        if (self._double_cover_test):
            for i in range(0, len(self._simplices)):
                double_cover = self._test_simplex(vxd[:, 0:2], len(self._simplices) - 1 - i)
                if (double_cover):
                    return mesh_neighborhood_processor_command.IGNORE

        self._image_uvx[vixs_a:(vixs_a + 1), :] = vxd
        self._push_simplex(np.vstack((vxd[:, 0:2], vqd[:, 0:2], vpd[:, 0:2])))

        return mesh_neighborhood_processor_command.EXPAND

    def _blit(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        pixels_src = texture_uvx_invert(weights_src @ self._image_uvx[indices_uvx, 0:2], self._image_buffer.shape, 1)
        mask = texture_test_inside(self._image_buffer, pixels_src[:, 0], pixels_src[:, 1])
        pixels_painted = np.count_nonzero(mask)
        if (pixels_painted > 0):
            dst = pixels_dst[mask, :]
            src = pixels_src[mask, :]
            self._render_buffer[dst[:, 1], dst[:, 0], :] = texture_read(self._image_buffer, src[:, 0], src[:, 1])
        return 1 if (pixels_painted > int(self._fill_test * pixels_dst.shape[0])) else 0

    def paint(self, mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level):
        call = self._blit if ((pixels_dst is not None) and (weights_src is not None)) else self._unwrap if (level > 0) else self._bootstrap
        return call(mesh_vertices, face_normal, origin, indices_vertices, indices_uvx, pixels_dst, weights_src, level)


def painter_create_color(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, color, tolerance=0, fixed=False):
    mno = mesh_neighborhood_operation_color(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_uvx, color, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint) if (not fixed) else mesh_neighborhood_processor_list(face_index, mno.paint)
    return mnp


def painter_create_brush(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, brush, tolerance=0):
    mno = mesh_neighborhood_operation_brush(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_uvx, origin, brush, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


def painter_create_decal(mesh_a, mesh_b, mesh_uvx, uv_transform, face_index, origin, decal, tolerance=0):
    mno = mesh_neighborhood_operation_decal(mesh_b.vertices.view(np.ndarray), mesh_b.faces.view(np.ndarray), mesh_b.face_normals, mesh_uvx, uv_transform, origin, decal, tolerance)
    mnp = mesh_neighborhood_processor(mesh_a, {face_index}, mno.paint)
    return mnp


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

    def decompose(self, point):
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
        offset, nx, ny, nz, xz, nxz = self.decompose(point)
        displacement = ny
        yaw = np.arctan2(nx, nz)
        return mesh_chart_local(displacement, yaw, offset, nx, ny, nz, xz, nxz)

    def to_spherical(self, point):
        offset, nx, ny, nz, xz, nxz = self.decompose(point)
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
    
    def decompose(self, frame, point):
        return frame.decompose(point)
    
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

def smpl_camera_align(K_smpl, K_dst, points_world):
    n = points_world.shape[0]
    N = 2 * n
    A = np.zeros((N, 3), dtype=points_world.dtype)
    b = np.zeros((N, 1), dtype=points_world.dtype)
    fxy = K_dst.reshape((-1))[[0, 4]]

    points_camera = points_world @ K_smpl
    points_image = points_camera[:, 0:2] / points_camera[:, 2:3]
    a = K_dst[2:3, 0:2] - points_image
    w = -(a * points_world[:, 2:3] + fxy * points_world[:, 0:2])

    A[0:n, 0] = fxy[0]
    A[n:N, 1] = fxy[1]
    A[0:n, 2] = a[:, 0]
    A[n:N, 2] = a[:, 1]
    b[0:n, 0] = w[:, 0]
    b[n:N, 0] = w[:, 1]

    return np.linalg.lstsq(A, b)[0].T


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
        bigtoe   = self._joints[19:20, :]
        smalltoe = self._joints[20:21, :]
        ankle    = self._joints[14:15, :]
        heel     = self._joints[21:22, :]

        return self._template_frame_foot(bigtoe, smalltoe, ankle, heel)
    
    def _create_frame_foot_right(self):
        bigtoe   = self._joints[22:23, :]
        smalltoe = self._joints[23:24, :]
        ankle    = self._joints[11:12, :]
        heel     = self._joints[24:25, :]

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
        bigtoe = self._joints[19:20, :]
        ankle  = self._joints[14:15, :]
        knee   = self._joints[13:14, :]

        return self._template_frame_lower_leg(bigtoe, ankle, knee)
    
    def _create_frame_lower_leg_right(self):
        bigtoe = self._joints[22:23, :]
        ankle  = self._joints[11:12, :]
        knee   = self._joints[10:11, :]

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
        ankle = self._joints[14:15, :]
        knee  = self._joints[13:14, :]
        hip   = self._joints[12:13, :]

        return self._template_frame_thigh(ankle, knee, hip)

    def _create_frame_thigh_right(self):
        ankle = self._joints[11:12, :]
        knee  = self._joints[10:11, :]
        hip   = self._joints[9:10, :]

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
        lhip = self._joints[12:13, :]
        mhip = self._joints[8:9, :]
        rhip = self._joints[9:10, :]
        neck = self._joints[1:2, :]

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
        lear = self._joints[18:19, :]
        rear = self._joints[17:18, :]
        neck = self._joints[1:2, :]
        nose = self._joints[0:1, :]

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
        wrist    = self._joints[7:8, :]
        elbow    = self._joints[6:7, :]
        shoulder = self._joints[5:6, :]

        return self._template_frame_upper_arm(wrist, elbow, shoulder)

    def _create_frame_upper_arm_right(self):
        wrist    = self._joints[4:5, :]
        elbow    = self._joints[3:4, :]
        shoulder = self._joints[2:3, :]

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
        wrist    = self._joints[7:8, :]
        elbow    = self._joints[6:7, :]
        shoulder = self._joints[5:6, :]

        return self._template_frame_lower_arm(wrist, elbow, shoulder)

    def _create_frame_lower_arm_right(self):
        wrist    = self._joints[4:5, :]
        elbow    = self._joints[3:4, :]
        shoulder = self._joints[2:3, :]

        return self._template_frame_lower_arm(wrist, elbow, shoulder)


class smpl_model_result:
    def __init__(self, vertices, vertices_world, faces, joints, joints_world):
        self.vertices = vertices
        self.vertices_world = vertices_world
        self.faces = faces
        self.joints = joints
        self.joints_world = joints_world


class smpl_model:
    _SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

    def __init__(self, model_path, num_betas, device):
        self._smpl_model = smplx.SMPLLayer(model_path=model_path, num_betas=num_betas).to(device)
        self._smpl_to_open_pose = torch.tensor(smpl_model._SMPL_TO_OPENPOSE, dtype=torch.long, device=device)

    def to_mesh(self, smpl_params, camera_translation, openpose_joints=True):
        smpl_output = self._smpl_model(**smpl_params)
        vertices = smpl_output.vertices
        joints = smpl_output.joints.index_select(1, self._smpl_to_open_pose) if (openpose_joints) else smpl_output.joints
        t = camera_translation.unsqueeze(1)
        vertices_world = (vertices + t).detach().cpu().numpy()
        joints_world = (joints + t).detach().cpu().numpy()
        vertices = vertices.cpu().numpy()
        joints = joints.cpu().numpy()
        return smpl_model_result(vertices, vertices_world, self._smpl_model.faces, joints, joints_world)


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

    def group_item_remove(self, group, name):
        nodes = self._groups.get(group, None)
        if (nodes is not None):
            item = nodes.get(name, None)
            if (item is not None):
                self._scene.remove_node(item)

    def group_item_set_pose(self, group, name, pose):
        nodes = self._groups.get(group, None)
        if (nodes is not None):
            item = nodes.get(name, None)
            if (item is not None):
                self._scene.set_pose(item, pose)

    def group_item_get_pose(self, group, name):
        nodes = self._groups.get(group, None)
        if (nodes is not None):
            item = nodes.get(name, None)
            if (item is not None):
                return self._scene.get_pose(item)
        return None

    def group_clear(self, group):
        nodes = self._groups.pop(group, None)
        if (nodes is not None):
            for name, item in nodes.items():
                self._scene.remove_node(item)


class renderer_mesh_paint:
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

    def layer_enable(self, layer_id, enable):
        self._layer_enable[layer_id] = enable

    def layer_clear(self, layer_id):
        self._layers[layer_id][:, :, :] = 0

    def layer_delete(self, layer_id):
        self._layers.pop(layer_id)

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

    def task_create_paint_brush(self, task_id, mesh_a, mesh_b, face_index, origin, brush_ids, tolerance=0):
        o = [self._brushes[brush_id].paint for brush_id in brush_ids]
        self._tasks[task_id] = painter_create_brush(mesh_a, mesh_b, self._uvx, self._uv_transform, face_index, origin, o, tolerance)
        
    def task_create_paint_decal(self, task_id, mesh_a, mesh_b, face_index, origin, decal_idx, tolerance=0):
        o = self._decals[decal_idx].paint
        self._tasks[task_id] = painter_create_decal(mesh_a, mesh_b, self._uvx, self._uv_transform, face_index, origin, o, tolerance)

    def task_execute(self, task_id, timeout, steps=1):
        self._tasks[task_id].invoke_timeslice(timeout, steps)

    def task_done(self, task_id):
        return self._tasks[task_id].done()
    
    def task_get(self, task_id):
        return self._tasks[task_id]
    
    def task_delete(self, task_id):
        self._tasks.pop(task_id)

    def flush(self, force_alpha=None):
        self._render_target[:, :, :] = self._background
        for key in sorted(self._layers.keys()):
            if (self._layer_enable[key]):
                self._render_target[:, :, :] = np.array(Image.alpha_composite(Image.fromarray(self._render_target), Image.fromarray(self._layers[key])))
        if (force_alpha is not None):
            self._render_target[:, :, 3] = force_alpha


class renderer_mesh_identifier:
    def __init__(self, group, name, kind):
        self.group = group
        self.name = name
        self.kind = kind


#------------------------------------------------------------------------------
# Renderer
#------------------------------------------------------------------------------

class renderer:
    def __init__(self, settings_offscreen, settings_scene, settings_camera, settings_camera_transform, settings_lamp):
        self._scene_control = renderer_scene_control(settings_offscreen, settings_scene, settings_camera, settings_camera_transform, settings_lamp)
        self._meshes = dict()
        self._cswvfx = dict()

    def smpl_load_model(self, model_path, num_betas, device):
        self._smpl_control = smpl_model(model_path, num_betas, device)
    
    def smpl_load_uv(self, filename_uv, texture_shape):
        _, _, self._mesh_a_faces, self._mesh_b_faces, _, self._mesh_b_uv, self._uv_transform = texture_load_uv(filename_uv)
        self._mesh_b_uvx = texture_uv_to_uvx(self._mesh_b_uv.copy(), texture_shape)
        self._texture_shape = texture_shape

    def smpl_get_mesh(self, smpl_params, camera_translation) -> smpl_model_result:
        return self._smpl_control.to_mesh(smpl_params, camera_translation)

    def camera_get_pose(self):
        return self._scene_control.camera_get_pose()

    def camera_get_projection_matrix(self):
        self._scene_control.camera_get_projection_matrix()

    def camera_get_transform_local(self):
        return self._scene_control.camera_get_transform_local()

    def camera_get_transform_plane(self):
        return self._scene_control.camera_get_transform_plane()   
    
    def camera_get_parameters(self):
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
            target = texture.copy()
            visual = texture_create_visual(self._mesh_b_uv, target)
            effect = renderer_mesh_paint(self._mesh_b_uvx, target, self._uv_transform, texture)
            effect.layer_create(0)
            effect.layer_enable(0, True)
            g[name] = [visual, effect]
        else:
            visual, effect = u
            effect.set_background(texture)
        return visual
    
    def _mesh_present(self, group, name, mesh_x, pose):
        mesh_p = mesh_to_renderer(mesh_x)
        self._scene_control.group_item_add(group, name, mesh_p, pose)

    def mesh_add_smpl(self, group, name, mesh, joints, texture, pose):
        visual = self._tvfx_add(group, name, texture)
        mesh_a_tri = mesh
        mesh_b_tri = mesh_expand(mesh_a_tri, self._uv_transform, self._mesh_b_faces, visual)
        mesh_a_map = smpl_mesh_chart_openpose(mesh_a_tri, joints)
        self._mesh_add(group, name, mesh_a_tri, mesh_b_tri, mesh_a_map, pose)
        return renderer_mesh_identifier(group, name, 'smpl')

    def mesh_add_user(self, group, name, mesh, pose):
        self._mesh_add(group, name, mesh, None, None, pose)
        return renderer_mesh_identifier(group, name, 'user')

    def mesh_set_pose(self, mesh_id, pose):
        self._meshes[mesh_id.group][mesh_id.name][3] = pose
        self._scene_control.group_item_set_pose(mesh_id.group, mesh_id.name, pose)

    def mesh_get_pose(self, mesh_id):
        return self._meshes[mesh_id.group][mesh_id.name][3]

    def mesh_present_smpl(self, mesh_id):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        self._mesh_present(mesh_id.group, mesh_id.name, mesh_b, pose)

    def mesh_present_user(self, mesh_id):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        self._mesh_present(mesh_id.group, mesh_id.name, mesh_a, pose)

    def mesh_remove_item(self, mesh_id):
        self._meshes[mesh_id.group].pop(mesh_id.name)
        self._scene_control.group_item_remove(mesh_id.group, mesh_id.name)

    def mesh_remove_group(self, group):
        self._meshes.pop(group)
        self._scene_control.group_clear(group)

    def mesh_operation_raycast(self, mesh_id, origin, direction) -> mesh_chart_point:
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        local_origin = math_transform_points(origin, pose.T, True)
        local_direction = math_transform_bearings(direction, pose.T, True)
        point, face_index = mesh_raycast(mesh_a, local_origin, local_direction)
        return mesh_chart_point(point, face_index, local_origin, local_direction, None)

    def mesh_operation_closest(self, mesh_id, origin) -> mesh_chart_point:
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        local_origin = math_transform_points(origin, pose.T, True)
        point, face_index, _, = mesh_closest(mesh_a, local_origin)
        return mesh_chart_point(point, face_index, local_origin, None, None)

    def smpl_chart_create_frame(self, mesh_id, region) -> mesh_chart_frame:
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        return chart.create_frame(region)
    
    def smpl_chart_from_cylindrical(self, mesh_id, frame, displacement, yaw) -> mesh_chart_point: 
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        return chart.from_cylindrical(frame, displacement, yaw)
    
    def smpl_chart_from_spherical(self, mesh_id, frame, yaw, pitch) -> mesh_chart_point:
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        return chart.from_spherical(frame, yaw, pitch)
    
    def smpl_chart_to_cylindrical(self, mesh_id, frame, point) -> mesh_chart_local:
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        local_point = math_transform_points(point, pose.T, True)
        return chart.to_cylindrical(frame, local_point)
    
    def smpl_chart_to_spherical(self, mesh_id, frame, point) -> mesh_chart_local:
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        local_point = math_transform_points(point, pose.T, True)
        return chart.to_spherical(frame, local_point)
    
    def smpl_chart_to_pose(self, mesh_id, frame):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        return chart.to_pose(frame)
    
    def smpl_paint_color_solid(self, mesh_id, anchor, color, stop_level, timeout=0.05, steps=1, tolerance=0, fixed=False, manual=False, color_id=0, task_id=3):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        face_index, point = (anchor.face_index, anchor.point) if (not fixed) else (anchor, None)
        effect.color_create_solid(color_id, color, stop_level, 0)
        effect.task_create_paint_color(task_id, mesh_a, mesh_b, face_index, point, color_id, tolerance, fixed)
        if (manual):
            return effect.task_get(task_id)
        effect.task_execute(task_id, timeout, steps)
        return effect.task_done(task_id)

    def smpl_paint_brush_solid(self, mesh_id, anchor, size, color, fill_test=0.0, timeout=0.05, steps=1, tolerance=0) -> bool:
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.brush_create_solid(0, size, color, 0, fill_test)
        effect.task_create_paint_brush(0, mesh_a, mesh_b, anchor.face_index, anchor.point, [0], tolerance)
        effect.task_execute(0, timeout, steps)
        return effect.task_done(0)
    
    def smpl_paint_brush_gradient(self, mesh_id, anchor, size, color_center, color_edge, hardness, fill_test=0.0, timeout=0.05, steps=1, tolerance=0) -> bool:
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.brush_create_gradient(1, size, color_center, color_edge, hardness, 0, fill_test)
        effect.task_create_paint_brush(1, mesh_a, mesh_b, anchor.face_index, anchor.point, [1], tolerance)
        effect.task_execute(1, timeout, steps)
        return effect.task_done(1)
    
    def smpl_paint_decal_solid(self, mesh_id, anchor, decal, align_prior, angle, scale, double_cover_test=True, fill_test=0.0, timeout=0.05, steps=1, tolerance_decal=0, tolerance_paint=0) -> bool:
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.texture_attach(0, decal)
        effect.decal_create_solid(0, align_prior, angle, scale, 0, 0, double_cover_test, fill_test, tolerance_decal)
        effect.task_create_paint_decal(2, mesh_a, mesh_b, anchor.face_index, anchor.point, 0, tolerance_paint)
        effect.task_execute(2, timeout, steps)
        return effect.task_done(2)
    
    def smpl_paint_decal_align_prior(self, mesh_id, anchor, align_axis, align_axis_fallback, tolerance=0):
        mesh_a, mesh_b, chart, pose = self._meshes[mesh_id.group][mesh_id.name]
        align_normal = mesh_a.face_normals[anchor.face_index:(anchor.face_index+1), :]
        align_prior, nap = math_normalize(align_axis - (align_normal @ align_axis.T) * align_normal)
        return align_prior if (nap > tolerance) else math_normalize(align_axis_fallback - (align_normal @ align_axis_fallback.T) * align_normal)[0]
    
    def smpl_paint_clear(self, mesh_id) -> None:
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.layer_clear(0)

    def smpl_paint_flush(self, mesh_id) -> None:
        visual, effect = self._cswvfx[mesh_id.group][mesh_id.name]
        effect.flush()

