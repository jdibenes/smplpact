#------------------------------------------------------------------------------
# SMPL Probe Paint demo
#
# Keyboard Controls:
# d/a: Increment/Decrement Camera yaw
# w/s: Increment/Decrement Camera pitch
# r/f: Zoom In/Out
# n/m: Camera move left/right
# u/j: Camera move up/down
# i/k: Camera move forward/backward
# Esc: Quit
#------------------------------------------------------------------------------

import os
import time
import json
import cv2
import numpy as np
import trimesh
import smplpact


class demo:
    def run(self):
        # Settings ------------------------------------------------------------
        self._device = 'cuda'

        self._smpl_dataset_path = 'C:/Users/jcds/Desktop/data_dump'
        self._smpl_model_path = './data/smpl/SMPL_NEUTRAL.pkl'
        self._smpl_uv_path = './data/smpl_uv.obj'
        self._smpl_texture_path = './data/textures/f_01_alb.002_1k.png'
        self._smpl_texture_load_alpha = False
        self._smpl_face_segmentation_path = './data/smpl_face_segmentation.json'
        
        self._viewport_width = 1280
        self._viewport_height = 720

        self._camera_fov_vertical = np.pi / 3
        self._camera_yaw_increment = 10
        self._camera_pitch_increment = 10
        self._camera_distance_increment = 0.1
        self._camera_use_plane = True

        self._cursor_radius = 0.015

        self._brush_size = 0.02
        self._brush_color_center = np.array([255, 0, 0, 255], dtype=np.uint8)

        self._probe_image_radius = 7
        self._probe_image_color = (255, 0, 255)
        self._probe_image_thickness = 3

        self._fps_period = 2.0
        # End Settings --------------------------------------------------------

        # Load RealSense parameters
        self._realsense_K = np.array([[605.2772, 0, 321.4230],[0, 604.9025, 245.44498],[0, 0, 1]], dtype=np.float32)
        self._realsense_R_dc = np.array([[0.99997896, -0.00459499, -0.00457422], [0.00462210, 0.99997175, 0.00593209], [0.00454684, -0.00595311, 0.99997193]], dtype=np.float32)
        self._realsense_t_dc = np.array([[ 0.014813979156315327], [-0.00004484738383325748], [0.0002225551288574934]], dtype=np.float32)

        # Load SMPL texture
        self._texture_array = smplpact.texture_load_image(self._smpl_texture_path, load_alpha=self._smpl_texture_load_alpha)

        # Create offscreen renderer
        fxy = smplpact.geometry_fov_to_f(self._camera_fov_vertical, self._viewport_height)

        cfg_offscreen = smplpact.renderer_create_settings_offscreen(self._viewport_width, self._viewport_height)
        cfg_scene = smplpact.renderer_create_settings_scene()
        cfg_camera = smplpact.renderer_create_settings_camera(fxy, fxy, self._viewport_width // 2, self._viewport_height // 2)
        cfg_camera_transform = smplpact.renderer_create_settings_camera_transform()
        cfg_lamp = smplpact.renderer_create_settings_lamp()
        cfg_smpl_model = smplpact.renderer_create_settings_smpl_model(self._smpl_model_path, 10, self._device)
        cfg_smpl_uv = smplpact.renderer_create_settings_smpl_uv(self._smpl_uv_path, self._texture_array.shape)

        self._offscreen_renderer = smplpact.renderer_context(cfg_offscreen, cfg_scene, cfg_camera, cfg_camera_transform, cfg_lamp, cfg_smpl_model, cfg_smpl_uv)
        
        # Create UI elements
        self._cursor_mesh = trimesh.creation.icosphere(radius=self._cursor_radius)
        self._cursor_pose = np.eye(4, dtype=np.float32)

        # Load segmentation
        with open(self._smpl_face_segmentation_path, 'rt') as segmentation_file:
            self._smpl_segmentation = json.load(segmentation_file)

        # Load dataset
        files, folders = smplpact.scan_path(self._smpl_dataset_path, folders_sort=True, folders_key=lambda x : int(os.path.split(x)[1]))
        index = 0

        # Print configuration
        print(f'Using device: {self._device}')
        print(f'SMPL texture shape: {self._texture_array.shape}')

        # Run inference and painting
        start = time.perf_counter()
        count = 0
        while (True):
            # Get paths
            self._smpl_message_path = os.path.join(folders[index], 'patient_pose_raw.json')
            self._probe_message_path = os.path.join(folders[index], 'hand_tracking.json')
            self._realsense_image_path = os.path.join(folders[index], 'realsense_color_frame.png')
            
            index = (index + 1) % len(folders)

            # Load pose message
            with open(self._smpl_message_path, 'rt') as json_file:
                self._pose_message = json.load(json_file)

            # Load probe message
            with open(self._probe_message_path, 'rt') as probe_file:
                self._probe_message = json.load(probe_file)

            # Load image
            self._realsense_image = cv2.imread(self._realsense_image_path, cv2.IMREAD_COLOR_BGR)
            if (self._realsense_image is None):
                continue

            with self._offscreen_renderer:
                status = self._paint()

            count += 1
            end = time.perf_counter()
            delta = end - start
            if (delta > self._fps_period):
                print(f'FPS: ', count / delta)
                start = end
                count = 0

            if (not status):
                break

    def _paint(self):
        # SMPL params to mesh
        smpl_params, smpl_K = self._offscreen_renderer.smpl_unpack(self._pose_message)
        smpl_ok, smpl_result = self._offscreen_renderer.smpl_get_mesh(smpl_params, smpl_K.T, self._realsense_K.T)
        smpl_vertices = smpl_result.vertices[0]
        smpl_joints = smpl_result.joints[0]
        smpl_faces = smpl_result.faces
        smpl_mesh = smplpact.mesh_create(smpl_vertices, smpl_faces)

        # Compute pose to set mesh upright
        # Poses convert from object to world
        smpl_mesh_pose = np.linalg.inv(smplpact.smpl_mesh_chart_openpose(smpl_mesh, smpl_joints).create_frame('body_center').to_pose()).T

        # Add SMPL mesh to the main scene
        smpl_mesh_id = self._offscreen_renderer.mesh_add_smpl('smpl', 'patient', smpl_mesh, smpl_joints, self._texture_array, smpl_mesh_pose)

        # Set probe position
        probe_hand = self._probe_message['hand_position']
        probe_position = np.array([[probe_hand['x'], probe_hand['y'], probe_hand['z']]], dtype=np.float32)
        probe_position = (probe_position * 1.004) + smplpact.math_normalize(probe_position)[0] * 0.100  # heuristic approximation (rms depth error for ~1m + probe height ~10cm?)
        #probe_position = (probe_position @ self._realsense_R_dc.T) + self._realsense_t_dc.T # depth/color seems to be already aligned?
        
        self._cursor_pose[:3, 3:4] = smplpact.math_transform_points(probe_position, smpl_mesh_pose.T, inverse=False).T
        
        # Add cursor to the main scene        
        cursor_mesh_id = self._offscreen_renderer.mesh_add_user('ui', 'cursor', self._cursor_mesh, self._cursor_pose)

        # Find closest point to mesh and paint
        cursor_anchor = self._offscreen_renderer.mesh_operation_closest(smpl_mesh_id, self._cursor_pose[:3, 3:4].T)
        if (cursor_anchor.point is not None):
            # Solid color option
            self._offscreen_renderer.smpl_paint_brush_solid(smpl_mesh_id, cursor_anchor, self._brush_size, self._brush_color_center, fill_test=0.25)

        # Finalize SMPL painting
        # Compute painted texture
        self._offscreen_renderer.smpl_paint_flush(smpl_mesh_id)
        # Remove painting for next frame (comment out to keep paintings across frames)
        self._offscreen_renderer.smpl_paint_clear(smpl_mesh_id)

        # Finalize mesh processing
        self._offscreen_renderer.mesh_present(smpl_mesh_id)
        self._offscreen_renderer.mesh_present(cursor_mesh_id)

        # Render
        color, depth = self._offscreen_renderer.scene_render()

        # Show rendered image
        cv2.imshow('SMPL Paint Demo', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        # Draw probe
        probe_image = (probe_position / probe_position[:, 2:3]) @ self._realsense_K.T
        center = (int(probe_image[0, 0]), int(probe_image[0, 1]))
        cv2.circle(self._realsense_image, center, self._probe_image_radius, self._probe_image_color, self._probe_image_thickness)

        # Show image
        cv2.imshow('RealSense', self._realsense_image)

        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF
 
        if (key == 68 or key == 100): # d
            self._offscreen_renderer.camera_adjust_parameters(yaw=self._camera_yaw_increment, relative=True)
        if (key == 65 or key == 97): # a
            self._offscreen_renderer.camera_adjust_parameters(yaw=-self._camera_yaw_increment, relative=True)
        if (key == 87 or key == 119): # w
            self._offscreen_renderer.camera_adjust_parameters(pitch=-self._camera_pitch_increment, relative=True)
        if (key == 83 or key == 115): # s
            self._offscreen_renderer.camera_adjust_parameters(pitch=self._camera_pitch_increment, relative=True)
        if (key == 82 or key == 114): #r
            self._offscreen_renderer.camera_adjust_parameters(distance=-self._camera_distance_increment, relative=True)
        if (key == 70 or key == 102): #f
            self._offscreen_renderer.camera_adjust_parameters(distance=self._camera_distance_increment, relative=True)
        if (key == 78 or key == 110): #n
            self._offscreen_renderer.camera_move_center([-self._camera_distance_increment, 0, 0], plane=self._camera_use_plane)
        if (key == 77 or key == 109): #m
            self._offscreen_renderer.camera_move_center([self._camera_distance_increment, 0, 0], plane=self._camera_use_plane)
        if (key == 85 or key == 117): #u
            self._offscreen_renderer.camera_move_center([0, self._camera_distance_increment, 0], plane=self._camera_use_plane)
        if (key == 74 or key == 106): #j
            self._offscreen_renderer.camera_move_center([0, -self._camera_distance_increment, 0], plane=self._camera_use_plane)
        if (key == 73 or key == 105): #i
            self._offscreen_renderer.camera_move_center([0, 0, -self._camera_distance_increment], plane=self._camera_use_plane)
        if (key == 75 or key == 107): #k
            self._offscreen_renderer.camera_move_center([0, 0, self._camera_distance_increment], plane=self._camera_use_plane)

        if (key == 27): # esc
            return False
        
        return True


def main():
    demo().run()


if __name__ == '__main__':
    main()

