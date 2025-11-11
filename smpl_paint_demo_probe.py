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
import torch
import trimesh
import smplpact


class demo:
    def run(self):
        # Settings
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._smpl_dataset_path = 'C:/Users/jcds/Desktop/Oct_22_stevens/output_pairs/'
        self._smpl_model_path = './data/smpl/SMPL_NEUTRAL.pkl'
        self._smpl_uv_path = './data/smpl_uv.obj'
        self._smpl_texture_path = './data/textures/f_01_alb.002_1k.png'
        self._smpl_texture_load_alpha = False
        
        self._viewport_width = 1280
        self._viewport_height = 720

        self._camera_fov_vertical = np.pi / 3
        self._camera_yaw_increment = 10
        self._camera_pitch_increment = 10
        self._camera_distance_increment = 0.1
        self._camera_use_plane = True

        self._cursor_radius = 0.015
        self._cursor_height = 0.04

        self._brush_size = 0.02
        self._brush_color_center = np.array([255, 0, 0, 255], dtype=np.uint8)
        self._brush_color_edge = np.array([255, 255, 0, 255], dtype=np.uint8)
        self._brush_hardness = 0.33

        self._fps_period = 2.0
        # End Settings

        print(f'Using device: {self._device}')

        # Create offscreen renderer
        fxy = smplpact.geometry_fov_to_f(self._camera_fov_vertical, self._viewport_height)

        cfg_offscreen = smplpact.renderer_create_settings_offscreen(self._viewport_width, self._viewport_height)
        cfg_scene = smplpact.renderer_create_settings_scene()
        cfg_camera = smplpact.renderer_create_settings_camera(fxy, fxy, self._viewport_width // 2, self._viewport_height // 2)
        cfg_camera_transform = smplpact.renderer_create_settings_camera_transform()
        cfg_lamp = smplpact.renderer_create_settings_lamp()
        
        self._offscreen_renderer = smplpact.renderer(cfg_offscreen, cfg_scene, cfg_camera, cfg_camera_transform, cfg_lamp)
        # Load SMPL texture
        self._texture_array = smplpact.texture_load_image(self._smpl_texture_path, load_alpha=self._smpl_texture_load_alpha)

        # Load SMPL model
        self._offscreen_renderer.smpl_load_model(self._smpl_model_path, 10, self._device)
        self._offscreen_renderer.smpl_load_uv(self._smpl_uv_path, self._texture_array.shape)

        # Load realsense intrinsics
        self._K = np.array([[1361.8736572265625, 0, 963.2017211914062],[0, 1361.03076171875, 552.251220703125],[0, 0, 1]], dtype=np.float32)
        
        # Create UI elements
        self._cursor_mesh = trimesh.creation.cone(radius=self._cursor_radius, height=self._cursor_height)
        self._cursor_pose = np.eye(4, dtype=np.float32)

        files, folders = smplpact.scan_path(self._smpl_dataset_path, folders_sort=True, folders_key=lambda x : int(x.split('_')[-1]))
        index = 0

        # Configure filters
        self._offscreen_renderer.smpl_filter_reset()
        self._offscreen_renderer.smpl_filter_set_bounding_box([670, 0, 1400, 1080], [smplpact.smpl_joints_openpose.MidHip, smplpact.smpl_joints_openpose.LHip, smplpact.smpl_joints_openpose.RHip], np.array([0.4, 0.3, 0.3], dtype=np.float32), 0.59)
        self._offscreen_renderer.smpl_filter_set_forward_face(None)
        self._offscreen_renderer.smpl_filter_set_exponential_single([0.25, 0.25, 0.25, 0.9])

        # Run inference and painting
        start = time.perf_counter()
        count = 0
        while (True):
            self._smpl_test_message_path = os.path.join(folders[index], 'pose.json')
            self._probe_test_message_path = os.path.join(folders[index], 'aruco.json')
            self._test_image_path = os.path.join(folders[index], 'realsense_image.jpg')
            print(folders[index])
            index = (index + 1) % len(folders)

            # Load test CameraHMR message
            with open(self._smpl_test_message_path, 'rt') as json_file:
                self._test_camerahmr_message = json.load(json_file)

            # Load test probe message
            with open(self._probe_test_message_path, 'rt') as probe_file:
                self._probe_message = json.load(probe_file)

            # Load test image
            self._test_image = cv2.imread(self._test_image_path, cv2.IMREAD_COLOR_BGR)
            #self._test_image = np.asarray(Image.open(self._test_image_path))[:,:,::-1]
            if (self._test_image is None):
                continue

            status = self._loop()

            count += 1
            end = time.perf_counter()
            delta = end - start
            if (delta > self._fps_period):
                print(f'FPS: ', count / delta)
                start = end
                count = 0

            if (not status):
                break

    def _loop(self):
        # SMPL params to mesh
        smpl_params, self._K_smpl = self.smpl_unpack_cliff(self._test_camerahmr_message)

        smpl_ok, smpl_result = self._offscreen_renderer.smpl_get_mesh(smpl_params, self._K_smpl.T, self._K.T)
        
        smpl_vertices = smpl_result.vertices[0]
        smpl_joints = smpl_result.joints[0]
        smpl_faces = smpl_result.faces
        # ^ FILTER for 0 meshes ^

        smpl_mesh = smplpact.mesh_create(smpl_vertices, smpl_faces, visual=None)

        # Compute pose to set mesh upright
        # Poses convert from object to world
        smpl_mesh_pose = np.linalg.inv(smplpact.smpl_mesh_chart_openpose(smpl_mesh, smpl_joints).create_frame('body_center').to_pose()).T

        # Add SMPL mesh to the main scene
        smpl_mesh_id = self._offscreen_renderer.mesh_add_smpl('smpl', 'patient', smpl_mesh, smpl_joints, self._texture_array, smpl_mesh_pose)

        # Set probe position
        probe_position = np.array([[self._probe_message['x'], self._probe_message['y'], self._probe_message['z']]], dtype=np.float32)
        #probe_position *= 0.94 # heuristic
        self._cursor_pose[3:4, :3] = smplpact.math_transform_points(probe_position, smpl_mesh_pose.T, inverse=False)
        
        # Add cursor to the main scene
        cursor_pose = self._cursor_pose.T
        cursor_mesh_id = self._offscreen_renderer.mesh_add_user('ui', 'cursor', self._cursor_mesh, cursor_pose)

        # Find closest point to mesh
        cursor_anchor = self._offscreen_renderer.mesh_operation_closest(smpl_mesh_id, self._cursor_pose[3:4, :3])

        # Paint SMPL mesh
        # Paint circular gradient at camera mesh intersection/closest
        if (cursor_anchor.point is not None):
            # Gradient option
            #self._offscreen_renderer.smpl_paint_brush_gradient(smpl_mesh_id, cursor_anchor, self._brush_size, self._brush_color_center, self._brush_color_edge, self._brush_hardness, fill_test=0.25)
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

        joints_image = (smpl_vertices/smpl_vertices[:,2:3]) @ self._K.T
        for i in range(0, joints_image.shape[0]):
            center = (int(joints_image[i, 0]), int(joints_image[i, 1]))
            cv2.circle(self._test_image, center, 5, (0, 255, 0), -1)

        cv2.imshow('RealSense', self._test_image)

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

    def smpl_unpack_camerahmr(self, msg):
        person_list = msg['persons']
        global_orient = torch.tensor([person['smpl_params']['global_orient'] for person in person_list], dtype=torch.float32, device=self._device)
        body_pose = torch.tensor([person['smpl_params']['body_pose'] for person in person_list], dtype=torch.float32, device=self._device)
        betas = torch.tensor([person['smpl_params']['betas'] for person in person_list], dtype=torch.float32, device=self._device)
        camera_translation = torch.tensor([person['camera_translation'] for person in person_list], dtype=torch.float32, device=self._device)
        smpl_params = { 'global_orient' : global_orient, 'body_pose' : body_pose, 'betas' : betas, 'transl' : camera_translation }
        f = person_list[0]['focal_length']
        w, h = msg['image_size']
        K_smpl = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32) 
        return smpl_params, K_smpl
    
    def smpl_unpack_cliff(self, msg):
        person_list = msg['persons']
        smpl_pose = torch.tensor([person['smpl_pose'] for person in person_list], dtype=torch.float32, device=self._device)
        global_orient = smpl_pose[:, 0:1, :, :]
        body_pose = smpl_pose[:, 1:, :, :]
        betas = torch.tensor([person['smpl_shape'] for person in person_list], dtype=torch.float32, device=self._device)
        camera_translation = torch.tensor([person['camera_params'] for person in person_list], dtype=torch.float32, device=self._device)
        smpl_params = { 'global_orient' : global_orient, 'body_pose' : body_pose, 'betas' : betas, 'transl' : camera_translation }
        f = person_list[0]['focal_length']
        w, h = msg['image_size']
        K_smpl = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32) 
        return smpl_params, K_smpl


def main():
    demo().run()


if __name__ == '__main__':
    main()

