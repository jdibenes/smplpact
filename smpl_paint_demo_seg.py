#------------------------------------------------------------------------------
# SMPL Leg Paint Based on Progress demo
#
# Keyboard Controls:
# q: Simulate Progress
# d/a: Increment/Decrement Camera yaw
# w/s: Increment/Decrement Camera pitch
# r/f: Zoom In/Out
# n/m: Camera move left/right
# u/j: Camera move up/down
# i/k: Camera move forward/backward
# Esc: Quit
#------------------------------------------------------------------------------

import time
import json
import cv2
import numpy as np
import torch
import smplpact


class demo:
    def run(self):
        # Settings
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._smpl_test_message_path = './data/test_camerahmr_message.txt'
        self._smpl_model_path = './data/smpl/SMPL_NEUTRAL.pkl'
        self._smpl_uv_path = './data/smpl_uv.obj'
        self._smpl_texture_path = './data/textures/f_01_alb.002_1k.png'
        self._smpl_face_seg_path = './data/smpl_face_segmentation.json'
        self._smpl_texture_load_alpha = False
        
        self._viewport_width = 1280
        self._viewport_height = 720

        self._camera_fov_vertical = np.pi / 3
        self._camera_yaw_increment = 10
        self._camera_pitch_increment = 10
        self._camera_distance_increment = 0.1
        self._camera_use_plane = True

        self._color_area = 'rightUpLeg'
        self._brush_color_center = np.array([255, 0, 0, 255], dtype=np.uint8)

        self._progress_increment = 0.125

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

        # Load test CameraHMR message
        with open(self._smpl_test_message_path, 'rt') as json_file:
            test_camerahmr_message = json.load(json_file)

        # Load SMPL segmentation
        with open(self._smpl_face_seg_path, 'rt') as seg_file:
            self._face_segmentation = json.load(seg_file)

        self._progress = 0.0

        # Run inference and painting
        start = time.perf_counter()
        count = 0
        while (self._loop(test_camerahmr_message)):
            count += 1
            end = time.perf_counter()
            delta = end - start
            if (delta > self._fps_period):
                print(f'FPS: ', count / delta)
                start = end
                count = 0

    def _loop(self, camerahmr_message):
        # SMPL params to mesh
        person_list = camerahmr_message['persons']
        smpl_params, camera_translation = self.smpl_unpack(person_list)
        smpl_result = self._offscreen_renderer.smpl_get_mesh(smpl_params, camera_translation)
        smpl_vertices = smpl_result.vertices[0]
        smpl_joints = smpl_result.joints[0]
        smpl_faces = smpl_result.faces
        smpl_mesh = smplpact.mesh_create(smpl_vertices, smpl_faces, visual=None)

        # Compute pose to set mesh upright
        # Poses convert from object to world
        smpl_mesh_pose = np.linalg.inv(smplpact.smpl_mesh_chart_openpose(smpl_mesh, smpl_joints).create_frame('body_center').to_pose()).T

        # Add SMPL mesh to the main scene
        smpl_mesh_id = self._offscreen_renderer.mesh_add_smpl('smpl', 'patient', smpl_mesh, smpl_joints, self._texture_array, smpl_mesh_pose)

        # Paint SMPL mesh
        # Paint leg based on progress
        leg_faces = self._face_segmentation[self._color_area]
        leg_range = int(self._progress * len(leg_faces))
        self._offscreen_renderer.smpl_paint_color_solid(smpl_mesh_id, leg_faces[0:leg_range], self._brush_color_center, 1, timeout=0.5, fixed=True)

        # Finalize SMPL painting
        # Compute painted texture
        self._offscreen_renderer.smpl_paint_flush(smpl_mesh_id)
        # Remove painting for next frame (comment out to keep paintings across frames)
        self._offscreen_renderer.smpl_paint_clear(smpl_mesh_id)

        # Finalize mesh processing
        self._offscreen_renderer.mesh_present_smpl(smpl_mesh_id)

        # Render
        color, depth = self._offscreen_renderer.scene_render()

        # Show rendered image
        cv2.imshow('SMPL Paint Demo', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF

        if (key == 81 or key == 113): # q: Simulate progress
            if (self._progress < 1.0):
                self._progress += self._progress_increment
                if (self._progress > 1.0):
                    self._progress = 1.0
            else:
                self._progress = 0
            print(f'Progress: {self._progress}')

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

    def smpl_unpack(self, person_list):
        global_orient = torch.tensor([person['smpl_params']['global_orient'] for person in person_list], dtype=torch.float32, device=self._device)
        body_pose = torch.tensor([person['smpl_params']['body_pose'] for person in person_list], dtype=torch.float32, device=self._device)
        betas = torch.tensor([person['smpl_params']['betas'] for person in person_list], dtype=torch.float32, device=self._device)
        smpl_params = { 'global_orient' : global_orient, 'body_pose' : body_pose, 'betas' : betas }
        camera_translation = torch.tensor([person['camera_translation'] for person in person_list], dtype=torch.float32, device=self._device)
        return smpl_params, camera_translation


def main():
    demo().run()


if __name__ == '__main__':
    main()

