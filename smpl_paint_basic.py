#------------------------------------------------------------------------------
# SMPL Basic Paint demo
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

import time
import json
import cv2
import numpy as np
import smplpact


class demo:
    def run(self):
        # Settings ------------------------------------------------------------
        self._device = 'cuda'
        
        self._smpl_message_path = './data/patient_pose_raw.json'
        self._smpl_model_path = './data/smpl/SMPL_NEUTRAL.pkl'
        self._smpl_uv_path = './data/smpl_uv.obj'
        self._smpl_texture_path = './data/textures/f_01_alb.002_1k.png'
        self._smpl_texture_load_alpha = False

        self._realsense_K = np.array([[605.2772, 0, 321.4230], [0, 604.9025, 245.44498], [0, 0, 1]], dtype=np.float32)
        
        self._viewport_width = 1280
        self._viewport_height = 720

        self._camera_fov_vertical = np.pi / 3
        self._camera_yaw_increment = 10
        self._camera_pitch_increment = 10
        self._camera_distance_increment = 0.1
        self._camera_use_plane = True

        self._fps_period = 2.0
        # End Settings --------------------------------------------------------

        # Load SMPL texture
        self._texture_array = smplpact.texture_load_image(self._smpl_texture_path, load_alpha=self._smpl_texture_load_alpha)

        # Create offscreen renderer
        fxy = smplpact.geometry_fov_to_f(self._camera_fov_vertical, self._viewport_height)

        cfg_offscreen = smplpact.renderer_create_settings_offscreen(self._viewport_width, self._viewport_height)
        cfg_scene = smplpact.renderer_create_settings_scene()
        cfg_camera = smplpact.renderer_create_settings_camera(fxy, fxy, self._viewport_width // 2, self._viewport_height // 2)
        cfg_camera_transform = smplpact.renderer_create_settings_camera_transform()
        cfg_lamp = smplpact.renderer_create_settings_lamp()
        cfg_smpl_model = smplpact.renderer_create_settings_smpl_model(self._smpl_uv_path, self._texture_array.shape, self._smpl_model_path, 10, self._device)

        self._offscreen_renderer = smplpact.renderer_context(cfg_offscreen, cfg_scene, cfg_camera, cfg_camera_transform, cfg_lamp, cfg_smpl_model)

        # Load test pose message
        with open(self._smpl_message_path, 'rt') as json_file:
            self._pose_message = json.load(json_file)

        # Print configuration
        print(f'Using device: {self._device}')
        print(f'SMPL texture shape: {self._texture_array.shape}')

        # Run inference and painting
        start = time.perf_counter()
        count = 0
        while (True):
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
        smpl_data = smpl_result.at(0)

        # Compute pose to set mesh upright
        # Poses convert from object to world
        smpl_mesh = smplpact.mesh_create(smpl_data.vertices, smpl_data.faces)
        chart = smplpact.smpl_mesh_chart_openpose(smpl_mesh, smpl_data.joints)
        frame = chart.create_frame('body_center')
        pose = frame.to_pose()
        smpl_mesh_pose = smplpact.math_invert_pose(pose).T

        # Add SMPL mesh to the main scene
        smpl_mesh_id = self._offscreen_renderer.mesh_add_smpl('smpl', 'patient', smpl_data, self._texture_array, smpl_mesh_pose)

        # Add your paint code here
        # ...

        # Finalize SMPL painting
        # Compute painted texture
        self._offscreen_renderer.smpl_paint_flush(smpl_mesh_id)
        # Remove painting for next frame (comment out to keep paintings across frames)
        self._offscreen_renderer.smpl_paint_clear(smpl_mesh_id)

        # Finalize mesh processing
        self._offscreen_renderer.mesh_present(smpl_mesh_id)

        # Render
        color, depth = self._offscreen_renderer.scene_render()

        # Show rendered image
        cv2.imshow('SMPL Paint Demo', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

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

