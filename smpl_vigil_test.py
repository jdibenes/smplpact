
import os
import json
import numpy as np
import torch
import cv2

from smplpact import *

class demo:
    def __init__(self):
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        weights_path = '.'#self._config.get("Pose Estimation", "weights", fallback=None)

        smpl_gender = 'female'#self._config.get("Pose Estimation", "smpl_gender", fallback="neutral")
        smpl_gender = str(smpl_gender).strip().upper()
        smpl_gender = {'NEUTRAL':'NEUTRAL', 'MALE':'MALE', 'FEMALE':'FEMALE'}.get(smpl_gender, 'NEUTRAL')
        self._smpl_model_path = weights_path + f'/data/smpl/SMPL_{smpl_gender}.pkl'
        print(smpl_gender)

        #self._smpl_model_path = weights_path + '/data/smpl/SMPL_MALE.pkl'
        self._smpl_uv_path = weights_path + '/data/smpl_uv.obj'
        self._smpl_texture_path = weights_path + '/data/textures/f_01_alb.002_1k.png'
        self._smpl_texture_load_alpha = False

        self._R_dc = np.array([
            [ 0.99997896, -0.00459499, -0.00457422],
            [ 0.00462210,  0.99997175,  0.00593209],
            [ 0.00454684, -0.00595311,  0.99997193]
        ], dtype=np.float32)

        self._t_dc = np.array([
            [ 0.014813979156315327],
            [-0.00004484738383325748],
            [ 0.0002225551288574934]
        ], dtype=np.float32)

        self._viewport_width = 1280
        self._viewport_height = 720

        self._camera_fov_vertical = np.pi / 3
        self._camera_yaw_increment = 10
        self._camera_pitch_increment = 10
        self._camera_distance_increment = 0.1
        self._camera_use_plane = True
        self._camera_focus_factor = 1.25

        self._text_font_name = 'arial.ttf'#weights_path + '/CameraHMR/data/arial.ttf'
        self._text_font_size = 512
        self._text_font_color = (255, 0, 0, 255)
        self._text_canvas_color = (255, 255, 255, 255)
        self._text_stroke_width = 1
        self._text_content = ['Probe', 'Here']
        self._text_line_spacing = 20
        self._text_pad_horizontal_ratio = 0.05
        self._text_pad_vertical_ratio = 0.1

        self._cursor_radius = 0.015
        self._cursor_height = 0.04
        self._cursor_offset_increment = 0.02
        self._cursor_angle_increment = np.radians(10)

        self._brush_size = 0.01
        self._brush_color_center = np.array([255, 0, 0, 255], dtype=np.uint8)
        self._brush_color_edge = np.array([255, 255, 0, 255], dtype=np.uint8)
        self._brush_hardness = 0.33
        self._decal_size = 10000 * 2
        self._decal_angle = 0
        self._joint_projection_radius = 3
        self._joint_projection_color = [255, 0, 255]

        self._fps_period = 2.0
        # End Settings

        # Load Realsense camera intrinsics parameters
        self._realsense_K = np.array([[605.2772, 0, 321.4230],[0, 604.9025, 245.44498],[0, 0, 1]], dtype=np.float32)

        # Load SMPL texture
        self._texture_array = texture_load_image(self._smpl_texture_path, load_alpha=self._smpl_texture_load_alpha)

        # Replace texture with solid color (for now...)
        self._texture_array[:, :, 0] = 242 # R
        self._texture_array[:, :, 1] = 190 # G
        self._texture_array[:, :, 2] = 177 # B

        # Create offscreen renderer
        fxy = self._realsense_K[0, 0] #geometry_fov_to_f(self._camera_fov_vertical, self._viewport_height)

        self._cfg_offscreen = renderer_create_settings_offscreen(self._viewport_width, self._viewport_height)
        self._cfg_scene = renderer_create_settings_scene()
        self._cfg_camera = renderer_create_settings_camera(fxy, fxy, self._viewport_width // 2, self._viewport_height // 2)
        self._cfg_camera_transform = renderer_create_settings_camera_transform(distance=0, pitch=180, min_pitch=-180, max_pitch=180, znear=0)
        self._cfg_lamp = renderer_create_settings_lamp()
        self._cfg_smpl_model = renderer_create_settings_smpl_model(self._smpl_uv_path, self._texture_array.shape, self._smpl_model_path, 10, self._device)
        self._cfg_smpl_filter_bb = renderer_create_settings_smpl_filter_bounding_box()#[0, 0, self._viewport_width, self._viewport_height], [smpl_joints_openpose.MidHip, smpl_joints_openpose.LHip, smpl_joints_openpose.RHip], [0.4, 0.3, 0.3], 0.59)
        self._cfg_smpl_filter_ff = renderer_create_settings_smpl_filter_forward_face()
        self._cfg_smpl_filter_ef = renderer_create_settings_smpl_filter_exponential_single([0.25, 0.25, 0.25, 0.9])
        self._cfg_smpl_filter_ow = renderer_create_settings_smpl_filter_fixed_joints()

        self._offscreen_renderer = renderer_context(self._cfg_offscreen, self._cfg_scene, self._cfg_camera, self._cfg_camera_transform, self._cfg_lamp, self._cfg_smpl_model, None, self._cfg_smpl_filter_bb, self._cfg_smpl_filter_ff, self._cfg_smpl_filter_ef, None, None, self._cfg_smpl_filter_ow, None)
        #self._offscreen_renderer = renderer_context(cfg_offscreen, cfg_scene, cfg_camera, cfg_camera_transform, cfg_lamp, cfg_smpl_model)

        # Create sample text texture
        font = texture_load_font(self._text_font_name, self._text_font_size)
        self._test_text = texture_create_multiline_text(self._text_content, font, self._text_font_color, self._text_canvas_color, self._text_stroke_width, self._text_line_spacing)
        self._test_text = texture_pad(self._test_text, self._text_pad_horizontal_ratio, self._text_pad_vertical_ratio, self._text_canvas_color)

        # SMPL regions
        self._smpl_regions = ['body_center', 'thigh_left', 'thigh_right', 'lower_leg_left', 'lower_leg_right', 'foot_left', 'foot_right', 'head_center', 'upper_arm_left', 'upper_arm_right', 'lower_arm_left', 'lower_arm_right']
        self._smpl_region_index = 0
        self._smpl_region = self._smpl_regions[self._smpl_region_index]

        # Create UI elements
        #self._cursor_mesh = trimesh.creation.cone(radius=self._cursor_radius, height=self._cursor_height)
        #self._cursor_pose = np.eye(4, dtype=np.float32)
        self._cursor_offset = 0
        self._cursor_angle = 0

        # True by default, can be toggled off via curses menu
        self._enable_pose_estimation = True

        self._last_valid_pose = None
        self._smpl_empty_reload = 5
        self._smpl_empty_counter = 5
        self._rotate_image_code = 0

    def _render_body_mesh(self, pose: dict):
        if (pose['status'] != 'success'):
            if (self._smpl_empty_counter <= 0):
                smpl_pose = None
            else:
                self._smpl_empty_counter -= 1
                smpl_pose = self._last_valid_pose
        else:
            self._smpl_empty_counter = self._smpl_empty_reload
            smpl_pose = pose

        # Identity pose (from smpl)
        smpl_mesh_pose = np.eye(4, 4, dtype=np.float32)

        if (smpl_pose is not None):
            #smpl_params, smpl_K = self._offscreen_renderer.smpl_unpack(smpl_pose)
            #smpl_ok, smpl_result = self._offscreen_renderer.smpl_get_mesh(smpl_params, smpl_K.T, self._realsense_K.T)
            smpl_meshes = self._offscreen_renderer.smpl_get_meshes(smpl_pose, self._realsense_K.T)
            smpl_data = smpl_meshes.get('patient', None)
            smpl_ok = smpl_data is not None


            if (smpl_ok):
                self._last_valid_pose = smpl_pose
                #smpl_data = smpl_result.at(0)

                # Add SMPL mesh to the main scene
                smpl_mesh_id = self._offscreen_renderer.mesh_add_smpl('smpl', 'patient', smpl_data, self._texture_array, smpl_mesh_pose)

                # Finalize SMPL painting
                # Compute painted texture
                self._offscreen_renderer.smpl_paint_flush(smpl_mesh_id)
                # Remove painting for next frame (comment out to keep paintings across frames)
                self._offscreen_renderer.smpl_paint_clear(smpl_mesh_id)

                # Finalize mesh processing
                self._offscreen_renderer.mesh_present(smpl_mesh_id)
        
        # Render
        color, depth = self._offscreen_renderer.scene_render()
        self._offscreen_renderer.mesh_remove_all()
        color = color.copy()

        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        if (self._rotate_image_code == 0):
            pass
        elif (self._rotate_image_code == 1):
            color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
        elif (self._rotate_image_code == 2):
            color = cv2.rotate(color, cv2.ROTATE_180)
        elif (self._rotate_image_code == 3):
            color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return color

    

if __name__ == "__main__":
    x = demo()
    smpl_message_path = os.path.join('./data/data_dump/1', 'patient_pose_raw.json')
    with open(smpl_message_path, 'rt') as json_file:
        pose_message = json.load(json_file)

    index = 0
    pose_message['status'] = 'success'

    while (True):
        if ((index % 101) == 50):
            pose_message['status'] = 'error'
            print('KILL')
        if ((index % 101) == 100):
            pose_message['status'] = 'success'
            print('RESTORE')
        index += 1

        with x._offscreen_renderer:
            color = x._render_body_mesh(pose_message)
        cv2.imshow("image", color)
        key = cv2.waitKey(33) & 0xFF
        if (key == 27):
            break

