#------------------------------------------------------------------------------
# SMPL paint demo
#
# Keyboard Controls:
# 1: Focus next SMPL region
# 2/3: Increment/Decrement Cursor offset
# 4/5: Increment/Decrement Cursor yaw
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
import trimesh
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
        
        self._viewport_width = 1280
        self._viewport_height = 720

        self._camera_fov_vertical = np.pi / 3
        self._camera_yaw_increment = 10
        self._camera_pitch_increment = 10
        self._camera_distance_increment = 0.1
        self._camera_use_plane = True
        self._camera_focus_factor = 1.25

        self._text_font_name = 'FreeSans.ttf'
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
        self._joint_projection_thickness = -1

        self._fps_period = 2.0
        # End Settings --------------------------------------------------------

        # Load SMPL texture
        self._texture_array = smplpact.texture_load_image(self._smpl_texture_path, load_alpha=self._smpl_texture_load_alpha)

        # Create sample text texture
        font = smplpact.texture_load_font(self._text_font_name, self._text_font_size)
        self._test_text = smplpact.texture_create_multiline_text(self._text_content, font, self._text_font_color, self._text_canvas_color, self._text_stroke_width, self._text_line_spacing)
        self._test_text = smplpact.texture_pad(self._test_text, self._text_pad_horizontal_ratio, self._text_pad_vertical_ratio, self._text_canvas_color)

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

        # SMPL regions
        self._smpl_regions = ['body_center', 'thigh_left', 'thigh_right', 'lower_leg_left', 'lower_leg_right', 'foot_left', 'foot_right', 'head_center', 'upper_arm_left', 'upper_arm_right', 'lower_arm_left', 'lower_arm_right']
        self._smpl_region_index = 0
        self._smpl_region = self._smpl_regions[self._smpl_region_index]
        
        # Create UI elements
        self._cursor_mesh = trimesh.creation.cone(radius=self._cursor_radius, height=self._cursor_height)
        self._cursor_pose = np.eye(4, dtype=np.float32)
        self._cursor_offset = 0
        self._cursor_angle = 0

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
        smpl_ok, smpl_result = self._offscreen_renderer.smpl_get_mesh(smpl_params, smpl_K.T, smpl_K.T)
        smpl_data = smpl_result.at(0)

        # Compute pose to set mesh upright
        # Poses convert from object to world
        smpl_mesh = smplpact.mesh_create(smpl_data.vertices, smpl_data.faces)
        smpl_mesh_pose = np.linalg.inv(smplpact.smpl_mesh_chart_openpose(smpl_mesh, smpl_data.joints).create_frame('body_center').to_pose()).T

        # Add SMPL mesh to the main scene
        smpl_mesh_id = self._offscreen_renderer.mesh_add_smpl('smpl', 'patient', smpl_data, self._texture_array, smpl_mesh_pose)

        # Change focus region when key is pressed
        # Camera orientation is preserved
        # Cursor coordinates are reset
        smpl_next_region = self._smpl_regions[self._smpl_region_index]
        
        if (smpl_next_region != self._smpl_region):
            smpl_frame = self._offscreen_renderer.smpl_chart_create_frame(smpl_mesh_id, smpl_next_region)
            
            focus_center = smplpact.math_transform_points(smpl_frame.center, smpl_mesh_pose.T, inverse=False)
            focus_points = smplpact.math_transform_points(smpl_frame.points, smpl_mesh_pose.T, inverse=False)
            focus_distance = self._offscreen_renderer.camera_solve_fov_z(focus_center, focus_points)

            self._offscreen_renderer.camera_adjust_parameters(center=focus_center, distance=self._camera_focus_factor * focus_distance, relative=False)
            self._cursor_offset = 0
            self._cursor_angle = 0
            self._smpl_region = smpl_next_region

        # Map current cylindrical coordinates to SMPL mesh point and face
        smpl_frame = self._offscreen_renderer.smpl_chart_create_frame(smpl_mesh_id, self._smpl_region)
        cursor_anchor = self._offscreen_renderer.smpl_chart_from_cylindrical(smpl_mesh_id, smpl_frame, self._cursor_offset, self._cursor_angle)

        # Set cursor position based on cylindrical coordinates
        # smpl_anchor.point is None when outside mesh
        local_cursor_orientation = np.vstack((np.cross(smpl_frame.up, -cursor_anchor.direction), smpl_frame.up, -cursor_anchor.direction))
        local_cursor_position = (cursor_anchor.point + self._cursor_height * cursor_anchor.direction) if (cursor_anchor.point is not None) else cursor_anchor.position
        
        self._cursor_pose[0:3, 0:3] = smplpact.math_transform_bearings(local_cursor_orientation, smpl_mesh_pose.T, inverse=False).T
        self._cursor_pose[0:3, 3:4] = smplpact.math_transform_points(local_cursor_position, smpl_mesh_pose.T, inverse=False).T
        
        # Add cursor to the main scene
        cursor_mesh_id = self._offscreen_renderer.mesh_add_user('ui', 'cursor', self._cursor_mesh, self._cursor_pose)

        # Perform ray casting from camera to mesh
        # This will be used to paint on the mesh where the camera is looking
        camera_pose = self._offscreen_renderer.camera_get_transform_local()
        camera_position = camera_pose[:3, 3:4].T
        camera_forward = -camera_pose[:3, 2:3].T
        camera_anchor = self._offscreen_renderer.mesh_operation_raycast(smpl_mesh_id, camera_position, camera_forward)

        # If raycast did not intersect the mesh then use the closest mesh point
        if (camera_anchor.point is None):
            camera_anchor = self._offscreen_renderer.mesh_operation_closest(smpl_mesh_id, camera_position)

        # Paint SMPL mesh
        # Paint decal at cursor position
        if (cursor_anchor.point is not None):
            # Fix loose degree of freedom about face normal, required to maintain consistent orientation
            decal_align_prior = self._offscreen_renderer.smpl_paint_decal_align_prior(smpl_mesh_id, cursor_anchor, smpl_frame.up, smpl_frame.front)
            self._offscreen_renderer.smpl_paint_decal_solid(smpl_mesh_id, cursor_anchor, self._test_text, decal_align_prior, self._decal_angle, self._decal_size, double_cover_test=False, fill_test=0.25)
        # Paint circular gradient at camera mesh intersection/closest
        if (camera_anchor.point is not None):
            self._offscreen_renderer.smpl_paint_brush_gradient(smpl_mesh_id, camera_anchor, self._brush_size, self._brush_color_center, self._brush_color_edge, self._brush_hardness, fill_test=0.25)
            # Solid color option
            #self._offscreen_renderer.smpl_paint_brush_solid(smpl_mesh_id, camera_anchor, self._brush_size, self._brush_color_center, fill_test=0.25)
        
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

        # Render focused joints
        #color = color.copy()
        #world_points = smplpact.math_transform_points(smpl_frame.points, smpl_mesh_pose.T, inverse=False)
        #image_points, local_points, camera_points = self._offscreen_renderer.camera_project_points(world_points, convention=(1, -1, -1))

        #for i in range(0, image_points.shape[0]):
        #    if (local_points[i, 2] > 0):
        #        center = (int(image_points[i, 0]), int(image_points[i, 1]))
        #        color = cv2.circle(color, center, self._joint_projection_radius, self._joint_projection_color, self._joint_projection_thickness)

        # Show rendered image
        cv2.imshow('SMPL Paint Demo', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF

        if (key == 49): # 1:
            self._smpl_region_index = (self._smpl_region_index + 1) % len(self._smpl_regions)
 
        if (key == 50): # 2:
            self._cursor_offset += self._cursor_offset_increment
        if (key == 51): # 3
            self._cursor_offset -= self._cursor_offset_increment
        if (key == 52): # 4 
            self._cursor_angle += self._cursor_angle_increment
        if (key == 53): # 5
            self._cursor_angle -= self._cursor_angle_increment

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

