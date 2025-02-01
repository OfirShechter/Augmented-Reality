import copy
from hmac import new
import os
import pickle
import re
import cv2
import numpy as np
import open3d as o3d
import trimesh


class FrameHelpers:
    pickle_base_folder = 'pickle_files'
    chunk_size = 100  # Number of frames per chunk

    @staticmethod
    def get_video_frames_and_params(video_path):
        """
        Get all frames from a video.

        :param video: The video object
        :return: A list of all frames
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise FileNotFoundError(f"Video file not found at {video_path}")

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
        new_size = (frame_width, frame_height)

        # get filename from video path
        filename = video_path.split('/')[-1][:-4]
        pickle_folder_path = f"{FrameHelpers.pickle_base_folder}/{filename}"
        print("pickle folder path:", pickle_folder_path)

        if os.path.exists(pickle_folder_path):
            try:
                frames = []
                chunk_files = sorted(os.listdir(pickle_folder_path), key=lambda x: int(
                    re.search(r'\d+', x).group()))
                for chunk_file in chunk_files:
                    print("chunk file:", chunk_file)
                    chunk_path = os.path.join(pickle_folder_path, chunk_file)
                    with open(chunk_path, 'rb') as f:
                        frames.extend(pickle.load(f))
                return frames, new_size
            except (EOFError, pickle.UnpicklingError):
                print("Pickle file is corrupted. Reprocessing the video.")

        os.makedirs(pickle_folder_path, exist_ok=False)
        print("folder path:", pickle_folder_path)

        success, frame = video.read()
        frames = []
        chunk_frames = []
        print("video was found:", success)
        i = 0
        chunk_index = 0
        while success:
            print("frame:", i)
            frame = cv2.resize(frame, new_size)
            chunk_frames.append(frame)
            cv2.imshow('Frame', frame)
            i += 1
            success, frame = video.read()
            # Save chunk if it reaches the chunk size
            if len(chunk_frames) == FrameHelpers.chunk_size or not success:
                chunk_path = os.path.join(
                    pickle_folder_path, f'chunk_{chunk_index}.pkl')
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_frames, f)
                frames.extend(chunk_frames)
                chunk_frames = []
                chunk_index += 1

        return frames, new_size

    @staticmethod
    def get_video_writer(output_video_path, video_capture, frame_size):
        """
        Get a video writer object.

        :param video_path: The path to the video
        :param frame_rate: The frame rate
        :param frame_size: The frame size
        :return: The video writer object
        """
        frame_rate = video_capture.get(
            cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    @staticmethod
    def render_model(mesh, rvec, tvec, intrinsic, obj_center):
        """Render the 3D model with Open3D from the estimated camera viewpoint."""
        r_mat = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        # cent = mesh.get_center()
        cent = obj_center
        cent = (0,0,0)
        print(cent)
        scale_factor = 0.015
        mesh.scale(scale_factor, center=cent)
        # mesh = mesh.rotate(r_mat, center=cent)
        # # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        # Adjust rotation matrix for Open3D
        R_conversion = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        R_o3d = R @ R_conversion  # Convert OpenCV R to Open3D R
        # Build Open3D transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R_o3d @ r_mat
        tvec = tvec.flatten()
        tvec = [tvec[0], tvec[1], tvec[2]]
        transformation_matrix[:3, 3] = tvec
        print("tvec", tvec)
        # mesh.transform(transformation_matrix)
        # # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        coordinate_frame.compute_vertex_normals()
        coordinate_frame.scale(scale_factor, center=cent)
        # Create a hidden Open3D visualizer
        vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False)
        vis.create_window(visible=False, width=intrinsic.width, height=intrinsic.height)

        # Add lighting (Phong shading)
        opt = vis.get_render_option()
        opt.light_on = True  # Enable lighting
        opt.background_color = np.array([255, 0, 0])  # Red background
        vis.add_geometry(mesh)

        # Get the view control to set the camera intrinsics
        ctr = vis.get_view_control()
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = intrinsic
        cam_params.extrinsic = transformation_matrix  # Aligns object with camera
        ctr.set_lookat(cent)
        
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
                
        # Capture image from Open3D
        # ctr.set_zoom(0.5)
        # ctr.set_lookat([0, 0, 0])
        # ctr.set_up([0, -1, 0])
        # ctr.set_front([0, 0, -1])

        print("before run")
        vis.poll_events()
        vis.update_renderer()
        # vis.run()
        print("after run")
        render = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # Convert Open3D float buffer to OpenCV format
        render = (np.array(render) * 255).astype(np.uint8)
        # Convert from RGB to BGR
        render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)

        return render