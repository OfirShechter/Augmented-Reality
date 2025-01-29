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
                chunk_files = sorted(os.listdir(pickle_folder_path), key=lambda x: int(re.search(r'\d+', x).group()))
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
                chunk_path = os.path.join(pickle_folder_path, f'chunk_{chunk_index}.pkl')
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
    def render_model(mesh, camera_matrix, rvec, tvec, image_size):
        """Render the 3D model with Open3D from the estimated camera viewpoint."""
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Build Open3D transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = tvec.flatten()

        # Apply transformation
        mesh.transform(transformation_matrix)
       # Ensure mesh has vertex colors
        if not mesh.has_vertex_colors():
            print("⚠️ Mesh has no vertex colors! Applying default color.")
            mesh.paint_uniform_color([0.8, 0.3, 0.3])  # Default reddish color

        # Ensure mesh has normals for shading
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Create a hidden Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=image_size[1], height=image_size[0])

        # Add lighting (Phong shading)
        opt = vis.get_render_option()
        opt.light_on = True  # Enable lighting
        opt.background_color = np.array([1, 1, 1])  # White background

        # Add the model to the scene
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        # Capture image from Open3D
        render = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # Convert Open3D float buffer to OpenCV format
        render = (np.array(render) * 255).astype(np.uint8)
        render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
        
        return render
    
    def load_glb_with_materials(file_path):
        """
        Load a GLB file with its colors and textures using Open3D and trimesh
        
        Args:
            file_path (str): Path to the GLB file
            
        Returns:
            o3d.geometry.TriangleMesh: Open3D mesh with materials
        """
        # First load with trimesh to get the full scene with materials
        print(file_path)
        scene = trimesh.load(file_path, file_type='glb')
        
        # Initialize an empty Open3D mesh
        final_mesh = o3d.geometry.TriangleMesh()
        
        # Process each geometry in the scene
        for name, geometry in scene.geometry.items():
            # Convert trimesh geometry to Open3D geometry
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(geometry.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(geometry.faces)
            print("geometry.visual.kind:", geometry.visual.kind)
            # Add vertex colors if available
            if geometry.visual.kind == 'vertex':
                mesh.vertex_colors = o3d.utility.Vector3dVector(geometry.visual.vertex_colors[:, :3])
            
            # Add texture if available
            if geometry.visual.kind == 'texture':
                mesh.triangle_uvs = o3d.utility.Vector2dVector(geometry.visual.uv)
                if hasattr(geometry.visual.material, 'image'):
                    texture = geometry.visual.material.image
                    mesh.textures = [o3d.geometry.Image(texture)]
            
            # Combine the mesh with the final mesh
            final_mesh += mesh
        
        return final_mesh