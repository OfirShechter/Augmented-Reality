from hmac import new
import os
import pickle
import re
import cv2


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
