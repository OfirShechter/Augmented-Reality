# ======= imports
import cv2
import numpy as np

# ======= constants
# taked data previusly saved using: np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
calibration_data_path = 'calibration_data.npz'
# load camera calibration data
calibration_data = np.load(calibration_data_path) # keys: mtx, dist, rvecs, tvecs
mtx, dist, rvecs, tvecs = calibration_data['mtx'], calibration_data['dist'], calibration_data['rvecs'], calibration_data['tvecs']

# === template image keypoint and descriptors
template_image_path = 'images/feature-full.webp'
template_img = cv2.imread(template_image_path)
if template_img is None:
    raise FileNotFoundError(f"Template image not found at {template_image_path}")

gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
feature_extractor = cv2.SIFT_create()
template_keypoints, template_descriptors = feature_extractor.detectAndCompute(gray_template, None)

rgp_template = cv2.drawKeypoints(template_img, template_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Template image with keypoints', rgp_template)

# ===== video input, output and metadata
video_path = 'videos/planar_surface_obj.mp4'
input_video = cv2.VideoCapture(video_path)
if not input_video.isOpened():
    raise FileNotFoundError(f"Video file not found at {video_path}")

output_video_path = 'videos/output_perspective_warping.avi'
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)
# Create a VideoWriter for saving output
output_writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)
# ========== run on all frames
while True:
    success, frame = input_video.read()
    if not success:
        break
    # ====== find keypoints matches of frame and template
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    feature_extractor = cv2.SIFT_create()

    # find the keypoints and descriptors with chosen feature_extractor
    kp_frame, desc_frame = feature_extractor.detectAndCompute(gray_frame, None)

    test_frame = cv2.drawKeypoints(rgb_frame, kp_frame, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow('Frame with keypoints', test_frame)
    wait_key = cv2.waitKey(1) & 0xFF
    if wait_key == ord('q'):
        break
    pass

    # ======== find homography
    # also in SIFT notebook
    pass

    # ++++++++ do warping of another image on template image
    # we saw this in SIFT notebook
    pass

    # =========== plot and save frame
    pass

# ======== end all
pass
