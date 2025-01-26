# ======= imports
import cv2
import numpy as np

from .helpers.frame_helpers import FrameHelpers

# ======= constants
feature_extractor = cv2.SIFT_create(nfeatures=500, edgeThreshold = 20)
bf = cv2.BFMatcher()

replace_image_path = 'images/castle.webp'
replace_img = cv2.imread(replace_image_path)
if replace_img is None:
    raise FileNotFoundError(f"Replace image not found at {replace_image_path}")
# # taked data previusly saved using: np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
# calibration_data_path = 'calibration_data.npz'
# # load camera calibration data
# calibration_data = np.load(calibration_data_path) # keys: mtx, dist, rvecs, tvecs
# mtx, dist, rvecs, tvecs = calibration_data['mtx'], calibration_data['dist'], calibration_data['rvecs'], calibration_data['tvecs']

# === template image keypoint and descriptors
template_image_path = 'images/features_page-0001.jpg'
template_img = cv2.imread(template_image_path)
template_img = cv2.resize(
    template_img, (template_img.shape[1] // 2, template_img.shape[0] // 2))
replace_img = cv2.resize(
    replace_img, (template_img.shape[1], template_img.shape[0]))
if template_img is None:
    raise FileNotFoundError(
        f"Template image not found at {template_image_path}")

gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
template_keypoints, template_descriptors = feature_extractor.detectAndCompute(
    gray_template, None)

rgp_template = cv2.drawKeypoints(
    gray_template, template_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Template image with keypoints', rgp_template)

# ===== video input, output and metadata
step_size = 1
frame_helpers = FrameHelpers()
video_path = 'videos/videos10/video10.mp4'
frames, new_size = frame_helpers.get_video_frames_and_params(video_path)
print("frames len:", len(frames))
input_video = cv2.VideoCapture(video_path) 

output_video_path = 'videos/output_perspective_warping.avi'
fps = input_video.get(cv2.CAP_PROP_FPS) // step_size
# Create a VideoWriter for saving output
output_writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    new_size
)
# ========== run on all frames
frame_index = 0
alpha = 0.4  # Smoothing factor for EMA
prev_H = None
while frame_index < len(frames):
    print("----------------------frame index:", frame_index)

    frame = frames[frame_index].copy()
    # ====== find keypoints matches of frame and template
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with chosen feature_extractor
    kp_frame, desc_frame = feature_extractor.detectAndCompute(
        gray_frame, None)

    test_frame = cv2.drawKeypoints(
        rgb_frame, kp_frame, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Frame with keypoints', test_frame)

    # ======== find homography
    # Match descriptors using KNN Matcher
    matches = bf.knnMatch(template_descriptors, desc_frame, k=2)

    # Apply ratio test
    good_match_arr = []
    for m, n in matches:
        if m.distance < 1 * n.distance:
            good_match_arr.append(m)
    # take top 10 with lowest distance
    good_match_arr = sorted(
        good_match_arr, key=lambda x: x.distance)[:75]
    print("good match len:", len(good_match_arr)) 
    if len(good_match_arr) < 4:
        print("Not enough matches found- len: ", len(good_match_arr))
        frame_index += step_size
        continue

    # Extract matched keypoints
    src_pts = np.array(
        [template_keypoints[m.queryIdx].pt for m in good_match_arr])
    dst_pts = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr])
        
    frame = frames[frame_index].copy()
    # Draw matches for visualization
    match_img = cv2.drawMatches(template_img, template_keypoints, frame, kp_frame,
                                good_match_arr, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', match_img)

    # Compute Homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # ++++++++ do warping of another image on template image
    if H is not None:
        # Smooth the transition using exponential moving average
        if prev_H is not None:
            H = alpha * H + (1 - alpha) * prev_H
        prev_H = H
        
        # Warp the cat image using the homography
        warped_replace = cv2.warpPerspective(
            replace_img, H, (frame.shape[1], frame.shape[0]))

        # Create a mask for overlay blending
        mask_warped = np.zeros_like(frame, dtype=np.uint8)
        frame[warped_replace > 0] = 0
        # Blend the warped image onto the frame
        frame = frame + warped_replace
        # cv2.putText(frame, f"Frame: {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("wraped_replace", warped_replace)
    else:
        print("Homography not found")
        frame_index + step_size
        pass # take prev frame- no homography found
    # =========== plot and save frame
    # Show output frame
    cv2.imshow('Warped Overlay', frame)

    # Save frame to output video
    output_writer.write(frame)

    frame_index += step_size
    # key = cv2.waitKey(0) & 0xFF
    # if key == ord('l'):  # 'l' for next frame
    #     frame_index = min(frame_index + step_size, len(frames) - step_size)
    #     print("l changed frame index to:", frame_index)
    #     continue
    # elif key == ord('k'):  # 'k' for previous frame
    #     frame_index = max(frame_index - step_size, 0)
    #     continue
    # elif key == ord('q'):  # 'q' to quit
    #     break
    # else:
    #     print('UNESSIGNED KEY PRESSED:', key)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# ======== end all
pass
