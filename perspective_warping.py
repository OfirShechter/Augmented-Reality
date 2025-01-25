# ======= imports
import cv2
import numpy as np

# ======= constants
feature_extractor = cv2.SIFT_create()
bf = cv2.BFMatcher()

replace_image_path = 'images/israel-flag.png'
replace_img = cv2.imread(replace_image_path)
if replace_img is None:
    raise FileNotFoundError(f"Replace image not found at {replace_image_path}")
# # taked data previusly saved using: np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
# calibration_data_path = 'calibration_data.npz'
# # load camera calibration data
# calibration_data = np.load(calibration_data_path) # keys: mtx, dist, rvecs, tvecs
# mtx, dist, rvecs, tvecs = calibration_data['mtx'], calibration_data['dist'], calibration_data['rvecs'], calibration_data['tvecs']

# === template image keypoint and descriptors
template_image_path = 'images/usa-flag.png'
template_img = cv2.imread(template_image_path)
replace_img = cv2.resize(replace_img, (template_img.shape[1], template_img.shape[0]))
if template_img is None:
    raise FileNotFoundError(f"Template image not found at {template_image_path}")

gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
template_keypoints, template_descriptors = feature_extractor.detectAndCompute(gray_template, None)

rgp_template = cv2.drawKeypoints(template_img, template_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Template image with keypoints', rgp_template)

# ===== video input, output and metadata
video_path = 'videos/flag-video.mp4'
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

    # find the keypoints and descriptors with chosen feature_extractor
    kp_frame, desc_frame = feature_extractor.detectAndCompute(gray_frame, None)

    test_frame = cv2.drawKeypoints(rgb_frame, kp_frame, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow('Frame with keypoints', test_frame)
    
    # ======== find homography
    # Match descriptors using KNN Matcher
    matches = bf.knnMatch(template_descriptors, desc_frame, k=2)

    # Apply ratio test
    good_match_arr = []
    pairs_match = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_match_arr.append(m)
            pairs_match.append([m, n])

    if len(good_match_arr) < 4:
        continue
    # show only 30 matches
    im_matches = cv2.drawMatchesKnn(
        rgb_frame,
        kp_frame,
        rgp_template,
        template_keypoints,
        pairs_match[0:30],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    
    cv2.imshow('Matches', im_matches)

    # Extract matched keypoints
    src_pts = np.array([template_keypoints[m.queryIdx].pt for m in good_match_arr])
    dst_pts = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr])

    # Compute Homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ++++++++ do warping of another image on template image
    if H is not None:
        # Warp the cat image using the homography
        warped_replace = cv2.warpPerspective(replace_img, H, (frame.shape[1], frame.shape[0]))

        # Create a mask for overlay blending
        mask_warped = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillConvexPoly(mask_warped, np.int32(dst_pts), (255, 255, 255))

        # Blend the warped image onto the frame
        frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask_warped)) + warped_replace

    # =========== plot and save frame
    # Show output frame
    cv2.imshow('Warped Overlay', frame)

    # Save frame to output video
    output_writer.write(frame)

    wait_key = cv2.waitKey(1) & 0xFF
    if wait_key == ord('q'):
        break

# ======== end all
pass
