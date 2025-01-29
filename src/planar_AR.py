# ======= imports
import cv2
import numpy as np

from .helpers.frame_helpers import FrameHelpers

# ======= constants
feature_extractor = cv2.SIFT_create(nfeatures=500, edgeThreshold = 20)
bf = cv2.BFMatcher()

replace_image_path = 'images/pattern_chessboard.png'
replace_img = cv2.imread(replace_image_path)
replace_img[replace_img == 0] = 1
if replace_img is None:
    raise FileNotFoundError(f"Replace image not found at {replace_image_path}")
# taked data previusly saved using: np.savez('calibration_data.npz', mtx=mtx, dist=dist)
calibration_data_path = 'calibration_data.npz'
# load camera calibration data
calibration_data = np.load(calibration_data_path) # keys: mtx, dist, rvecs, tvecs
K = calibration_data['mtx']
dist_coeffs = calibration_data['dist']
# ====== cube object points
chessboard_size = (9, 6)
square_size = 2.5  # Cube height in cm
# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Cube 3D points
cube_points = 3 * square_size * np.float32([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, -1], [1, 0, -1], [1, 1, -1], [0, 1, -1]
])

def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


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
        continue # take prev frame- no homography found

    # imgpts = cv2.projectPoints(cube_points, rvecs[frame_index], tvecs[frame_index], mtx, dist)[0]
    # drawn_image = draw_obj(frame, imgpts)
    # Show output frame
    # cv2.imshow('Chessboard', frame)
    
    
    # ++++++++ take subset of keypoints that obey homography (both frame and reference)
    # this is at most 3 lines- 2 of which are really the same
    # HINT: the function from above should give you this almost completely
    gray_wraped_frame = cv2.cvtColor(warped_replace, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_wraped_frame, chessboard_size, None)
    if not ret:
        print("Chessboard not found in warped image")
        frame_index += step_size
        continue
    sub_corners = cv2.cornerSubPix(gray_wraped_frame, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
    # `cv2.solvePnP` is a function that receives:
    # - xyz of the template in centimeter in camera world (x,3)
    # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    # - camera K
    # - camera dist_coeffs
    # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
    #
    # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    # For this we just need the template width and height in cm.
    #
    # this part is 2 rows
    ret, rvec, tvec = cv2.solvePnP(objp, sub_corners, K, dist_coeffs)

    # ++++++ draw object with r_vec and t_vec on top of rgb frame
    # We saw how to draw cubes in camera calibration. (copy paste)
    # after this works you can replace this with the draw function from the renderer class renderer.draw() (1 line)
    if ret:
        imgpts = cv2.projectPoints(cube_points, rvec, tvec, K, dist_coeffs)[0]
        # # Draw the projected points on the frame
        # for pt in imgpts:
        #     print("pt:", pt)
        #     pt = tuple(pt.ravel().astype(int))
        #     cv2.circle(frame, pt, 5, (0, 255, 0), 5)
        
        # cv2.imshow('projectPoints', frame)
        img_with_cube = draw_cube(frames[frame_index].copy(), imgpts)
        cv2.imshow('Cube', img_with_cube)

    # =========== plot and save frame
    pass

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

    # Save frame to output video
    output_writer.write(img_with_cube)
# ======== end all
pass
