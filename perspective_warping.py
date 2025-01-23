# ======= imports
import numpy as np

# ======= constants
# taked data previusly saved using: np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
calibration_data_path = 'calibration_data.npz'
# load camera calibration data
calibration_data = np.load(calibration_data_path)
print(calibration_data)

# === template image keypoint and descriptors
template_image_path = 'images/template.jpg'

# ===== video input, output and metadata
video_path = 'videos/video.mp4'
output_video_path = 'videos/output_video.mp4'


# ========== run on all frames
while True:
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
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
