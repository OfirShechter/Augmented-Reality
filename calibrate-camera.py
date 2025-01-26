import cv2
import numpy as np

# Define the chessboard size
chessboard_size = (9, 6)
frame_interval = 20

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []
imgpoints = []

# Capture video from the default camera
# video_path = 'videos/videos10/cal10.mp4'
video_path = 'videos/working-videos/output_cal_planner.avi'
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

        cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('image points:', len(imgpoints))
# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
if not ret:
    print("Camera calibration failed.")
    exit()
# Save the calibration results
np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

print("Camera calibrated successfully.")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)