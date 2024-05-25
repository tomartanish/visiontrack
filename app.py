import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Initialize the MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class FacePoseDetector(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []

        # Get the result from MediaPipe Face Mesh
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1]
            ])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, _ = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360

            if y < -10 or y > 10:
                text = "Not looking at screen"
            elif x < -10 or x > 10:
                text = "Adjust your screen downwards" if x < -10 else "Adjust your screen upwards"
            else:
                text = "Good Eye Level"

            # Display the nose direction
            nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

st.title("Vision Track")
st.write(
    "Vision Track monitors your eye posture using your webcam, "
    "providing real-time feedback and duration stats to promote healthier screen habits."
)

webrtc_ctx = webrtc_streamer(
    key="face-pose",
    video_processor_factory=FacePoseDetector,
    async_processing=True,
    sendback_audio=False,
)

if webrtc_ctx.video_transformer:
    st.video(webrtc_ctx.state)  # Use the state object to display the video output

st.text("Run the Camera to detect posture.")
