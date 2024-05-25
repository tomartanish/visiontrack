import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

class FacePoseDetector(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []
        
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1]
            ])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x = angles[0] * 360
            y = angles[1] * 360

            if y < -10 or y > 10:
                text = "Not looking at screen"
            elif x < -10 or x > 10:
                text = "Adjust your screen downwards" if x < -10 else "Adjust your screen upwards"
            else:
                text = "Good Eye Level"

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
    video_transformer_factory=FacePoseDetector,
    async_transform=True,
)

if webrtc_ctx.video_transformer:
    st.image(webrtc_ctx.video_transformer.state, channels="BGR")  # Display the processed video frames

st.text("Run the Camera to detect posture.")
