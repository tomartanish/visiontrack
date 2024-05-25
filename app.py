import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2
import numpy as np
import time

st.set_page_config(page_title="Vision Track", page_icon=":world_map:️", layout="wide")

class FaceMeshTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.start_time_good = None
        self.start_time_bad = None
        self.good_eye_level_duration = 0
        self.bad_eye_level_duration = 0
        self.text = "Initializing..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        img_h, img_w, img_c = img.shape
        face_3d = []
        face_2d = []

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
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # Determine head orientation
                if y < -10 or y > 10:
                    self.text = "Not looking at screen"
                elif x < -10 or x > 10:
                    self.text = "Adjust your screen downwards" if x < -10 else "Adjust your screen upwards"
                    if self.start_time_bad is None and self.start_time_good is None:
                        self.start_time_bad = time.time()
                    elif self.start_time_bad is None and self.start_time_good is not None:
                        self.good_eye_level_duration += (time.time() - self.start_time_good)
                        self.start_time_good = None
                        self.start_time_bad = time.time()
                    else:
                        self.bad_eye_level_duration += (time.time() - self.start_time_bad)
                        self.start_time_bad = None
                elif -10 <= x <= 10:
                    self.text = "Good Eye Level"
                    if self.start_time_good is None and self.start_time_bad is None:
                        self.start_time_good = time.time()
                    elif self.start_time_good is None and self.start_time_bad is not None:
                        self.bad_eye_level_duration += (time.time() - self.start_time_bad)
                        self.start_time_bad = None
                        self.start_time_good = time.time()
                    else:
                        self.good_eye_level_duration += (time.time() - self.start_time_good)
                        self.start_time_good = None

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(img, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                cv2.putText(img, self.text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        else:
            self.text = "No face detected"
            cv2.putText(img, self.text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

webrtc_streamer(key="example", video_transformer_factory=FaceMeshTransformer)

# Display metrics
st.write("### Metrics")
st.write(f"Current posture: {FaceMeshTransformer().text}")
st.write(f"Good eye level duration: {FaceMeshTransformer().good_eye_level_duration:.1f} seconds")
st.write(f"Bad eye level duration: {FaceMeshTransformer().bad_eye_level_duration:.1f} seconds")
st.write(f"Total screen time: {FaceMeshTransformer().good_eye_level_duration + FaceMeshTransformer().bad_eye_level_duration:.1f} seconds")

# Footer
footer = """<style>
a:link , a:visited {
    color: blue;
    background-color: transparent;
    text-decoration: underline;
}
a:hover, a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
}
.footer {
    position: fixed;
    left: 2;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: white;
    text-align: left;
}
</style>
<div class="footer">
<p>Made with ❤ by Team Brackets</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
