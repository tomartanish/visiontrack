import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import streamlit_shadcn_ui as ui
import matplotlib.pyplot as plt
import base64

st.set_page_config(
    page_title="Vision Track",
    page_icon=":world_map:️",
    layout="wide",
)
left, right = st.columns(2)

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

# Read the image and encode it to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

bg_image_base64 = get_base64_image("bg.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewBlockContainer"] {{
    background-image: url("data:image/png;base64,{bg_image_base64}");
    background-size: cover;
}}

[data-testid="stHeader"] {{
    background-color: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

element1 = """
[data-testid="element-container"] {
    background-color: rgba(0,0,0,0);
}
"""

good_duration = 0
bad_duration = 0
total_screentime = good_duration + bad_duration

with left:
    "# Vision Track"
    with ui.element("div", className="flex gap-2", key="buttons_group1"):
        ui.element("link_button", text="Diagnosia | Home", url="https://diagnosia.netlify.app", variant="outline", key="btn1")

    "Vision Track monitors your eye posture using your webcam, providing real-time feedback and duration stats to promote healthier screen habits."

# Initialize the MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if 'checkbox_state' not in st.session_state:
    st.session_state['checkbox_state'] = False

if 'run_indefinitely' not in st.session_state:
    st.session_state['run_indefinitely'] = False

def toggle_checkbox():
    st.session_state['checkbox_state'] = not st.session_state['checkbox_state']

# Streamlit app title
with right:
    # Checkbox for indefinite running
    run_indefinitely = st.checkbox('Run Indefinitely', value=st.session_state['run_indefinitely'])
    st.session_state['run_indefinitely'] = run_indefinitely

    # Slider placeholder
    slider_placeholder = st.empty()

    # Button to toggle the tool
    run = st.button("Toggle Tool", on_click=toggle_checkbox, type="primary")
    
    posture_placeholder = st.empty()
    goodtime_placeholder = st.empty()
    badtime_placeholder = st.empty()
    totalgoodtime_placeholder = st.empty()
    totalbadtime_placeholder = st.empty()
    totaltime_placeholder = st.empty()
    cols = st.columns(3)

# Disable the slider if the camera is running indefinitely
if st.session_state['run_indefinitely']:
    total_time = None
    slider_placeholder.write("Running indefinitely...")
else:
    total_time = slider_placeholder.slider('Set the total time (seconds)', min_value=1, max_value=60, value=10)
    st.session_state['total_time'] = total_time

cap = cv2.VideoCapture(0)

# Variables to track time
start_time_good = None
start_time_bad = None
start_time_total = None
bad_eye_level_duration = 0
good_eye_level_duration = 0
text = "Initializing..."

# Check if 'run' checkbox is selected
with left:
    FRAME_WINDOW = st.image([])
    if st.session_state['checkbox_state']:
        start_time = time.time()
        start_time_bad = time.time()
        start_time_good = time.time()
        if total_time:
            end_time = start_time + total_time
        else:
            end_time = None

    while st.session_state['checkbox_state'] and (end_time is None or time.time() < end_time):
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Get the result from MediaPipe Face Mesh
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
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
                    text = "Not looking at screen"
                elif x < -10 or x > 10:
                    text = "Adjust your screen downwards" if x < -10 else "Adjust your screen upwards"
                    if start_time_bad is None and start_time_good is None:
                        start_time_bad = time.time()
                    elif start_time_bad is None and start_time_good is not None:
                        good_eye_level_duration += (time.time() - start_time_good)
                        start_time_good = None
                        start_time_bad = time.time()
                    else:
                        bad_eye_level_duration += (time.time() - start_time_bad)
                        start_time_bad = None
                elif -10 <= x <= 10:
                    text = "Good Eye Level"
                    if start_time_good is None and start_time_bad is None:
                        start_time_good = time.time()
                    elif start_time_good is None and start_time_bad is not None:
                        bad_eye_level_duration += (time.time() - start_time_bad)
                        start_time_bad = None
                        start_time_good = time.time()
                    else:
                        good_eye_level_duration += (time.time() - start_time_good)
                        start_time_good = None

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        else:
            text = "No face detected"
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        FRAME_WINDOW.image(image)

        good_duration = round(good_eye_level_duration, 1)
        bad_duration = round(bad_eye_level_duration, 1)
        total_screentime = good_duration + bad_duration

        with right:
            posture_placeholder.metric(label="Current posture", value=text)
            goodtime_placeholder.metric(label="Good eye level duration", value=f"{good_duration} seconds")
            badtime_placeholder.metric(label="Bad eye level duration", value=f"{bad_duration} seconds")
            totaltime_placeholder.metric(label="Total screen time", value=f"{total_screentime} seconds")

cap.release()
cv2.destroyAllWindows()
