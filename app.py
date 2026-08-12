import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import tempfile
import os
import subprocess
import imageio_ffmpeg


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YarnX – The Future of Yarn Inspection",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg,
            #f7f9fc 0%,
            #eef4ff 50%,
            #f8fafc 100%
        );
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        color: #172554;
        font-size: 34px;
        font-weight: 800;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-heading {
        color: #172554;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    /* Description */
    .description-text {
        color: #475569;
        font-size: 16px;
        line-height: 1.7;
    }

    /* Result */
    .good-box {
        background: #ecfdf5;
        border: 2px solid #86efac;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 15px;
    }

    .good-title {
        color: #15803d;
        font-size: 28px;
        font-weight: 800;
    }

    .bad-box {
        background: #fff1f2;
        border: 2px solid #fca5a5;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }

    .bad-title {
        color: #dc2626;
        font-size: 28px;
        font-weight: 800;
    }

    .defect-info {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        color: #334155;
        font-size: 17px;
    }

    .waiting-box {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 55px 20px;
        text-align: center;
        margin-top: 20px;
    }

    .waiting-title {
        color: #475569;
        font-size: 23px;
        font-weight: 800;
    }

    .waiting-text {
        color: #64748b;
        font-size: 15px;
        margin-top: 8px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
        font-size: 16px;
    }

    /* Radio */
    div[role="radiogroup"] {
        gap: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-top: 25px;
        padding-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "result_ready" not in st.session_state:
    st.session_state.result_ready = False

if "result_type" not in st.session_state:
    st.session_state.result_type = None

if "result_data" not in st.session_state:
    st.session_state.result_data = None


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"

CONF_THRESHOLD = 0.25


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error("❌ Model file `best.pt` not found.")

    st.warning(
        "Please upload best.pt to the same GitHub folder "
        "where app.py is located."
    )

    st.code(
        "app.py\n"
        "best.pt\n"
        "requirements.txt"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO(str(MODEL_PATH))


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.code(str(e))

    st.stop()


# ============================================================
# GET BEST DETECTION
# ============================================================

def get_best_detection(result):

    detections = []

    if result.boxes is None:
        return None

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        detections.append(
            (
                class_name,
                confidence
            )
        )

    if len(detections) == 0:
        return None

    return max(
        detections,
        key=lambda x: x[1]
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image):

    result = model.predict(
        source=np.array(image),
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    best = get_best_detection(result)

    if best is None:

        return {
            "status": "good"
        }

    annotated = result.plot()

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    return {
        "status": "bad",
        "defect": best[0],
        "confidence": best[1],
        "image": annotated
    }


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(uploaded_video):

    input_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_temp.write(
        uploaded_video.getbuffer()
    )

    input_temp.close()

    input_path = input_temp.name

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():

        os.remove(input_path)

        raise RuntimeError(
            "Could not open uploaded video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 20

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # Temporary AVI
    avi_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".avi"
    )

    avi_temp.close()

    avi_path = avi_temp.name

    fourcc = cv2.VideoWriter_fourcc(
        *"MJPG"
    )

    writer = cv2.VideoWriter(
        avi_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        os.remove(input_path)
        os.remove(avi_path)

        raise RuntimeError(
            "Could not create processed video."
        )

    detected_defects = {}

    frame_number = 0

    progress = st.progress(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = model.predict(
            source=rgb_frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )[0]

        # Save detections
        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = result.names[
                    class_id
                ]

                if class_name not in detected_defects:

                    detected_defects[
                        class_name
                    ] = confidence

                else:

                    detected_defects[
                        class_name
                    ] = max(
                        detected_defects[
                            class_name
                        ],
                        confidence
                    )

        # Draw boxes
        annotated_frame = result.plot()

        writer.write(
            annotated_frame
        )

        frame_number += 1

        if total_frames > 0:

            progress.progress(
                min(
                    frame_number / total_frames,
                    1.0
                )
            )

    progress.empty()

    cap.release()
    writer.release()

    # --------------------------------------------------------
    # GOOD VIDEO
    # --------------------------------------------------------

    if len(detected_defects) == 0:

        try:
            os.remove(input_path)
            os.remove(avi_path)
        except:
            pass

        return {
            "status": "good"
        }


    # --------------------------------------------------------
    # BAD VIDEO
    # --------------------------------------------------------

    best_defect = max(
        detected_defects,
        key=detected_defects.get
    )

    best_confidence = detected_defects[
        best_defect
    ]

    # --------------------------------------------------------
    # Convert AVI -> MP4
    # using imageio-ffmpeg
    # --------------------------------------------------------

    mp4_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    mp4_temp.close()

    mp4_path = mp4_temp.name

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        avi_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        mp4_path
    ]

    try:

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

    except Exception:

        # If conversion fails, use AVI
        mp4_path = avi_path


    # Read final video
    with open(
        mp4_path,
        "rb"
    ) as f:

        processed_video = f.read()


    # Cleanup
    try:

        os.remove(input_path)

        if avi_path != mp4_path:
            os.remove(avi_path)

        os.remove(mp4_path)

    except:
        pass


    return {
        "status": "bad_video",
        "defect": best_defect,
        "confidence": best_confidence,
        "video": processed_video
    }


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    st.markdown(
        '<div class="main-title">'
        '🧶 YarnX – The Future of Yarn Inspection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Smart Yarn Quality Inspection'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TOP SECTION
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 2],
        gap="large"
    )


    with left:

        st.subheader(
            "AI Career for Women (AICW)"
        )

        st.markdown(
            "### Capstone Project"
        )

        st.write("")

        if st.button(
            "🔍  PREDICT",
            use_container_width=True
        ):

            st.session_state.page = 2

            st.session_state.result_ready = False

            st.session_state.result_type = None

            st.session_state.result_data = None

            st.rerun()


    with right:

        st.subheader(
            "Project Description"
        )

        st.write(
            "Yarn quality inspection system using "
            "AI and computer vision to identify "
            "yarn defects from images, camera input, "
            "and videos."
        )


    st.divider()


    # --------------------------------------------------------
    # TEAM INFORMATION
    # --------------------------------------------------------

    team_col, gmail_col, guide_col = st.columns(
        [1.5, 1.5, 1],
        gap="large"
    )


    with team_col:

        st.subheader(
            "TEAM MEMBERS"
        )

        st.write(
            "1. Gutti.pavani devi Priya"
        )

        st.write(
            "2. Somasani.sasi priya"
        )

        st.write(
            "3. Galidevara.Rama Devi"
        )

        st.write(
            "4. Rambala.Harshitha sai Lakshmi"
        )


    with gmail_col:

        st.subheader(
            "GMAIL"
        )

        st.write(
            "gutthipavanidevipriya@gmail.com"
        )

        st.write(
            "Sasipriya8090@gmail.com"
        )

        st.write(
            "ramadevigalidevara0gmail.com"
        )

        st.write(
            "harshitharambala3@gmail.com"
        )


    with guide_col:

        st.subheader(
            "GUIDE NAME"
        )

        st.write(
            "Md.Abdul Aziz"
        )

        st.write("")

        st.markdown(
            "**Designation**"
        )

        st.write(
            "Co Lead & Trainer AICW"
        )


    st.markdown(
        '<div class="footer">'
        'YarnX – The Future of Yarn Inspection'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2
# ============================================================

else:

    st.markdown(
        '<div class="main-title">'
        '🧶 YarnX – The Future of Yarn Inspection'
        '</div>',
        unsafe_allow_html=True
    )


    if st.button(
        "← Back to Project"
    ):

        st.session_state.page = 1

        st.session_state.result_ready = False

        st.session_state.result_type = None

        st.session_state.result_data = None

        st.rerun()


    st.divider()


    # ========================================================
    # INPUT / RESULT
    # ========================================================

    input_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # INPUT
    # ========================================================

    with input_col:

        st.subheader(
            "📥 INPUT"
        )


        input_type = st.radio(
            "Select Input Type:",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ],
            horizontal=True
        )


        # ====================================================
        # IMAGE INPUT
        # ====================================================

        if input_type == "🖼️ Image":

            uploaded_image = st.file_uploader(
                "Upload Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="image_upload"
            )


            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")


                st.caption(
                    "ORIGINAL IMAGE"
                )


                # Small preview
                st.image(
                    image,
                    width=360
                )


                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing image..."
                    ):

                        result = analyze_image(
                            image
                        )


                    st.session_state.result_ready = True

                    st.session_state.result_type = (
                        result["status"]
                    )

                    st.session_state.result_data = result

                    st.rerun()


        # ====================================================
        # CAMERA INPUT
        # ====================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Capture Yarn Image"
            )


            if camera_image:

                image = Image.open(
                    camera_image
                ).convert("RGB")


                st.caption(
                    "CAMERA CAPTURE"
                )


                st.image(
                    image,
                    width=360
                )


                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing camera image..."
                    ):

                        result = analyze_image(
                            image
                        )


                    st.session_state.result_ready = True

                    st.session_state.result_type = (
                        result["status"]
                    )

                    st.session_state.result_data = result

                    st.rerun()


        # ====================================================
        # VIDEO INPUT
        # ====================================================

        else:

            uploaded_video = st.file_uploader(
                "Upload Video",
                type=[
                    "mp4",
                    "mov",
                    "avi",
                    "mkv"
                ],
                key="video_upload"
            )


            if uploaded_video:

                st.caption(
                    "ORIGINAL VIDEO"
                )


                # Small video preview
                st.video(
                    uploaded_video
                )


                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        try:

                            result = analyze_video(
                                uploaded_video
                            )

                            st.session_state.result_ready = True

                            st.session_state.result_type = (
                                result["status"]
                            )

                            st.session_state.result_data = result

                            st.rerun()

                        except Exception as e:

                            st.error(
                                "❌ Video processing failed."
                            )

                            st.code(
                                str(e)
                            )


    # ========================================================
    # RESULT
    # ========================================================

    with result_col:

        st.subheader(
            "🤖 INSPECTION RESULT"
        )


        # ====================================================
        # WAITING
        # ====================================================

        if not st.session_state.result_ready:

            st.markdown(
                """
                <div class="waiting-box">

                    <div style="font-size:40px;">
                        ⏳
                    </div>

                    <div class="waiting-title">
                        WAITING FOR ANALYSIS
                    </div>

                    <div class="waiting-text">
                        Analyze chesaka result ikkada display avvali.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # GOOD
        # ====================================================

        elif st.session_state.result_type == "good":

            st.markdown(
                """
                <div class="good-box">

                    <div class="good-title">
                        🟢 GOOD QUALITY
                    </div>

                    <p>
                        No defect detected.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # BAD IMAGE
        # ====================================================

        elif st.session_state.result_type == "bad":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="bad-box">

                    <div class="bad-title">
                        🔴 BAD QUALITY
                    </div>

                    <p>
                        Defect Detected
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.info(
                f"Defect: {data['defect']}"
            )


            st.write(
                f"**Confidence:** "
                f"{data['confidence'] * 100:.2f}%"
            )


            st.caption(
                "IMAGE WITH DEFECT BOX"
            )


            st.image(
                data["image"],
                width=360
            )


        # ====================================================
        # BAD VIDEO
        # ====================================================

        elif st.session_state.result_type == "bad_video":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="bad-box">

                    <div class="bad-title">
                        🔴 BAD QUALITY
                    </div>

                    <p>
                        Defect Detected
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.info(
                f"Defect: {data['defect']}"
            )


            st.write(
                f"**Confidence:** "
                f"{data['confidence'] * 100:.2f}%"
            )


            st.caption(
                "PROCESSED VIDEO WITH DEFECT BOXES"
            )


            # Smaller processed video
            st.video(
                data["video"]
            )


    st.markdown(
        '<div class="footer">'
        'YarnX – The Future of Yarn Inspection'
        '</div>',
        unsafe_allow_html=True
    )
