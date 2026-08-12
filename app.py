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
    page_title="YarnX – Yarn Inspection",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #17365d;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #667085;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Cards */
.info-card {
    background: white;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.07);
    border: 1px solid #e6eaf0;
    min-height: 210px;
}

.card-title {
    color: #17365d;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 15px;
}

.card-text {
    color: #475467;
    font-size: 15px;
    line-height: 1.7;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
    border: none;
    background: #ef5b57;
    color: white;
}

.stButton > button:hover {
    background: #d94c48;
    color: white;
}

/* Section */
.section-title {
    color: #17365d;
    font-size: 25px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Waiting */
.waiting-box {
    background: white;
    border-radius: 18px;
    padding: 55px 20px;
    text-align: center;
    border: 1px solid #e4e7ec;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
}

.waiting-icon {
    font-size: 45px;
    margin-bottom: 8px;
}

.waiting-title {
    color: #667085;
    font-size: 21px;
    font-weight: 750;
}

/* Result */
.result-card {
    background: white;
    border-radius: 18px;
    padding: 25px;
    border: 1px solid #e4e7ec;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
}

.result-title {
    color: #17365d;
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 15px;
}

.result-item {
    font-size: 16px;
    color: #344054;
    padding: 7px 0;
}

/* Preview */
.preview-box {
    background: white;
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #e4e7ec;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
}

/* Footer */
.footer {
    text-align: center;
    color: #667085;
    margin-top: 30px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOCATIONS
# ============================================================

MODEL_LOCATIONS = [
    Path("best (6).pt"),
    Path("best.pt"),
    Path("weights/best.pt"),
    Path("models/best.pt"),
    Path("yarn_model_100ep/weights/best.pt"),
]


# ============================================================
# FIND MODEL
# ============================================================

def find_model():

    for path in MODEL_LOCATIONS:
        if path.exists() and path.stat().st_size > 100000:
            return path

    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = find_model()

    if model_path is None:
        return None, None

    try:
        model = YOLO(str(model_path))
        return model, model_path

    except Exception as e:
        return None, str(e)


model, model_info = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "result_image" not in st.session_state:
    st.session_state.result_image = None

if "result_video" not in st.session_state:
    st.session_state.result_video = None

if "result_text" not in st.session_state:
    st.session_state.result_text = None


# ============================================================
# PAGE 1 – HOME
# ============================================================

if st.session_state.page == 1:

    st.markdown(
        '<div class="main-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-Powered Yarn Quality Inspection System</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Top information
    # --------------------------------------------------------

    col1, col2 = st.columns([1, 2])

    with col1:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        AI Career for Women (AICW)
        </div>

        <div class="card-text">

        <b>Capstone Project</b><br><br>

        An AI-based yarn inspection application
        designed to support automatic detection of
        yarn defects using computer vision.

        </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        Project Description
        </div>

        <div class="card-text">

        <b>YarnX</b> is an intelligent yarn quality
        inspection system that uses Artificial Intelligence,
        Deep Learning and Computer Vision to identify
        visible yarn defects automatically.

        The system can analyze <b>images, camera input
        and videos</b> and identify defects such as
        <b>loop fiber</b> and <b>protruding fiber</b>.

        The trained YOLO object-detection model examines
        the yarn sample and produces detection results
        with bounding boxes and confidence scores.

        This system helps reduce manual inspection effort,
        improves inspection consistency and supports
        faster yarn-quality assessment for textile
        manufacturing and weaving applications.

        </div>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --------------------------------------------------------
    # Predict button
    # --------------------------------------------------------

    if st.button("🔍 PREDICT"):

        st.session_state.page = 2
        st.rerun()

    st.write("")

    # --------------------------------------------------------
    # Team details
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">TEAM MEMBERS</div>

        <div class="card-text">

        1. Gutti.pavani devi Priya<br><br>

        2. Somasani.sasi priya<br><br>

        3. Galidevara.Rama Devi<br><br>

        4. Rambala.Harshitha sai Lakshmi

        </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">GMAIL</div>

        <div class="card-text">

        gutthipavanidevipriya@gmail.com<br><br>

        Sasipriya8090@gmail.com<br><br>

        ramadevigalidevara0gmail.com<br><br>

        harshitharambala3@gmail.com

        </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">GUIDE NAME</div>

        <div class="card-text">

        <b>Md. Abdul Aziz</b><br><br>

        <b>Designation</b><br><br>

        Co Lead & Trainer AICW

        </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2 – INSPECTION
# ============================================================

else:

    st.markdown(
        '<div class="main-title">🧶 YarnX – Yarn Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-Based Yarn Quality Analysis</div>',
        unsafe_allow_html=True
    )

    # Back button

    if st.button("⬅️ Back to Home"):

        st.session_state.page = 1
        st.session_state.result_image = None
        st.session_state.result_video = None
        st.session_state.result_text = None

        st.rerun()

    st.write("")

    left, right = st.columns([1, 1.35], gap="large")


    # ========================================================
    # INPUT
    # ========================================================

    with left:

        st.markdown(
            '<div class="section-title">📥 INPUT</div>',
            unsafe_allow_html=True
        )

        input_type = st.radio(
            "Select Input Type",
            ["🖼️ Image", "🎥 Video"],
            horizontal=True
        )

        uploaded_file = None

        if input_type == "🖼️ Image":

            uploaded_file = st.file_uploader(
                "Upload Yarn Image",
                type=["jpg", "jpeg", "png", "webp"]
            )

        else:

            uploaded_file = st.file_uploader(
                "Upload Yarn Video",
                type=["mp4", "avi", "mov", "mkv", "webm"]
            )


        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        if uploaded_file is not None and input_type == "🖼️ Image":

            image = Image.open(uploaded_file).convert("RGB")

            st.markdown(
                '<div class="preview-box"><b>Image Preview</b></div>',
                unsafe_allow_html=True
            )

            # SMALL IMAGE
            st.image(
                image,
                width=420
            )


        # ----------------------------------------------------
        # VIDEO PREVIEW
        # ----------------------------------------------------

        if uploaded_file is not None and input_type == "🎥 Video":

            st.markdown(
                '<div class="preview-box"><b>Video Preview</b></div>',
                unsafe_allow_html=True
            )

            video_bytes = uploaded_file.getvalue()

            st.video(video_bytes)


        st.write("")

        analyze = st.button(
            "🔍 Analyze Image / Video"
        )


    # ========================================================
    # RESULT
    # ========================================================

    with right:

        st.markdown(
            '<div class="section-title">🤖 INSPECTION RESULT</div>',
            unsafe_allow_html=True
        )

        # Waiting heading stays
        if (
            st.session_state.result_image is None
            and st.session_state.result_video is None
        ):

            st.markdown("""
            <div class="waiting-box">

                <div class="waiting-icon">⏳</div>

                <div class="waiting-title">
                    WAITING FOR ANALYSIS
                </div>

            </div>
            """, unsafe_allow_html=True)


        # ----------------------------------------------------
        # IMAGE RESULT
        # ----------------------------------------------------

        if st.session_state.result_image is not None:

            st.markdown("""
            <div class="result-card">

            <div class="result-title">
            🔎 Inspection Result
            </div>

            </div>
            """, unsafe_allow_html=True)

            st.image(
                st.session_state.result_image,
                width=420
            )

            if st.session_state.result_text:

                st.success(st.session_state.result_text)


        # ----------------------------------------------------
        # VIDEO RESULT
        # ----------------------------------------------------

        if st.session_state.result_video is not None:

            st.markdown("""
            <div class="result-card">

            <div class="result-title">
            🎥 Video Inspection Result
            </div>

            </div>
            """, unsafe_allow_html=True)

            st.video(
                st.session_state.result_video
            )

            if st.session_state.result_text:

                st.success(st.session_state.result_text)


# ============================================================
# ANALYSIS
# ============================================================

if st.session_state.page == 2 and "analyze" in locals() and analyze:

    if uploaded_file is None:

        st.warning("Please upload an image or video first.")

        st.stop()


    if model is None:

        st.error("Model could not be loaded.")

        st.info(
            "Please make sure the trained model file is present in the GitHub repository."
        )

        st.stop()


    # ========================================================
    # IMAGE ANALYSIS
    # ========================================================

    if input_type == "🖼️ Image":

        try:

            image = Image.open(uploaded_file).convert("RGB")

            results = model.predict(
                source=np.array(image),
                conf=0.20,
                iou=0.45,
                imgsz=640,
                verbose=False
            )

            result = results[0]

            plotted = result.plot()

            plotted = cv2.cvtColor(
                plotted,
                cv2.COLOR_BGR2RGB
            )

            st.session_state.result_image = plotted

            names = model.names

            detected = []

            if result.boxes is not None:

                for cls, conf in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist()
                ):

                    class_name = names[int(cls)]

                    detected.append(
                        f"{class_name} ({conf * 100:.1f}%)"
                    )


            if detected:

                st.session_state.result_text = (
                    "Detected defects: " + ", ".join(detected)
                )

            else:

                st.session_state.result_text = (
                    "No yarn defect detected in the uploaded image."
                )

            st.rerun()


        except Exception as e:

            st.error("Analysis failed.")

            st.exception(e)


    # ========================================================
    # VIDEO ANALYSIS
    # ========================================================

    else:

        input_path = None
        output_path = None

        try:

            # Save uploaded video

            suffix = Path(uploaded_file.name).suffix

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )

            input_file.write(
                uploaded_file.getbuffer()
            )

            input_file.close()

            input_path = input_file.name


            # Output file

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_file.close()

            output_path = output_file.name


            # ------------------------------------------------
            # Open video
            # ------------------------------------------------

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():

                st.error(
                    "Video could not be opened. Please upload a valid video."
                )

                st.stop()


            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0 or np.isnan(fps):

                fps = 20.0


            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )


            # Prevent extremely large output

            MAX_WIDTH = 720

            if width > MAX_WIDTH:

                scale = MAX_WIDTH / width

                width = MAX_WIDTH

                height = int(height * scale)


            # Make dimensions even

            width = width - (width % 2)
            height = height - (height % 2)


            # ------------------------------------------------
            # Temporary AVI writer
            # ------------------------------------------------

            temp_avi = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".avi"
            )

            temp_avi.close()


            fourcc = cv2.VideoWriter_fourcc(
                *"MJPG"
            )

            writer = cv2.VideoWriter(
                temp_avi.name,
                fourcc,
                fps,
                (width, height)
            )


            detected_classes = set()

            frame_count = 0


            # ------------------------------------------------
            # Process every frame
            # ------------------------------------------------

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1


                # Resize frame

                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )


                # YOLO detection

                results = model.predict(
                    source=frame,
                    conf=0.20,
                    iou=0.45,
                    imgsz=640,
                    verbose=False
                )


                result = results[0]


                # Collect classes

                if result.boxes is not None:

                    for cls in result.boxes.cls.tolist():

                        detected_classes.add(
                            model.names[int(cls)]
                        )


                # Draw detections

                annotated = result.plot()


                writer.write(
                    annotated
                )


            cap.release()
            writer.release()


            # ------------------------------------------------
            # Convert AVI -> MP4
            # ------------------------------------------------

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    temp_avi.name,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "28",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    output_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )


            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as f:

                video_data = f.read()


            st.session_state.result_video = video_data


            if detected_classes:

                st.session_state.result_text = (
                    "Detected defects: "
                    + ", ".join(sorted(detected_classes))
                )

            else:

                st.session_state.result_text = (
                    "No yarn defect detected in the uploaded video."
                )


            # Cleanup

            try:
                os.remove(input_path)
            except:
                pass

            try:
                os.remove(temp_avi.name)
            except:
                pass

            try:
                os.remove(output_path)
            except:
                pass


            st.rerun()


        except Exception as e:

            st.error("Video analysis failed.")

            st.exception(e)

            try:

                cap.release()

            except:

                pass
