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
# BASIC CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1200px;
    padding-top: 0.7rem !important;
    padding-bottom: 0.5rem !important;
}

/* TITLE */

.yarnx-title {
    text-align: center;
    color: #17365d;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 2px 0;
}

.yarnx-subtitle {
    text-align: center;
    color: #667085;
    font-size: 14px;
    margin: 0 0 12px 0;
}


/* CARDS */

.info-card {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 14px;
    padding: 17px 19px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.card-title {
    color: #17365d;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 9px;
}

.card-text {
    color: #475467;
    font-size: 13.5px;
    line-height: 1.45;
}


/* SECTION */

.section-title {
    color: #17365d;
    font-size: 22px;
    font-weight: 800;
    margin: 4px 0 8px 0;
}


/* WAITING */

.waiting-box {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 14px;
    padding: 38px 10px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.waiting-icon {
    font-size: 34px;
}

.waiting-title {
    color: #667085;
    font-size: 18px;
    font-weight: 800;
    margin-top: 5px;
}


/* BUTTON */

.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 700;
    font-size: 15px;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #667085;
    font-size: 12px;
    margin-top: 10px;
}


/* REDUCE GENERAL GAPS */

div[data-testid="stVerticalBlock"] {
    gap: 0.45rem;
}

div[data-testid="stHorizontalBlock"] {
    gap: 0.8rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

def show_title():

    st.markdown(
        '<div class="yarnx-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="yarnx-subtitle">AI-Powered Smart Yarn Quality Inspection</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FIND MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_model():

    # Exact file that you uploaded
    preferred = [
        "best (6).pt",
        "best.pt",
        "best (5).pt",
        "best (4).pt",
        "best (3).pt",
        "best (2).pt"
    ]

    for filename in preferred:

        path = BASE_DIR / filename

        if path.exists():

            try:
                if path.stat().st_size > 1_000_000:
                    return path
            except:
                pass


    # Search inside repository
    model_files = []

    for path in BASE_DIR.rglob("*.pt"):

        try:

            size = path.stat().st_size

            if size > 1_000_000:
                model_files.append((path, size))

        except:
            pass


    if model_files:

        model_files.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return model_files[0][0]


    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    model_path = find_model()

    if model_path is None:
        return None, None, "No trained .pt model found."

    try:

        model = YOLO(str(model_path))

        return model, model_path, None

    except Exception as e:

        return None, model_path, str(e)


model, model_path, model_error = get_model()


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
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    show_title()


    # --------------------------------------------------------
    # MAIN DESCRIPTION
    # --------------------------------------------------------

    left, right = st.columns(
        [0.9, 1.6],
        gap="small"
    )


    with left:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        AI Career for Women (AICW)
        </div>

        <div class="card-text">

        <b>Capstone Project</b><br><br>

        YarnX is an AI-powered yarn inspection
        application developed to automatically
        identify visible yarn quality defects.

        <br><br>

        The system combines Artificial Intelligence,
        Deep Learning and Computer Vision to support
        faster and more consistent yarn inspection.

        </div>

        </div>
        """, unsafe_allow_html=True)


    with right:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        Project Description
        </div>

        <div class="card-text">

        <b>YarnX – The Future of Yarn Inspection</b>
        is an AI-powered smart yarn quality inspection
        system designed to automatically analyze yarn
        samples and identify visible defects.

        <br><br>

        The application uses a trained
        <b>YOLO deep-learning object detection model</b>
        together with computer vision techniques to
        detect yarn defects from <b>images, camera
        captures and videos</b>.

        <br><br>

        The trained model identifies defects such as
        <b>loop fiber</b> and <b>protruding fiber</b>
        and displays detected regions using bounding
        boxes and confidence scores.

        <br><br>

        The system reduces manual inspection effort,
        improves consistency and supports faster yarn
        quality monitoring in textile and manufacturing
        environments.

        </div>

        </div>
        """, unsafe_allow_html=True)


    st.write("")


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    pc1, pc2, pc3 = st.columns(
        [1, 1, 1]
    )

    with pc2:

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            st.session_state.page = 2

            st.session_state.result_image = None
            st.session_state.result_video = None
            st.session_state.result_text = None

            st.rerun()


    st.write("")


    # --------------------------------------------------------
    # TEAM INFORMATION
    # --------------------------------------------------------

    t1, t2, t3 = st.columns(
        3,
        gap="small"
    )


    with t1:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        TEAM MEMBERS
        </div>

        <div class="card-text">

        1. Gutti.pavani devi Priya<br>
        2. Somasani.sasi priya<br>
        3. Galidevara.Rama Devi<br>
        4. Rambala.Harshitha sai Lakshmi

        </div>

        </div>
        """, unsafe_allow_html=True)


    with t2:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        GMAIL
        </div>

        <div class="card-text">

        gutthipavanidevipriya@gmail.com<br>
        Sasipriya8090@gmail.com<br>
        ramadevigalidevara0@gmail.com<br>
        harshitharambala3@gmail.com

        </div>

        </div>
        """, unsafe_allow_html=True)


    with t3:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        GUIDE NAME
        </div>

        <div class="card-text">

        <b>Md. Abdul Aziz</b><br><br>

        <b>Designation:</b><br>
        Co Lead & Trainer AICW

        </div>

        </div>
        """, unsafe_allow_html=True)


    st.markdown(
        '<div class="footer">YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2
# ============================================================

else:

    show_title()


    # --------------------------------------------------------
    # BACK BUTTON
    # --------------------------------------------------------

    if st.button("⬅️ Back to Home"):

        st.session_state.page = 1

        st.session_state.result_image = None
        st.session_state.result_video = None
        st.session_state.result_text = None

        st.rerun()


    st.write("")


    # --------------------------------------------------------
    # INPUT + RESULT
    # --------------------------------------------------------

    input_col, result_col = st.columns(
        [0.95, 1.25],
        gap="small"
    )


    # ========================================================
    # INPUT
    # ========================================================

    with input_col:

        st.markdown(
            '<div class="section-title">📥 INPUT</div>',
            unsafe_allow_html=True
        )


        input_type = st.radio(
            "Select Input Type",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ],
            horizontal=True
        )


        uploaded_file = None


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if input_type == "🖼️ Image":

            uploaded_file = st.file_uploader(
                "Upload Yarn Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp"
                ],
                key="image_upload"
            )


            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.caption("Image Preview")

                st.image(
                    image,
                    width=330
                )


        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        elif input_type == "📷 Camera":

            st.caption(
                "Capture a yarn image using your camera"
            )

            uploaded_file = st.camera_input(
                "📷 Take Yarn Photo"
            )


            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    width=330
                )


        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        else:

            uploaded_file = st.file_uploader(
                "Upload Yarn Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                    "webm"
                ],
                key="video_upload"
            )


            if uploaded_file:

                st.caption("Video Preview")

                # Put video inside narrow column
                v1, v2, v3 = st.columns(
                    [0.08, 0.84, 0.08]
                )

                with v2:

                    st.video(
                        uploaded_file.getvalue()
                    )


        st.write("")


        analyze = st.button(
            "🔍 Analyze Image / Video",
            use_container_width=True
        )


    # ========================================================
    # RESULT
    # ========================================================

    with result_col:

        st.markdown(
            '<div class="section-title">🤖 INSPECTION RESULT</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # WAITING
        # ----------------------------------------------------

        if (
            st.session_state.result_image is None
            and
            st.session_state.result_video is None
        ):

            # IMPORTANT:
            # No HTML here.
            # This prevents <div> text appearing in UI.

            st.info(
                "⏳  WAITING FOR ANALYSIS"
            )


        # ----------------------------------------------------
        # IMAGE RESULT
        # ----------------------------------------------------

        if st.session_state.result_image is not None:

            st.markdown(
                '<div class="result-card">'
                '<div class="result-title">'
                '🔎 Inspection Result'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


            st.image(
                st.session_state.result_image,
                width=360
            )


            if st.session_state.result_text:

                st.success(
                    st.session_state.result_text
                )


        # ----------------------------------------------------
        # VIDEO RESULT
        # ----------------------------------------------------

        if st.session_state.result_video is not None:

            st.markdown(
                '<div class="result-card">'
                '<div class="result-title">'
                '🎥 Video Inspection Result'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )


            rv1, rv2, rv3 = st.columns(
                [0.05, 0.90, 0.05]
            )

            with rv2:

                st.video(
                    st.session_state.result_video
                )


            if st.session_state.result_text:

                st.success(
                    st.session_state.result_text
                )


# ============================================================
# ANALYSIS
# ============================================================

if (
    st.session_state.page == 2
    and "analyze" in locals()
    and analyze
):


    # --------------------------------------------------------
    # NO INPUT
    # --------------------------------------------------------

    if uploaded_file is None:

        st.warning(
            "Please upload an image, capture an image, or upload a video first."
        )

        st.stop()


    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if model is None:

        st.error(
            "Model could not be loaded."
        )

        if model_path:

            st.info(
                f"Model path: {model_path}"
            )

        if model_error:

            st.code(
                model_error
            )

        st.stop()


    # ========================================================
    # IMAGE / CAMERA ANALYSIS
    # ========================================================

    if input_type in [
        "🖼️ Image",
        "📷 Camera"
    ]:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            results = model.predict(
                source=np.array(image),
                conf=0.20,
                iou=0.45,
                imgsz=640,
                verbose=False
            )


            result = results[0]


            # Draw bounding boxes
            plotted = result.plot()


            plotted = cv2.cvtColor(
                plotted,
                cv2.COLOR_BGR2RGB
            )


            st.session_state.result_image = plotted


            detected = []


            if result.boxes is not None:

                for cls, conf in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist()
                ):

                    class_name = model.names[
                        int(cls)
                    ]

                    detected.append(
                        f"{class_name} ({conf * 100:.1f}%)"
                    )


            if detected:

                st.session_state.result_text = (
                    "⚠️ Defect detected: "
                    +
                    ", ".join(detected)
                )

            else:

                st.session_state.result_text = (
                    "✅ No yarn defect detected – Good Quality"
                )


            st.rerun()


        except Exception as e:

            st.error(
                "Image analysis failed."
            )

            st.exception(e)


    # ========================================================
    # VIDEO ANALYSIS
    # ========================================================

    elif input_type == "🎥 Video":

        input_path = None
        avi_path = None
        output_path = None

        try:

            # ------------------------------------------------
            # SAVE INPUT
            # ------------------------------------------------

            suffix = Path(
                uploaded_file.name
            ).suffix


            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )


            temp_input.write(
                uploaded_file.getbuffer()
            )


            temp_input.close()


            input_path = temp_input.name


            # ------------------------------------------------
            # OPEN VIDEO
            # ------------------------------------------------

            cap = cv2.VideoCapture(
                input_path
            )


            if not cap.isOpened():

                st.error(
                    "Video could not be opened."
                )

                st.stop()


            fps = cap.get(
                cv2.CAP_PROP_FPS
            )


            if fps <= 0:

                fps = 20


            original_width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )


            original_height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )


            # ------------------------------------------------
            # REDUCE VIDEO SIZE
            # ------------------------------------------------

            MAX_WIDTH = 640


            if original_width > MAX_WIDTH:

                scale = (
                    MAX_WIDTH /
                    original_width
                )

                width = MAX_WIDTH

                height = int(
                    original_height * scale
                )

            else:

                width = original_width
                height = original_height


            width -= width % 2
            height -= height % 2


            # ------------------------------------------------
            # AVI
            # ------------------------------------------------

            temp_avi = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".avi"
            )

            temp_avi.close()

            avi_path = temp_avi.name


            fourcc = cv2.VideoWriter_fourcc(
                *"MJPG"
            )


            writer = cv2.VideoWriter(
                avi_path,
                fourcc,
                fps,
                (width, height)
            )


            detected_classes = set()


            # ------------------------------------------------
            # PROCESS FRAMES
            # ------------------------------------------------

            while True:

                ret, frame = cap.read()


                if not ret:
                    break


                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )


                results = model.predict(
                    source=frame,
                    conf=0.20,
                    iou=0.45,
                    imgsz=640,
                    verbose=False
                )


                result = results[0]


                if result.boxes is not None:

                    for cls in result.boxes.cls.tolist():

                        detected_classes.add(
                            model.names[int(cls)]
                        )


                annotated = result.plot()


                writer.write(
                    annotated
                )


            cap.release()
            writer.release()


            # ------------------------------------------------
            # MP4
            # ------------------------------------------------

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


            temp_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_output.close()

            output_path = temp_output.name


            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    avi_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "30",
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


            with open(
                output_path,
                "rb"
            ) as f:

                video_data = f.read()


            st.session_state.result_video = video_data


            if detected_classes:

                st.session_state.result_text = (
                    "⚠️ Defects detected: "
                    +
                    ", ".join(
                        sorted(detected_classes)
                    )
                )

            else:

                st.session_state.result_text = (
                    "✅ No yarn defect detected – Good Quality"
                )


            # ------------------------------------------------
            # CLEANUP
            # ------------------------------------------------

            for p in [
                input_path,
                avi_path,
                output_path
            ]:

                try:

                    if p and os.path.exists(p):
                        os.remove(p)

                except:
                    pass


            st.rerun()


        except Exception as e:

            st.error(
                "Video analysis failed."
            )

            st.exception(e)

            try:
                cap.release()
            except:
                pass
