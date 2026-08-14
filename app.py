import os
import tempfile
import subprocess
import base64
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YarnX – The Future of Yarn Inspection",
    page_icon="🧶",
    layout="wide"
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "best (6).pt"

_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()

    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "image_output" not in st.session_state:
    st.session_state.image_output = None

if "image_defects" not in st.session_state:
    st.session_state.image_defects = []

if "video_output" not in st.session_state:
    st.session_state.video_output = None

if "video_defects" not in st.session_state:
    st.session_state.video_defects = {}


# ============================================================
# DARK THEME CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL DARK BACKGROUND
       ======================================================== */

    .stApp {
        min-height: 100vh;

        background:
            radial-gradient(
                circle at 5% 8%,
                rgba(124, 58, 237, 0.22) 0%,
                transparent 28%
            ),
            radial-gradient(
                circle at 95% 12%,
                rgba(37, 99, 235, 0.20) 0%,
                transparent 30%
            ),
            radial-gradient(
                circle at 10% 90%,
                rgba(14, 165, 233, 0.13) 0%,
                transparent 27%
            ),
            radial-gradient(
                circle at 92% 90%,
                rgba(236, 72, 153, 0.14) 0%,
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #070b17 0%,
                #0b1020 45%,
                #100b1d 100%
            );

        color: #f1f5f9;
    }


    /* ========================================================
       PAGE SIZE
       ======================================================== */

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 1450px;
    }


    /* ========================================================
       REMOVE STREAMLIT DEFAULT HEADER
       ======================================================== */

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        width: 100%;
        min-height: 72px;

        border: 2px solid rgba(139, 92, 246, 0.75);
        border-radius: 16px;

        display: flex;
        align-items: center;
        justify-content: center;

        padding: 0 20px;
        margin-top: 0;
        margin-bottom: 20px;

        transform: translateY(12px);

        overflow: hidden;
        text-align: center;

        font-size: 28px;
        font-weight: 800;

        color: #f5f3ff;

        background:
            linear-gradient(
                90deg,
                rgba(46, 16, 76, 0.92),
                rgba(17, 42, 80, 0.92),
                rgba(70, 20, 53, 0.92)
            );

        box-shadow:
            0 8px 28px rgba(0, 0, 0, 0.40),
            0 0 25px rgba(124, 58, 237, 0.12);
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        text-decoration: none !important;
    }

    h1 a,
    h2 a,
    h3 a,
    h4 a,
    h5 a,
    h6 a {
        display: none !important;
    }


    /* ========================================================
       NORMAL CARDS
       ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;

        border: 1px solid rgba(148, 163, 184, 0.18) !important;

        background:
            linear-gradient(
                145deg,
                rgba(22, 28, 48, 0.90),
                rgba(13, 18, 34, 0.90)
            ) !important;

        box-shadow:
            0 8px 28px rgba(0, 0, 0, 0.38) !important;

        backdrop-filter: blur(12px);
    }


    /* ========================================================
       AICW CARD
       ======================================================== */

    .aicw-card {
        min-height: 255px;
        border-radius: 18px;
        padding: 25px;

        background:
            linear-gradient(
                145deg,
                rgba(45, 25, 74, 0.96),
                rgba(18, 42, 75, 0.94)
            );

        border: 1px solid rgba(139, 92, 246, 0.30);

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.40);
    }


    /* ========================================================
       AICW TITLE
       ======================================================== */

    .aicw-title {
        font-size: 30px;
        font-weight: 850;
        color: #f5f3ff;
        line-height: 1.2;
        margin-bottom: 15px;
    }


    .aicw-subtitle {
        font-size: 27px;
        font-weight: 800;
        color: #c4b5fd;
        margin-bottom: 25px;
    }


    .aicw-capstone {
        font-size: 22px;
        font-weight: 800;
        color: #e2e8f0;
    }


    /* ========================================================
       WOMAN VISUAL
       ======================================================== */

    .woman-box {
        text-align: center;
        margin-top: -5px;
        margin-bottom: 10px;
    }

    .woman-circle {
        width: 100px;
        height: 100px;

        margin: auto;

        border-radius: 50%;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
            linear-gradient(
                145deg,
                #312e81,
                #1e3a8a
            );

        border:
            2px solid rgba(167, 139, 250, 0.45);

        box-shadow:
            0 10px 28px rgba(0, 0, 0, 0.45);

        font-size: 55px;
    }

    .yarn-small {
        font-size: 24px;
        margin-top: -25px;
        margin-left: 65px;
        position: relative;
        z-index: 2;
    }


    /* ========================================================
       PROJECT HEADING
       ======================================================== */

    .project-heading {
        color: #c4b5fd;
        font-size: 28px;
        font-weight: 850;
        margin-bottom: 15px;
    }


    /* ========================================================
       PROJECT TEXT
       ======================================================== */

    .project-text {
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.65;
        margin-bottom: 12px;
    }


    /* ========================================================
       EMAIL LINKS
       ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] a {
        color: #93c5fd !important;
        font-weight: 600;
        text-decoration: none !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        width: 100% !important;
        min-height: 48px !important;

        border: none !important;
        border-radius: 13px !important;

        font-size: 16px !important;
        font-weight: 800 !important;

        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;

        background:
            linear-gradient(
                135deg,
                #7c3aed 0%,
                #4f46e5 50%,
                #2563eb 100%
            ) !important;

        box-shadow:
            0 7px 20px rgba(79, 70, 229, 0.35) !important;

        transition: 0.2s ease;
    }

    .stButton > button:hover {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;

        background:
            linear-gradient(
                135deg,
                #8b5cf6 0%,
                #6366f1 50%,
                #3b82f6 100%
            ) !important;

        transform: translateY(-2px);

        box-shadow:
            0 10px 25px rgba(99, 102, 241, 0.45) !important;
    }

    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 800 !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    section[data-testid="stFileUploaderDropzone"] {
        background:
            linear-gradient(
                145deg,
                rgba(20, 27, 46, 0.95),
                rgba(12, 18, 32, 0.95)
            ) !important;

        border-radius: 13px !important;

        border:
            1px dashed rgba(129, 140, 248, 0.45) !important;
    }


    section[data-testid="stFileUploaderDropzone"] button {
        background:
            linear-gradient(
                135deg,
                #1e293b,
                #172554
            ) !important;

        border:
            1px solid rgba(129, 140, 248, 0.45) !important;

        border-radius: 10px !important;

        color: #e2e8f0 !important;
        -webkit-text-fill-color: #e2e8f0 !important;

        font-weight: 700 !important;

        min-height: 40px !important;

        box-shadow:
            0 3px 10px rgba(0, 0, 0, 0.30) !important;
    }

    section[data-testid="stFileUploaderDropzone"] button:hover {
        background:
            linear-gradient(
                135deg,
                #312e81,
                #1e40af
            ) !important;

        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    section[data-testid="stFileUploaderDropzone"] button p,
    section[data-testid="stFileUploaderDropzone"] button span,
    section[data-testid="stFileUploaderDropzone"] button div {
        color: #e2e8f0 !important;
        -webkit-text-fill-color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    section[data-testid="stFileUploaderDropzone"] small {
        color: #94a3b8 !important;
    }


    /* ========================================================
       RADIO
       ======================================================== */

    div[data-testid="stRadio"] label {
        font-weight: 600;
        color: #cbd5e1 !important;
    }

    div[data-testid="stRadio"] p {
        color: #cbd5e1 !important;
    }


    /* ========================================================
       FILE UPLOADER LABEL
       ======================================================== */

    .stFileUploader label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }


    /* ========================================================
       INPUT / RESULT LABELS
       ======================================================== */

    .stApp p,
    .stApp label {
        color: #cbd5e1;
    }

    .stApp strong,
    .stApp b {
        color: #f1f5f9;
    }


    /* ========================================================
       GOOD QUALITY
       ======================================================== */

    .good-quality {
        border: 2px solid #22c55e;

        border-radius: 12px;

        padding: 10px;

        text-align: center;

        font-size: 21px;

        font-weight: bold;

        color: #86efac;

        background:
            linear-gradient(
                135deg,
                rgba(20, 83, 45, 0.70),
                rgba(22, 101, 52, 0.45)
            );

        box-shadow:
            0 5px 18px rgba(34, 197, 94, 0.12);
    }


    /* ========================================================
       BAD QUALITY
       ======================================================== */

    .bad-quality {
        border: 2px solid #ef4444;

        border-radius: 12px;

        padding: 10px;

        text-align: center;

        font-size: 21px;

        font-weight: bold;

        color: #fca5a5;

        background:
            linear-gradient(
                135deg,
                rgba(127, 29, 29, 0.75),
                rgba(69, 10, 10, 0.60)
            );

        box-shadow:
            0 5px 18px rgba(239, 68, 68, 0.14);
    }


    /* ========================================================
       DEFECT CARD
       ======================================================== */

    .defect-card {
        border: 1px solid rgba(248, 113, 113, 0.55);

        border-radius: 10px;

        padding: 11px 13px;

        margin-top: 7px;

        color: #fecaca;

        background:
            linear-gradient(
                135deg,
                rgba(69, 10, 10, 0.82),
                rgba(51, 18, 18, 0.72)
            );

        box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.22);
    }


    /* ========================================================
       STREAMLIT INPUTS
       ======================================================== */

    input,
    textarea,
    select {
        background-color: #111827 !important;
        color: #e5e7eb !important;
    }


    /* ========================================================
       INFO MESSAGE
       ======================================================== */

    div[data-testid="stAlert"] {
        background: rgba(30, 41, 59, 0.85) !important;
        border: 1px solid rgba(96, 165, 250, 0.30) !important;
        color: #dbeafe !important;
    }


    /* ========================================================
       SPINNER
       ======================================================== */

    div[data-testid="stSpinner"] {
        color: #c4b5fd !important;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE DISPLAY
# ============================================================

def show_fixed_image(
    image,
    width=400,
    height=240,
    border_color="#6366f1",
    background="#111827"
):

    if isinstance(image, np.ndarray):
        image = Image.fromarray(
            image.astype(np.uint8)
        )

    image = image.copy()

    image.thumbnail(
        (width - 12, height - 12),
        Image.Resampling.LANCZOS
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    st.markdown(
        f"""
        <div style="
            width:{width}px;
            height:{height}px;
            margin:8px auto 12px auto;
            border:2px solid {border_color};
            border-radius:12px;
            background:{background};
            display:flex;
            align-items:center;
            justify-content:center;
            overflow:hidden;
            box-shadow:0 6px 18px rgba(0,0,0,0.35);
        ">
            <img
                src="data:image/png;base64,{encoded}"
                style="
                    max-width:{width - 12}px;
                    max-height:{height - 12}px;
                    width:auto;
                    height:auto;
                    object-fit:contain;
                "
            >
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# VIDEO DISPLAY
# ============================================================

def show_fixed_video(
    video_path,
    width=400,
    height=240,
    border_color="#6366f1",
    background="#111827"
):

    try:

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        encoded = base64.b64encode(
            video_bytes
        ).decode("utf-8")

        st.markdown(
            f"""
            <div style="
                width:{width}px;
                height:{height}px;
                margin:8px auto 12px auto;
                border:2px solid {border_color};
                border-radius:12px;
                background:{background};
                display:flex;
                align-items:center;
                justify-content:center;
                overflow:hidden;
                box-shadow:0 6px 18px rgba(0,0,0,0.35);
            ">
                <video
                    controls
                    style="
                        width:{width - 4}px;
                        max-width:{width - 4}px;
                        max-height:{height - 4}px;
                        object-fit:contain;
                    "
                >
                    <source
                        src="data:video/mp4;base64,{encoded}"
                        type="video/mp4"
                    >
                </video>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            f"❌ Unable to display video: {e}"
        )


# ============================================================
# YOLO BOXES
# ============================================================

def draw_yolo_boxes(frame, result):

    output = frame.copy()

    defects = []

    if result.boxes is None or len(result.boxes) == 0:
        return output, defects

    for box in result.boxes:

        coords = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        x1, y1, x2, y2 = coords

        confidence = float(
            box.conf[0].cpu().item()
        )

        class_id = int(
            box.cls[0].cpu().item()
        )

        defect_name = model.names[class_id]

        defects.append(
            {
                "name": defect_name,
                "confidence": confidence,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            6
        )

        label = (
            f"{defect_name} "
            f"{confidence * 100:.1f}%"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2

        text_size, _ = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        text_width, text_height = text_size

        label_y = max(
            y1,
            text_height + 15
        )

        cv2.rectangle(
            output,
            (
                x1,
                label_y - text_height - 12
            ),
            (
                x1 + text_width + 12,
                label_y + 4
            ),
            (255, 0, 0),
            -1
        )

        cv2.putText(
            output,
            label,
            (
                x1 + 6,
                label_y - 6
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return output, defects


# ============================================================
# VIDEO CONVERSION
# ============================================================

def convert_video_for_browser(input_path):

    try:

        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    except Exception:

        return input_path

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    command = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_path
    ]

    try:

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        return output_path

    except Exception:

        return input_path


# ============================================================
# FIRST PAGE
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # ========================================================
    # AICW
    # ========================================================

    with left:

        with st.container(border=True):

            st.markdown(
                '<div class="aicw-title">'
                'AI Career for Women'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="aicw-subtitle">'
                '(AICW)'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(
                [70, 30]
            )

            with col1:

                st.markdown(
                    '<div class="aicw-capstone">'
                    'Capstone Project'
                    '</div>',
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    """
                    <div class="woman-box">
                        <div class="woman-circle">
                            👩‍💻
                        </div>
                        <div class="yarn-small">
                            🧶
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.write("")

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "inspection"
            st.rerun()


    # ========================================================
    # PROJECT DESCRIPTION
    # ========================================================

    with right:

        with st.container(border=True):

            st.markdown(
                '<div class="project-heading">'
                'Project Description'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="project-text">
                    YarnX is an AI-powered yarn quality inspection
                    system designed to automatically detect and
                    identify yarn defects using Computer Vision
                    and Deep Learning.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="project-text">
                    The system accepts yarn images, camera input,
                    and videos for inspection. A trained YOLO model
                    analyzes the yarn and identifies defective
                    regions by drawing bounding boxes around
                    detected defects.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="project-text">
                    The system displays the detected defect,
                    confidence score, and final quality result as
                    GOOD or BAD. This helps reduce manual inspection
                    effort and supports faster and more accurate
                    yarn quality assessment.
                </div>
                """,
                unsafe_allow_html=True
            )


    st.write("")

    team, gmail, guide = st.columns(
        [1.35, 1.25, 0.9],
        gap="medium"
    )


    # ========================================================
    # TEAM
    # ========================================================

    with team:

        with st.container(border=True):

            st.markdown("### 👩‍💻 TEAM MEMBERS")

            st.write("1. Gutti.Pavani Devi Priya")
            st.write("2. Somasani.Sasi Priya")
            st.write("3. Galidevara.Rama Devi")
            st.write("4. Rambala.Harshitha Sai Lakshmi")


    # ========================================================
    # GMAIL
    # ========================================================

    with gmail:

        with st.container(border=True):

            st.markdown("### 📧 GMAIL")

            st.markdown(
                "gutthipavanidevipriya@gmail.com"
            )

            st.markdown(
                "Sasipriya8090@gmail.com"
            )

            st.markdown(
                "ramadevigalidevara0@gmail.com"
            )

            st.markdown(
                "harshitharambala3@gmail.com"
            )


    # ========================================================
    # GUIDE
    # ========================================================

    with guide:

        with st.container(border=True):

            st.markdown("### 🎓 GUIDE NAME")

            st.write("Md. Abdul Aziz")

            st.markdown("### DESIGNATION")

            st.write("Co Lead & Trainer AICW")


# ============================================================
# SECOND PAGE
# ============================================================

else:

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # BACK
    # ========================================================

    if st.button("⬅ Back"):

        st.session_state.page = "home"

        st.session_state.image_output = None
        st.session_state.image_defects = []

        st.session_state.video_output = None
        st.session_state.video_defects = {}

        st.rerun()

    st.write("")


    left, right = st.columns(
        [1, 1],
        gap="medium"
    )


    # ========================================================
    # INPUT
    # ========================================================

    with left:

        st.subheader("📥 INPUT")

        st.write("Select Input Type:")

        input_type = st.radio(
            "",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ],
            horizontal=True
        )


        # ====================================================
        # IMAGE
        # ====================================================

        if input_type == "🖼️ Image":

            uploaded_image = st.file_uploader(
                "Upload Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp"
                ],
                key="image_upload"
            )

            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")

                st.write("**INPUT PREVIEW**")

                show_fixed_image(
                    image,
                    width=400,
                    height=240,
                    border_color="#6366f1",
                    background="#111827"
                )

                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=0.15,
                            verbose=False
                        )[0]

                    output_image, defects = (
                        draw_yolo_boxes(
                            np.array(image),
                            result
                        )
                    )

                    st.session_state.image_output = (
                        output_image
                    )

                    st.session_state.image_defects = (
                        defects
                    )

                    st.session_state.video_output = None
                    st.session_state.video_defects = {}

                    st.rerun()


        # ====================================================
        # CAMERA
        # ====================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Capture Yarn"
            )

            if camera_image:

                image = Image.open(
                    camera_image
                ).convert("RGB")

                st.write("**CAMERA PREVIEW**")

                show_fixed_image(
                    image,
                    width=400,
                    height=240,
                    border_color="#6366f1",
                    background="#111827"
                )

                if st.button(
                    "🔍 Analyze Camera",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=0.15,
                            verbose=False
                        )[0]

                    output_image, defects = (
                        draw_yolo_boxes(
                            np.array(image),
                            result
                        )
                    )

                    st.session_state.image_output = (
                        output_image
                    )

                    st.session_state.image_defects = (
                        defects
                    )

                    st.session_state.video_output = None
                    st.session_state.video_defects = {}

                    st.rerun()


        # ====================================================
        # VIDEO
        # ====================================================

        else:

            uploaded_video = st.file_uploader(
                "Upload Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv"
                ],
                key="video_upload"
            )

            if uploaded_video:

                preview_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                preview_file.write(
                    uploaded_video.getvalue()
                )

                preview_file.close()

                st.write("**INPUT VIDEO**")

                show_fixed_video(
                    preview_file.name,
                    width=400,
                    height=240,
                    border_color="#6366f1",
                    background="#111827"
                )

                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        input_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_temp.write(
                            uploaded_video.getvalue()
                        )

                        input_temp.close()

                        input_path = input_temp.name

                        cap = cv2.VideoCapture(
                            input_path
                        )

                        if not cap.isOpened():

                            st.error(
                                "❌ Unable to open uploaded video."
                            )

                            st.stop()

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

                        fps = cap.get(
                            cv2.CAP_PROP_FPS
                        )

                        if fps <= 0:
                            fps = 25

                        output_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        output_temp.close()

                        raw_output = output_temp.name

                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )

                        writer = cv2.VideoWriter(
                            raw_output,
                            fourcc,
                            fps,
                            (width, height)
                        )

                        if not writer.isOpened():

                            cap.release()

                            st.error(
                                "❌ Unable to create output video."
                            )

                            st.stop()

                        all_defects = {}
                        last_boxes = []

                        while True:

                            ret, frame = cap.read()

                            if not ret:
                                break

                            result = model.predict(
                                source=frame,
                                conf=0.10,
                                verbose=False
                            )[0]

                            current_boxes = []

                            if (
                                result.boxes is not None
                                and len(result.boxes) > 0
                            ):

                                for box in result.boxes:

                                    coords = (
                                        box.xyxy[0]
                                        .cpu()
                                        .numpy()
                                        .astype(int)
                                    )

                                    x1, y1, x2, y2 = coords

                                    confidence = float(
                                        box.conf[0]
                                        .cpu()
                                        .item()
                                    )

                                    class_id = int(
                                        box.cls[0]
                                        .cpu()
                                        .item()
                                    )

                                    defect_name = model.names[
                                        class_id
                                    ]

                                    current_boxes.append(
                                        {
                                            "box": (
                                                x1,
                                                y1,
                                                x2,
                                                y2
                                            ),
                                            "name":
                                                defect_name,
                                            "confidence":
                                                confidence
                                        }
                                    )

                                    if defect_name not in all_defects:

                                        all_defects[
                                            defect_name
                                        ] = confidence

                                    elif confidence > all_defects[
                                        defect_name
                                    ]:

                                        all_defects[
                                            defect_name
                                        ] = confidence

                            if len(current_boxes) > 0:
                                last_boxes = current_boxes

                            if len(current_boxes) > 0:
                                boxes_to_draw = current_boxes
                            else:
                                boxes_to_draw = last_boxes

                            processed_frame = frame.copy()

                            for detection in boxes_to_draw:

                                x1, y1, x2, y2 = (
                                    detection["box"]
                                )

                                name = detection["name"]

                                confidence = (
                                    detection["confidence"]
                                )

                                cv2.rectangle(
                                    processed_frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (0, 0, 255),
                                    6
                                )

                                label = (
                                    f"{name} "
                                    f"{confidence * 100:.1f}%"
                                )

                                font = (
                                    cv2.FONT_HERSHEY_SIMPLEX
                                )

                                font_scale = 0.75
                                thickness = 2

                                text_size, _ = (
                                    cv2.getTextSize(
                                        label,
                                        font,
                                        font_scale,
                                        thickness
                                    )
                                )

                                text_width, text_height = (
                                    text_size
                                )

                                label_y = max(
                                    y1,
                                    text_height + 15
                                )

                                cv2.rectangle(
                                    processed_frame,
                                    (
                                        x1,
                                        label_y
                                        - text_height
                                        - 12
                                    ),
                                    (
                                        x1
                                        + text_width
                                        + 12,
                                        label_y + 4
                                    ),
                                    (0, 0, 255),
                                    -1
                                )

                                cv2.putText(
                                    processed_frame,
                                    label,
                                    (
                                        x1 + 6,
                                        label_y - 6
                                    ),
                                    font,
                                    font_scale,
                                    (255, 255, 255),
                                    thickness,
                                    cv2.LINE_AA
                                )

                            writer.write(
                                processed_frame
                            )

                        cap.release()
                        writer.release()

                        final_video = (
                            convert_video_for_browser(
                                raw_output
                            )
                        )

                        st.session_state.video_output = (
                            final_video
                        )

                        st.session_state.video_defects = (
                            all_defects
                        )

                        st.session_state.image_output = None
                        st.session_state.image_defects = []

                        st.rerun()


    # ========================================================
    # RESULT
    # ========================================================

    with right:

        st.subheader(
            "🤖 INSPECTION RESULT"
        )


        # ====================================================
        # IMAGE RESULT
        # ====================================================

        if st.session_state.image_output is not None:

            st.write(
                "**ANALYZED IMAGE**"
            )

            show_fixed_image(
                st.session_state.image_output,
                width=500,
                height=300,
                border_color="#8b5cf6",
                background="#111827"
            )

            defects = st.session_state.image_defects

            if len(defects) > 0:

                st.markdown(
                    """
                    <div class="bad-quality">
                        ❌ BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("### Detected Defects")

                for defect in defects:

                    st.markdown(
                        f"""
                        <div class="defect-card">
                            🔴 <b>Defect:</b>
                            {defect["name"]}
                            &nbsp;&nbsp;&nbsp;
                            📊 <b>Confidence:</b>
                            {defect["confidence"] * 100:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.markdown(
                    """
                    <div class="good-quality">
                        ✅ GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # VIDEO RESULT
        # ====================================================

        elif st.session_state.video_output is not None:

            st.write(
                "**ANALYZED VIDEO**"
            )

            show_fixed_video(
                st.session_state.video_output,
                width=500,
                height=300,
                border_color="#8b5cf6",
                background="#111827"
            )

            defects = st.session_state.video_defects

            if len(defects) > 0:

                st.markdown(
                    """
                    <div class="bad-quality">
                        ❌ BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("### Detected Defects")

                for name, confidence in defects.items():

                    st.markdown(
                        f"""
                        <div class="defect-card">
                            🔴 <b>Defect:</b>
                            {name}
                            &nbsp;&nbsp;&nbsp;
                            📊 <b>Confidence:</b>
                            {confidence * 100:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.markdown(
                    """
                    <div class="good-quality">
                        ✅ GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # DEFAULT
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
