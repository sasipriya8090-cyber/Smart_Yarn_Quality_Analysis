import os
import tempfile
import subprocess

import cv2
import numpy as np
import pandas as pd
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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MODEL FINDER
# ============================================================

MODEL_NAMES = [
    "best.pt",
    "best (6).pt",
    "Copy of best.pt"
]


def find_model():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    possible_dirs = [
        base_dir,
        os.path.join(base_dir, "model"),
        os.path.join(base_dir, "weights"),
        os.path.join(base_dir, "trained_model"),
        os.path.join(
            base_dir,
            "trained_model",
            "weights"
        )
    ]

    for folder in possible_dirs:

        for filename in MODEL_NAMES:

            path = os.path.join(
                folder,
                filename
            )

            if os.path.exists(path):
                return path

    # Recursive search
    for root, dirs, files in os.walk(base_dir):

        for filename in MODEL_NAMES:

            if filename in files:

                return os.path.join(
                    root,
                    filename
                )

    return None


# ============================================================
# TORCH COMPATIBILITY
# ============================================================

_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):

    kwargs.setdefault(
        "weights_only",
        False
    )

    return _original_torch_load(
        *args,
        **kwargs
    )


torch.load = patched_torch_load


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = find_model()

    if model_path is None:

        st.error(
            "❌ best.pt model not found."
        )

        st.stop()

    model = YOLO(
        model_path
    )

    return model


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "home",
    "image_result": None,
    "image_detections": [],
    "image_quality": None,
    "video_result": None,
    "video_detections": [],
    "video_quality": None
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# QUALITY SETTINGS
# ============================================================

# Existing model:
#
# 0 = loop_fiber
# 1 = protruding_fiber
#
# Good/Bad is estimated here because
# the current model does not contain quality classes.

BAD_CONFIDENCE = 0.85
BAD_AREA_RATIO = 0.03


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 1500px !important;
    }

    /* TOP TITLE */

    .main-title {
        width: 100%;
        min-height: 68px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        border: 2px solid #6a1b9a;
        border-radius: 16px;
        padding: 10px 20px;
        margin-bottom: 20px;

        color: #4a148c;
        font-size: 30px;
        font-weight: 800;

        background: linear-gradient(
            90deg,
            #ede7f6,
            #e3f2fd,
            #fce4ec
        );

        box-shadow:
            0 4px 14px
            rgba(106,27,154,0.15);
    }

    /* BUTTON */

    .stButton > button {
        min-height: 46px;
        border-radius: 12px;
        border: none;

        color: white;
        font-size: 16px;
        font-weight: 800;

        background: linear-gradient(
            90deg,
            #6a1b9a,
            #3949ab
        );

        box-shadow:
            0 5px 14px
            rgba(106,27,154,0.25);
    }

    .stButton > button:hover {
        color: white;
    }

    /* HEADINGS */

    .section-heading {
        font-size: 28px;
        font-weight: 800;
        color: #263238;
        margin-top: 5px;
        margin-bottom: 15px;
    }

    /* QUALITY */

    .good-quality {
        border: 2px solid #388e3c;
        border-radius: 12px;
        padding: 10px;
        text-align: center;

        font-size: 20px;
        font-weight: 800;

        color: #1b5e20;
        background: #e8f5e9;

        margin-top: 12px;
        margin-bottom: 12px;
    }

    .bad-quality {
        border: 2px solid #d32f2f;
        border-radius: 12px;
        padding: 10px;
        text-align: center;

        font-size: 20px;
        font-weight: 800;

        color: #b71c1c;
        background: #ffebee;

        margin-top: 12px;
        margin-bottom: 12px;
    }

    /* INFO */

    .info-title {
        font-size: 17px;
        font-weight: 800;
        color: #4a148c;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CLASS NAME
# ============================================================

def get_class_name(class_id):

    try:

        return str(
            model.names[class_id]
        ).lower().strip()

    except Exception:

        return "unknown"


# ============================================================
# FIBER NAME
# ============================================================

def get_fiber_type(class_name):

    name = (
        class_name
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    if "loop" in name:
        return "Loop Fiber"

    if "protrud" in name:
        return "Protruding Fiber"

    return name.title()


# ============================================================
# QUALITY ESTIMATION
# ============================================================

def calculate_quality(
    confidence,
    area_ratio
):

    if (
        confidence >= BAD_CONFIDENCE
        or area_ratio >= BAD_AREA_RATIO
    ):

        return "Bad Quality"

    return "Good Quality"


# ============================================================
# DRAW BOX + LABEL
# ============================================================

def draw_detection(
    image,
    x1,
    y1,
    x2,
    y2,
    fiber_type,
    quality
):

    if quality == "Bad Quality":

        # RED
        color = (
            0,
            0,
            255
        )

    else:

        # GREEN
        color = (
            0,
            190,
            0
        )

    # Thick bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        6
    )

    label = (
        f"{fiber_type} | {quality}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.65

    thickness = 2

    text_size, _ = cv2.getTextSize(
        label,
        font,
        scale,
        thickness
    )

    text_width, text_height = text_size

    label_top = max(
        0,
        y1 - text_height - 18
    )

    # Label background
    cv2.rectangle(
        image,
        (
            x1,
            label_top
        ),
        (
            x1 + text_width + 18,
            y1
        ),
        color,
        -1
    )

    # White label
    cv2.putText(
        image,
        label,
        (
            x1 + 8,
            y1 - 7
        ),
        font,
        scale,
        (
            255,
            255,
            255
        ),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(image):

    image_array = np.array(
        image
    )

    result = model.predict(
        source=image_array,
        conf=0.10,
        verbose=False
    )[0]

    output = image_array.copy()

    detections = []

    height, width = output.shape[:2]

    image_area = (
        height * width
    )

    has_bad = False

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

            class_name = get_class_name(
                class_id
            )

            fiber_type = get_fiber_type(
                class_name
            )

            box_width = max(
                0,
                x2 - x1
            )

            box_height = max(
                0,
                y2 - y1
            )

            box_area = (
                box_width
                * box_height
            )

            area_ratio = (
                box_area / image_area
                if image_area > 0
                else 0
            )

            quality = calculate_quality(
                confidence,
                area_ratio
            )

            if quality == "Bad Quality":

                has_bad = True

            detections.append(
                {
                    "fiber_type": fiber_type,
                    "quality": quality,
                    "confidence": confidence
                }
            )

            draw_detection(
                output,
                x1,
                y1,
                x2,
                y2,
                fiber_type,
                quality
            )

    # Overall quality

    if len(detections) == 0:

        overall_quality = (
            "NO FIBER DETECTED"
        )

    elif has_bad:

        overall_quality = (
            "BAD QUALITY"
        )

    else:

        overall_quality = (
            "GOOD QUALITY"
        )

    return (
        output,
        detections,
        overall_quality
    )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(input_path):

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open video."
        )

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

    output_path = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    detected_pairs = set()

    has_bad = False

    has_good = False

    # Last detected boxes
    # are kept so boxes remain visible
    last_detections = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = model.predict(
            source=frame,
            conf=0.10,
            verbose=False
        )[0]

        current_detections = []

        image_area = (
            width * height
        )

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

                class_name = get_class_name(
                    class_id
                )

                fiber_type = get_fiber_type(
                    class_name
                )

                box_width = max(
                    0,
                    x2 - x1
                )

                box_height = max(
                    0,
                    y2 - y1
                )

                box_area = (
                    box_width
                    * box_height
                )

                area_ratio = (
                    box_area / image_area
                    if image_area > 0
                    else 0
                )

                quality = calculate_quality(
                    confidence,
                    area_ratio
                )

                if quality == "Bad Quality":

                    has_bad = True

                else:

                    has_good = True

                detected_pairs.add(
                    (
                        fiber_type,
                        quality
                    )
                )

                current_detections.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        fiber_type,
                        quality
                    )
                )

        # Keep previous detection boxes
        # visible when next frame temporarily
        # misses detection.

        if len(current_detections) > 0:

            last_detections = (
                current_detections
            )

        detections_to_draw = (
            current_detections
            if len(current_detections) > 0
            else last_detections
        )

        processed = frame.copy()

        for (
            x1,
            y1,
            x2,
            y2,
            fiber_type,
            quality
        ) in detections_to_draw:

            draw_detection(
                processed,
                x1,
                y1,
                x2,
                y2,
                fiber_type,
                quality
            )

        writer.write(
            processed
        )

    cap.release()

    writer.release()

    # Browser-compatible MP4

    browser_video = convert_video(
        output_path
    )

    if len(detected_pairs) == 0:

        overall_quality = (
            "NO FIBER DETECTED"
        )

    elif has_bad:

        overall_quality = (
            "BAD QUALITY"
        )

    else:

        overall_quality = (
            "GOOD QUALITY"
        )

    return (
        browser_video,
        detected_pairs,
        overall_quality
    )


# ============================================================
# VIDEO CONVERSION
# ============================================================

def convert_video(input_path):

    try:

        import imageio_ffmpeg

        ffmpeg = (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )

    except Exception:

        return input_path

    output_path = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name
    )

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
# FIBER TABLE
# ============================================================

def show_fiber_table(
    detections
):

    if len(detections) == 0:

        st.info(
            "No fiber detected by the model."
        )

        return

    rows = []

    seen = set()

    for detection in detections:

        fiber = detection[
            "fiber_type"
        ]

        quality = detection[
            "quality"
        ]

        key = (
            fiber,
            quality
        )

        if key in seen:
            continue

        seen.add(key)

        rows.append(
            {
                "Fiber Type": fiber,
                "Quality": quality
            }
        )

    df = pd.DataFrame(
        rows
    )

    # Native Streamlit table.
    # This prevents HTML source code
    # from appearing on the screen.

    st.table(df)


# ============================================================
# VIDEO TABLE
# ============================================================

def show_video_table(
    detected_pairs
):

    if len(detected_pairs) == 0:

        st.info(
            "No fiber detected by the model."
        )

        return

    rows = []

    for fiber, quality in sorted(
        detected_pairs
    ):

        rows.append(
            {
                "Fiber Type": fiber,
                "Quality": quality
            }
        )

    df = pd.DataFrame(
        rows
    )

    st.table(df)


# ============================================================
# ============================================================
# HOME PAGE
# ============================================================
# ============================================================

if st.session_state.page == "home":

    # TITLE

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )

    # PROJECT DESCRIPTION

    left, right = st.columns(
        [35, 65],
        gap="medium"
    )

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                """
                <h2 style="
                    text-align:center;
                    color:#263238;
                    margin-top:20px;
                ">
                    AI Career for Women
                    <br>
                    (AICW)
                </h2>

                <h3 style="
                    text-align:center;
                    color:#37474f;
                    margin-top:35px;
                ">
                    Capstone Project
                </h3>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        if st.button(
            "🔍  PREDICT",
            use_container_width=True
        ):

            st.session_state.page = (
                "inspection"
            )

            st.rerun()

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                """
                <h2 style="
                    color:#4a148c;
                ">
                    Project Description
                </h2>

                <p style="
                    line-height:1.65;
                    font-size:15px;
                    color:#263238;
                ">
                YarnX is an AI-powered yarn quality
                inspection system designed to automatically
                detect and identify yarn fiber defects using
                Computer Vision and Deep Learning.
                </p>

                <p style="
                    line-height:1.65;
                    font-size:15px;
                    color:#263238;
                ">
                The system accepts yarn images, camera input,
                and videos for inspection. A trained YOLO model
                analyzes the yarn and identifies fiber regions
                by drawing bounding boxes around detected fibers.
                </p>

                <p style="
                    line-height:1.65;
                    font-size:15px;
                    color:#263238;
                ">
                The system displays the detected fiber type,
                confidence score, and quality result as GOOD
                or BAD. This helps reduce manual inspection
                effort and supports faster yarn quality
                assessment.
                </p>
                """,
                unsafe_allow_html=True
            )

    # TEAM AREA

    st.write("")

    team_col, email_col, guide_col = st.columns(
        [1.2, 1.3, 0.9],
        gap="medium"
    )

    with team_col:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="info-title">'
                '👩‍💻 TEAM MEMBERS'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                "1. Gutti.Pavani Devi Priya"
            )

            st.write(
                "2. Somasani.Sasi Priya"
            )

            st.write(
                "3. Galidevara.Rama Devi"
            )

            st.write(
                "4. Rambala.Harshitha Sai Lakshmi"
            )

    with email_col:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="info-title">'
                '📧 GMAIL'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                "gutthipavanidevipriya@gmail.com"
            )

            st.write(
                "Sasipriya8090@gmail.com"
            )

            st.write(
                "ramadevigalidevara0@gmail.com"
            )

            st.write(
                "harshitharambala3@gmail.com"
            )

    with guide_col:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="info-title">'
                '👨‍🏫 GUIDE NAME'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                "Md. Abdul Aziz"
            )

            st.markdown(
                '<div class="info-title">'
                'DESIGNATION'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                "Co Lead & Trainer AICW"
            )


# ============================================================
# ============================================================
# INSPECTION PAGE
# ============================================================
# ============================================================

else:

    # TITLE

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )

    # BACK

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.session_state.image_result = None
        st.session_state.image_detections = []
        st.session_state.image_quality = None

        st.session_state.video_result = None
        st.session_state.video_detections = []
        st.session_state.video_quality = None

        st.rerun()

    st.write("")

    # BALANCED COLUMNS

    input_col, result_col = st.columns(
        [48, 52],
        gap="large"
    )

    # ========================================================
    # INPUT
    # ========================================================

    with input_col:

        st.markdown(
            """
            <div class="section-heading">
                📥 INPUT
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "Select Input Type:"
        )

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

                st.write(
                    "**INPUT IMAGE**"
                )

                st.image(
                    image,
                    width=300
                )

                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        (
                            result,
                            detections,
                            quality
                        ) = process_image(
                            image
                        )

                    st.session_state.image_result = (
                        result
                    )

                    st.session_state.image_detections = (
                        detections
                    )

                    st.session_state.image_quality = (
                        quality
                    )

                    st.session_state.video_result = None

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

                st.write(
                    "**INPUT IMAGE**"
                )

                st.image(
                    image,
                    width=300
                )

                if st.button(
                    "🔍 Analyze Camera",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        (
                            result,
                            detections,
                            quality
                        ) = process_image(
                            image
                        )

                    st.session_state.image_result = (
                        result
                    )

                    st.session_state.image_detections = (
                        detections
                    )

                    st.session_state.image_quality = (
                        quality
                    )

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

                st.write(
                    "**INPUT VIDEO**"
                )

                st.video(
                    uploaded_video,
                    format="video/mp4",
                    width=360
                )

                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video..."
                    ):

                        temp_input = (
                            tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".mp4"
                            )
                        )

                        temp_input.write(
                            uploaded_video.getvalue()
                        )

                        temp_input.close()

                        (
                            result_video,
                            detections,
                            quality
                        ) = process_video(
                            temp_input.name
                        )

                    st.session_state.video_result = (
                        result_video
                    )

                    st.session_state.video_detections = (
                        detections
                    )

                    st.session_state.video_quality = (
                        quality
                    )

                    st.session_state.image_result = None

                    st.rerun()

    # ========================================================
    # RESULT
    # ========================================================

    with result_col:

        st.markdown(
            """
            <div class="section-heading">
                🎯 RESULT
            </div>
            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # IMAGE RESULT
        # ====================================================

        if (
            st.session_state.image_result
            is not None
        ):

            st.write(
                "**ANALYZED IMAGE**"
            )

            st.image(
                st.session_state.image_result,
                width=380
            )

            quality = (
                st.session_state.image_quality
            )

            if quality == "GOOD QUALITY":

                st.markdown(
                    """
                    <div class="good-quality">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif quality == "BAD QUALITY":

                st.markdown(
                    """
                    <div class="bad-quality">
                        🔴 BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.info(
                    "No fiber detected."
                )

            st.markdown(
                "### Detected Fibers"
            )

            show_fiber_table(
                st.session_state.image_detections
            )

        # ====================================================
        # VIDEO RESULT
        # ====================================================

        elif (
            st.session_state.video_result
            is not None
        ):

            st.write(
                "**ANALYZED VIDEO**"
            )

            st.video(
                st.session_state.video_result,
                format="video/mp4",
                width=400
            )

            quality = (
                st.session_state.video_quality
            )

            if quality == "GOOD QUALITY":

                st.markdown(
                    """
                    <div class="good-quality">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif quality == "BAD QUALITY":

                st.markdown(
                    """
                    <div class="bad-quality">
                        🔴 BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.info(
                    "No fiber detected."
                )

            st.markdown(
                "### Detected Fibers"
            )

            show_video_table(
                st.session_state.video_detections
            )

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
