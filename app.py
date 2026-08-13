import os
import tempfile
import subprocess

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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "best.pt"),
    os.path.join(BASE_DIR, "best (6).pt"),
    os.path.join(BASE_DIR, "model", "best.pt"),
    os.path.join(
        BASE_DIR,
        "trained_model",
        "weights",
        "best.pt"
    ),
    os.path.join(
        BASE_DIR,
        "weights",
        "best.pt"
    )
]


def find_model():

    for path in MODEL_CANDIDATES:

        if os.path.exists(path):
            return path

    for root, dirs, files in os.walk(BASE_DIR):

        for file in files:

            if file.endswith(".pt"):

                return os.path.join(
                    root,
                    file
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
            "❌ best.pt not found. "
            "Keep best.pt inside the project folder."
        )

        st.stop()

    return YOLO(model_path)


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "image_result" not in st.session_state:
    st.session_state.image_result = None

if "image_quality" not in st.session_state:
    st.session_state.image_quality = None

if "image_defects" not in st.session_state:
    st.session_state.image_defects = []

if "video_result" not in st.session_state:
    st.session_state.video_result = None

if "video_quality" not in st.session_state:
    st.session_state.video_quality = None

if "video_defects" not in st.session_state:
    st.session_state.video_defects = []


# ============================================================
# QUALITY RULE
# ============================================================
#
# Current model has:
#
# 0 -> loop_fiber
# 1 -> protruding_fiber
#
# The model itself does NOT contain Good/Bad classes.
#
# For this app:
#
# - no protruding fiber detected -> GOOD
# - protruding fiber detected    -> BAD
#
# Loop fiber is detected internally but is not shown
# in the final image/result when the quality is GOOD.
#
# ============================================================


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1rem !important;
        max-width: 1450px !important;
    }

    /* ================= TITLE ================= */

    .main-title {
        width: 100%;
        min-height: 68px;
        box-sizing: border-box;

        display: flex;
        align-items: center;
        justify-content: center;

        padding: 10px 20px;

        border: 2px solid #6a1b9a;
        border-radius: 16px;

        color: #4a148c;

        font-size: 29px;
        font-weight: 800;

        background: linear-gradient(
            90deg,
            #ede7f6,
            #e3f2fd,
            #fce4ec
        );

        box-shadow:
            0 5px 15px
            rgba(106,27,154,0.14);

        margin-bottom: 18px;
    }

    /* ================= SECTION ================= */

    .section-heading {
        font-size: 25px;
        font-weight: 800;
        color: #263238;
        margin-bottom: 10px;
    }

    /* ================= BUTTON ================= */

    .stButton > button {
        min-height: 46px;

        border: none;
        border-radius: 12px;

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

    /* ================= QUALITY ================= */

    .good-quality {
        border: 2px solid #388e3c;
        border-radius: 11px;

        padding: 10px;

        text-align: center;

        font-size: 19px;
        font-weight: 800;

        color: #1b5e20;

        background: #e8f5e9;

        margin: 10px 0;
    }

    .bad-quality {
        border: 2px solid #d32f2f;
        border-radius: 11px;

        padding: 10px;

        text-align: center;

        font-size: 19px;
        font-weight: 800;

        color: #b71c1c;

        background: #ffebee;

        margin: 10px 0;
    }

    /* ================= DEFECT CARD ================= */

    .defect-card {
        border: 1px solid #ef9a9a;
        border-radius: 10px;

        padding: 9px 12px;

        margin-top: 8px;

        background: #fff8f8;

        font-size: 15px;
    }

    /* ================= HOME INFO ================= */

    .info-title {
        font-size: 16px;
        font-weight: 800;
        color: #4a148c;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CLASS NAME
# ============================================================

def get_class_name(class_id):

    try:
        return str(
            model.names[class_id]
        ).lower().strip()

    except Exception:
        return "unknown"


# ============================================================
# FIBER TYPE
# ============================================================

def get_fiber_type(class_name):

    name = (
        class_name
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    if "loop" in name:
        return "LOOP FIBER"

    if "protrud" in name:
        return "PROTRUDING FIBER"

    return "UNKNOWN"


# ============================================================
# DRAW RED BOX
# ============================================================

def draw_red_box(
    image,
    x1,
    y1,
    x2,
    y2,
    confidence
):

    red = (
        0,
        0,
        255
    )

    # BOX

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        red,
        4
    )

    # LABEL

    label = (
        f"PROTRUDING FIBER "
        f"{confidence * 100:.1f}%"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65

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
        image,
        (
            x1,
            label_y - text_height - 10
        ),
        (
            x1 + text_width + 10,
            label_y + 4
        ),
        red,
        -1
    )

    cv2.putText(
        image,
        label,
        (
            x1 + 5,
            label_y - 5
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image):

    image_array = np.array(image)

    result = model.predict(
        source=image_array,
        conf=0.25,
        verbose=False
    )[0]

    output = image_array.copy()

    bad_detections = []

    loop_detected = False

    protruding_detected = False

    # ========================================================
    # READ DETECTIONS
    # ========================================================

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

            # =================================================
            # LOOP FIBER
            # =================================================

            if fiber_type == "LOOP FIBER":

                loop_detected = True

                # Do NOT show box in final image
                # for GOOD quality.

                continue

            # =================================================
            # PROTRUDING FIBER
            # =================================================

            if fiber_type == "PROTRUDING FIBER":

                protruding_detected = True

                bad_detections.append(
                    {
                        "fiber":
                            "PROTRUDING FIBER",

                        "confidence":
                            confidence
                    }
                )

                # Red box

                draw_red_box(
                    output,
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence
                )

    # ========================================================
    # QUALITY
    # ========================================================

    if protruding_detected:

        quality = "BAD"

    else:

        quality = "GOOD"

    return (
        output,
        bad_detections,
        quality
    )


# ============================================================
# CONVERT VIDEO FOR BROWSER
# ============================================================

def convert_video_for_browser(
    input_path
):

    try:

        import imageio_ffmpeg

        ffmpeg = (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )

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
# PROCESS VIDEO
# ============================================================

def process_video(
    input_path
):

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

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

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

    all_bad_detections = []

    # Keep the last detected protruding fiber
    # boxes so they do not disappear immediately.

    last_bad_boxes = []

    has_bad = False

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )[0]

        current_bad_boxes = []

        # ====================================================
        # DETECTIONS
        # ====================================================

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

                # =================================================
                # LOOP FIBER
                # =================================================

                if fiber_type == "LOOP FIBER":

                    continue

                # =================================================
                # PROTRUDING FIBER
                # =================================================

                if fiber_type == "PROTRUDING FIBER":

                    has_bad = True

                    current_bad_boxes.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence
                        )
                    )

                    all_bad_detections.append(
                        {
                            "fiber":
                                "PROTRUDING FIBER",

                            "confidence":
                                confidence
                        }
                    )

        # ====================================================
        # REMEMBER LAST BAD BOXES
        # ====================================================

        if len(current_bad_boxes) > 0:

            last_bad_boxes = (
                current_bad_boxes
            )

        boxes_to_draw = (

            current_bad_boxes

            if len(current_bad_boxes) > 0

            else last_bad_boxes
        )

        processed = frame.copy()

        # ====================================================
        # DRAW RED BOXES
        # ====================================================

        for (
            x1,
            y1,
            x2,
            y2,
            confidence
        ) in boxes_to_draw:

            draw_red_box(
                processed,

                x1,
                y1,
                x2,
                y2,

                confidence
            )

        writer.write(
            processed
        )

    cap.release()

    writer.release()

    # ========================================================
    # QUALITY
    # ========================================================

    if has_bad:

        quality = "BAD"

    else:

        quality = "GOOD"

    final_video = (
        convert_video_for_browser(
            output_path
        )
    )

    return (
        final_video,
        all_bad_detections,
        quality
    )


# ============================================================
# HOME PAGE
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
        gap="medium"
    )

    # ========================================================
    # LEFT CARD
    # ========================================================

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                """
                <div style="
                    text-align:center;
                    padding:20px;
                ">

                <h2 style="
                    color:#263238;
                    margin-bottom:25px;
                ">

                    AI Career for Women
                    <br>
                    (AICW)

                </h2>

                <h3 style="
                    color:#37474f;
                ">

                    Capstone Project

                </h3>

                </div>
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

    # ========================================================
    # PROJECT DESCRIPTION
    # ========================================================

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
                    font-size:15px;
                    line-height:1.7;
                ">

                YarnX is an AI-powered yarn quality
                inspection system designed to automatically
                detect yarn fiber defects using Computer
                Vision and Deep Learning.

                </p>

                <p style="
                    font-size:15px;
                    line-height:1.7;
                ">

                The system accepts yarn images, camera
                input and videos. A trained YOLO model
                analyzes the yarn and identifies fiber
                regions.

                </p>

                <p style="
                    font-size:15px;
                    line-height:1.7;
                ">

                YarnX provides a simple GOOD or BAD result
                and highlights detected protruding fiber
                defects with red bounding boxes and
                confidence scores.

                </p>
                """,
                unsafe_allow_html=True
            )

    # ========================================================
    # TEAM INFORMATION
    # ========================================================

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
                """
                <div class="info-title">
                    👩‍💻 TEAM MEMBERS
                </div>
                """,
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
                """
                <div class="info-title">
                    📧 GMAIL
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                "gutthipavanidevipriya@gmail.com"
            )

            st.write(
                "Sasipriya8090@gmail.com"
            )

            st.write(
                "ramadevi.galidevara0@gmail.com"
            )

            st.write(
                "harshitharambala3@gmail.com"
            )

    with guide_col:

        with st.container(
            border=True
        ):

            st.markdown(
                """
                <div class="info-title">
                    👨‍🏫 GUIDE NAME
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                "Md. Abdul Aziz"
            )

            st.markdown(
                """
                <div class="info-title">
                    DESIGNATION
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                "Co Lead & Trainer AICW"
            )


# ============================================================
# INSPECTION PAGE
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

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.session_state.image_result = None
        st.session_state.image_quality = None
        st.session_state.image_defects = []

        st.session_state.video_result = None
        st.session_state.video_quality = None
        st.session_state.video_defects = []

        st.rerun()

    st.write("")

    # ========================================================
    # EQUAL INPUT / RESULT COLUMNS
    # ========================================================

    input_col, result_col = st.columns(
        [1, 1],
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

                # SMALL IMAGE

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
                            result_image,
                            defects,
                            quality
                        ) = process_image(
                            image
                        )

                    st.session_state.image_result = (
                        result_image
                    )

                    st.session_state.image_defects = (
                        defects
                    )

                    st.session_state.image_quality = (
                        quality
                    )

                    st.session_state.video_result = (
                        None
                    )

                    st.session_state.video_defects = (
                        []
                    )

                    st.session_state.video_quality = (
                        None
                    )

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
                            result_image,
                            defects,
                            quality
                        ) = process_image(
                            image
                        )

                    st.session_state.image_result = (
                        result_image
                    )

                    st.session_state.image_defects = (
                        defects
                    )

                    st.session_state.image_quality = (
                        quality
                    )

                    st.session_state.video_result = (
                        None
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
                    width=320
                )

                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        temp_video = (
                            tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".mp4"
                            )
                        )

                        temp_video.write(
                            uploaded_video.getvalue()
                        )

                        temp_video.close()

                        (
                            output_video,
                            defects,
                            quality
                        ) = process_video(
                            temp_video.name
                        )

                    st.session_state.video_result = (
                        output_video
                    )

                    st.session_state.video_defects = (
                        defects
                    )

                    st.session_state.video_quality = (
                        quality
                    )

                    st.session_state.image_result = (
                        None
                    )

                    st.session_state.image_defects = (
                        []
                    )

                    st.session_state.image_quality = (
                        None
                    )

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
                "**OUTPUT IMAGE**"
            )

            # SAME SMALL SIZE AS INPUT

            st.image(
                st.session_state.image_result,
                width=300
            )

            quality = (
                st.session_state.image_quality
            )

            if quality == "GOOD":

                st.markdown(
                    """
                    <div class="good-quality">

                        🟢 GOOD QUALITY

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Good quality:
                # NO fibers / defects displayed

            else:

                st.markdown(
                    """
                    <div class="bad-quality">

                        🔴 BAD QUALITY

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # =================================================
                # DEFECT DETAILS ONLY FOR BAD QUALITY
                # =================================================

                if (
                    len(
                        st.session_state.image_defects
                    ) > 0
                ):

                    st.markdown(
                        "### 🔴 DETECTED DEFECT"
                    )

                    shown = set()

                    for defect in (
                        st.session_state.image_defects
                    ):

                        key = (
                            defect["fiber"],
                            round(
                                defect["confidence"],
                                2
                            )
                        )

                        if key in shown:
                            continue

                        shown.add(key)

                        confidence = (
                            defect["confidence"]
                            * 100
                        )

                        st.markdown(
                            f"""
                            <div class="defect-card">

                            🔴
                            <b>
                            {defect["fiber"]}
                            </b>

                            <br>

                            Confidence:
                            <b>
                            {confidence:.2f}%
                            </b>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:

                    st.write(
                        "No defect detected."
                    )

        # ====================================================
        # VIDEO RESULT
        # ====================================================

        elif (
            st.session_state.video_result
            is not None
        ):

            st.write(
                "**OUTPUT VIDEO**"
            )

            st.video(
                st.session_state.video_result,
                width=320
            )

            quality = (
                st.session_state.video_quality
            )

            if quality == "GOOD":

                st.markdown(
                    """
                    <div class="good-quality">

                        🟢 GOOD QUALITY

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="bad-quality">

                        🔴 BAD QUALITY

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if (
                    len(
                        st.session_state.video_defects
                    ) > 0
                ):

                    st.markdown(
                        "### 🔴 DETECTED DEFECT"
                    )

                    highest_confidence = 0

                    for defect in (
                        st.session_state.video_defects
                    ):

                        highest_confidence = max(
                            highest_confidence,
                            defect["confidence"]
                        )

                    st.markdown(
                        f"""
                        <div class="defect-card">

                        🔴
                        <b>
                        PROTRUDING FIBER
                        </b>

                        <br>

                        Confidence:
                        <b>
                        {highest_confidence * 100:.2f}%
                        </b>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.write(
                        "No defect detected."
                    )

        # ====================================================
        # BEFORE ANALYSIS
        # ====================================================

        else:

            st.info(
                "Upload an image or video "
                "and click Analyze."
            )
