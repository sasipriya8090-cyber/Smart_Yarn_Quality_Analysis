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
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "best.pt"),
    os.path.join(BASE_DIR, "best (6).pt"),
    os.path.join(BASE_DIR, "best (1).pt"),
    os.path.join(BASE_DIR, "model", "best.pt"),
    os.path.join(BASE_DIR, "trained_model", "weights", "best.pt"),
    os.path.join(BASE_DIR, "weights", "best.pt"),
]


def find_model():

    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path

    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".pt") and file in [
                "best.pt",
                "best (6).pt",
                "best (1).pt"
            ]:
                return os.path.join(root, file)

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
            "❌ best.pt model not found. "
            "Please keep your model file beside app.py."
        )

        st.stop()

    model = YOLO(model_path)

    return model


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
    st.session_state.video_defects = {}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 1500px !important;
}


/* =========================================================
   TITLE
   ========================================================= */

.main-title {
    width: 100%;
    box-sizing: border-box;

    min-height: 70px;

    display: flex;
    align-items: center;
    justify-content: center;

    padding: 12px 20px;

    border: 2px solid #6a1b9a;
    border-radius: 16px;

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
        0 5px 15px rgba(106,27,154,0.14);

    margin-bottom: 22px;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {

    min-height: 48px;

    border: none;
    border-radius: 13px;

    color: white;

    font-size: 16px;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #6a1b9a,
        #3949ab
    );

    box-shadow:
        0 5px 14px rgba(106,27,154,0.25);
}

.stButton > button:hover {

    color: white;

    transform: translateY(-1px);

    box-shadow:
        0 8px 20px rgba(106,27,154,0.35);
}


/* =========================================================
   SECTION HEADINGS
   ========================================================= */

.section-heading {

    font-size: 28px;
    font-weight: 800;

    color: #263238;

    margin-bottom: 15px;
}


/* =========================================================
   QUALITY
   ========================================================= */

.good-quality {

    border: 2px solid #388e3c;
    border-radius: 12px;

    padding: 11px;

    text-align: center;

    font-size: 20px;
    font-weight: 800;

    color: #1b5e20;

    background: #e8f5e9;

    margin-top: 12px;
    margin-bottom: 15px;
}


.bad-quality {

    border: 2px solid #d32f2f;
    border-radius: 12px;

    padding: 11px;

    text-align: center;

    font-size: 20px;
    font-weight: 800;

    color: #b71c1c;

    background: #ffebee;

    margin-top: 12px;
    margin-bottom: 15px;
}


/* =========================================================
   DEFECT CARD
   ========================================================= */

.defect-card {

    border: 1px solid #ef9a9a;

    border-radius: 10px;

    padding: 10px;

    margin-top: 8px;

    background: #fff7f7;

    font-size: 16px;
}


/* =========================================================
   TEAM CARDS
   ========================================================= */

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
# MODEL CLASS HELPERS
# ============================================================

def get_class_name(class_id):

    try:
        return str(
            model.names[class_id]
        ).lower().strip()

    except Exception:
        return "unknown"


def classify_detection(class_name):

    name = (
        class_name
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    # Existing model:
    #
    # 0 = loop_fiber
    # 1 = protruding_fiber
    #
    # Application logic:
    #
    # LOOP FIBER       -> GOOD
    # PROTRUDING FIBER -> BAD

    if "protrud" in name:
        return "bad"

    if "loop" in name:
        return "good"

    return "unknown"


# ============================================================
# DRAW BAD DETECTION
# ============================================================

def draw_bad_box(
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

    # RED BOX

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        red,
        5
    )

    # LABEL

    label = (
        f"PROTRUDING FIBER "
        f"{confidence * 100:.1f}%"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.70

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
        text_height + 18
    )

    # Red label background

    cv2.rectangle(
        image,
        (
            x1,
            label_y - text_height - 13
        ),
        (
            x1 + text_width + 14,
            label_y + 5
        ),
        red,
        -1
    )

    # White text

    cv2.putText(
        image,
        label,
        (
            x1 + 7,
            label_y - 6
        ),
        font,
        font_scale,
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

    image_array = np.array(image)

    result = model.predict(
        source=image_array,
        conf=0.25,
        verbose=False
    )[0]

    output = image_array.copy()

    defects = []

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

            category = classify_detection(
                class_name
            )

            # ==============================================
            # LOOP FIBER
            # ==============================================
            #
            # GOOD QUALITY
            #
            # Do NOT draw box.
            # Do NOT show confidence.
            # Do NOT show defect.

            if category == "good":
                continue

            # ==============================================
            # UNKNOWN
            # ==============================================

            if category != "bad":
                continue

            # ==============================================
            # PROTRUDING FIBER
            # ==============================================
            #
            # BAD QUALITY
            # Red box + confidence

            has_bad = True

            defects.append(
                {
                    "name": "PROTRUDING FIBER",
                    "confidence": confidence
                }
            )

            draw_bad_box(
                output,
                x1,
                y1,
                x2,
                y2,
                confidence
            )

    if has_bad:

        quality = "BAD"

    else:

        quality = "GOOD"

    return (
        output,
        defects,
        quality
    )


# ============================================================
# VIDEO CONVERSION
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

    # Last detected bad boxes.
    #
    # This keeps the red boxes visible
    # during temporary YOLO frame misses.

    last_bad_boxes = []

    all_defects = {}

    has_bad = False

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

                category = classify_detection(
                    class_name
                )

                # GOOD LOOP FIBER:
                # no box

                if category == "good":
                    continue

                # UNKNOWN:
                # ignore

                if category != "bad":
                    continue

                # BAD PROTRUDING FIBER

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

                # Keep highest confidence

                old_conf = all_defects.get(
                    "PROTRUDING FIBER",
                    0
                )

                if confidence > old_conf:

                    all_defects[
                        "PROTRUDING FIBER"
                    ] = confidence

        # Update remembered boxes

        if len(current_bad_boxes) > 0:

            last_bad_boxes = (
                current_bad_boxes
            )

        # If current frame has no detection,
        # use previous bad boxes.

        boxes_to_draw = (
            current_bad_boxes
            if len(current_bad_boxes) > 0
            else last_bad_boxes
        )

        processed = frame.copy()

        for (
            x1,
            y1,
            x2,
            y2,
            confidence
        ) in boxes_to_draw:

            draw_bad_box(
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

    final_video = (
        convert_video_for_browser(
            output_path
        )
    )

    if has_bad:

        quality = "BAD"

    else:

        quality = "GOOD"

    return (
        final_video,
        all_defects,
        quality
    )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    # ========================================================
    # TITLE
    # ========================================================

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # PROJECT SECTION
    # ========================================================

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
                input, and videos. A trained YOLO model
                analyzes the yarn and identifies fiber
                regions.
                </p>

                <p style="
                    font-size:15px;
                    line-height:1.7;
                ">
                YarnX provides a simple GOOD or BAD quality
                result. Protruding fibers are highlighted
                using red bounding boxes with confidence
                scores.
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

    # ========================================================
    # TITLE
    # ========================================================

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

        st.session_state.image_result = None
        st.session_state.image_quality = None
        st.session_state.image_defects = []

        st.session_state.video_result = None
        st.session_state.video_quality = None
        st.session_state.video_defects = {}

        st.rerun()

    st.write("")

    # ========================================================
    # BALANCED INPUT / RESULT
    # ========================================================

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

                # SMALL INPUT IMAGE

                st.image(
                    image,
                    width=260
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
                        {}
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
                    width=260
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

                # SMALL INPUT VIDEO

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

            # SMALL OUTPUT IMAGE

            st.image(
                st.session_state.image_result,
                width=330
            )

            quality = (
                st.session_state.image_quality
            )

            # =================================================
            # GOOD QUALITY
            # =================================================

            if quality == "GOOD":

                st.markdown(
                    """
                    <div class="good-quality">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =================================================
            # BAD QUALITY
            # =================================================

            else:

                st.markdown(
                    """
                    <div class="bad-quality">
                        🔴 BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    "### 🔴 Detected Fiber"
                )

                for defect in (
                    st.session_state.image_defects
                ):

                    confidence = (
                        defect["confidence"]
                        * 100
                    )

                    st.markdown(
                        f"""
                        <div class="defect-card">
                            🔴
                            <b>PROTRUDING FIBER</b>
                            <br>
                            📊 Confidence:
                            <b>{confidence:.2f}%</b>
                        </div>
                        """,
                        unsafe_allow_html=True
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

            # SMALL OUTPUT VIDEO

            st.video(
                st.session_state.video_result,
                width=340
            )

            quality = (
                st.session_state.video_quality
            )

            # =================================================
            # GOOD VIDEO
            # =================================================

            if quality == "GOOD":

                st.markdown(
                    """
                    <div class="good-quality">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =================================================
            # BAD VIDEO
            # =================================================

            else:

                st.markdown(
                    """
                    <div class="bad-quality">
                        🔴 BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    "### 🔴 Detected Fiber"
                )

                for (
                    name,
                    confidence
                ) in (
                    st.session_state.video_defects.items()
                ):

                    st.markdown(
                        f"""
                        <div class="defect-card">
                            🔴
                            <b>PROTRUDING FIBER</b>
                            <br>
                            📊 Confidence:
                            <b>{confidence * 100:.2f}%</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # ====================================================
        # NO RESULT
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
