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
# MODEL FINDER
# ============================================================

MODEL_NAMES = [
    "best (6).pt",
    "best.pt",
    "Copy of best.pt"
]


def find_model():

    # First check the same folder as app.py
    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    locations = [
        base_dir,
        os.path.join(base_dir, "model"),
        os.path.join(base_dir, "trained_model"),
        os.path.join(
            base_dir,
            "trained_model",
            "weights"
        ),
        os.path.join(base_dir, "weights")
    ]

    for location in locations:

        for name in MODEL_NAMES:

            path = os.path.join(
                location,
                name
            )

            if os.path.exists(path):

                return path


    # Recursive search inside project
    for root, dirs, files in os.walk(
        base_dir
    ):

        for name in MODEL_NAMES:

            if name in files:

                return os.path.join(
                    root,
                    name
                )


    return None


# ============================================================
# TORCH COMPATIBILITY
# ============================================================

_original_torch_load = torch.load


def patched_torch_load(
    *args,
    **kwargs
):

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
            "❌ Model file not found. "
            "Please keep best.pt / best (6).pt "
            "inside the project."
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

if "page" not in st.session_state:

    st.session_state.page = "home"


if "image_result" not in st.session_state:

    st.session_state.image_result = None


if "image_defects" not in st.session_state:

    st.session_state.image_defects = []


if "image_quality" not in st.session_state:

    st.session_state.image_quality = None


if "video_output" not in st.session_state:

    st.session_state.video_output = None


if "video_defects" not in st.session_state:

    st.session_state.video_defects = {}


if "video_quality" not in st.session_state:

    st.session_state.video_quality = None


# ============================================================
# HEURISTIC QUALITY SETTINGS
# ============================================================

# Existing model knows:
#
# 0 -> loop_fiber
# 1 -> protruding_fiber
#
# It does NOT know Good/Bad.
#
# Therefore Good/Bad below is estimated using
# confidence + detected region size.

BAD_CONFIDENCE = 0.85
BAD_AREA_RATIO = 0.03


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       PAGE
       ====================================================== */

    .block-container {

        padding-top: 2.0rem !important;
        padding-bottom: 1rem !important;

        max-width: 1500px !important;
    }


    /* ======================================================
       MAIN TITLE
       ====================================================== */

    .main-title {

        width: 100%;

        min-height: 66px;

        display: flex;

        align-items: center;

        justify-content: center;

        box-sizing: border-box;

        border: 2px solid #6a1b9a;

        border-radius: 16px;

        padding: 12px 20px;

        margin-bottom: 18px;

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
            0 4px 12px
            rgba(106, 27, 154, 0.12);
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {

        min-height: 46px;

        border-radius: 12px;

        border: none;

        font-size: 16px;

        font-weight: 800;

        color: white;

        background: linear-gradient(
            90deg,
            #6a1b9a,
            #3949ab
        );

        box-shadow:
            0 5px 14px
            rgba(106, 27, 154, 0.25);
    }


    .stButton > button:hover {

        color: white;

        transform: translateY(-1px);

        box-shadow:
            0 7px 18px
            rgba(106, 27, 154, 0.35);
    }


    /* ======================================================
       SECTION HEADING
       ====================================================== */

    .section-heading {

        font-size: 28px;

        font-weight: 800;

        color: #263238;

        margin-top: 8px;

        margin-bottom: 12px;
    }


    /* ======================================================
       QUALITY
       ====================================================== */

    .good-quality {

        border: 2px solid #388e3c;

        border-radius: 12px;

        padding: 9px;

        text-align: center;

        font-size: 20px;

        font-weight: 800;

        color: #1b5e20;

        background: #e8f5e9;

        margin: 10px 0;
    }


    .bad-quality {

        border: 2px solid #d32f2f;

        border-radius: 12px;

        padding: 9px;

        text-align: center;

        font-size: 20px;

        font-weight: 800;

        color: #b71c1c;

        background: #ffebee;

        margin: 10px 0;
    }


    /* ======================================================
       RESULT TABLE
       ====================================================== */

    .fiber-table {

        width: 100%;

        border-collapse: collapse;

        margin-top: 8px;

        border-radius: 10px;

        overflow: hidden;

        font-size: 15px;
    }


    .fiber-table th {

        background: #ede7f6;

        color: #4a148c;

        font-weight: 800;

        padding: 10px;

        text-align: left;

        border-bottom: 2px solid #d1c4e9;
    }


    .fiber-table td {

        padding: 9px 10px;

        border-bottom: 1px solid #e0e0e0;

        color: #263238;
    }


    .good-cell {

        color: #2e7d32;

        font-weight: 800;
    }


    .bad-cell {

        color: #c62828;

        font-weight: 800;
    }


    /* ======================================================
       TEAM CARDS
       ====================================================== */

    .info-title {

        font-size: 17px;

        font-weight: 800;

        color: #4a148c;

        margin-bottom: 8px;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CLASS NAME
# ============================================================

def get_class_name(
    class_id
):

    try:

        return str(
            model.names[class_id]
        ).lower().strip()

    except Exception:

        return "unknown"


# ============================================================
# FIBER TYPE
# ============================================================

def get_fiber_type(
    class_name
):

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
# QUALITY
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
# IMAGE PROCESSING
# ============================================================

def process_image(
    image
):

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


    image_height, image_width = (
        output.shape[:2]
    )


    image_area = (
        image_height
        * image_width
    )


    has_bad = False


    # ========================================================
    # DETECTIONS
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


            # ------------------------------------------------
            # COLOR
            # ------------------------------------------------

            if quality == "Bad Quality":

                has_bad = True

                color = (
                    0,
                    0,
                    255
                )

            else:

                color = (
                    0,
                    180,
                    0
                )


            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            detections.append(
                {
                    "fiber_type":
                        fiber_type,

                    "quality":
                        quality,

                    "confidence":
                        confidence,

                    "box":
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                }
            )


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                output,

                (x1, y1),

                (x2, y2),

                color,

                5
            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (
                f"{fiber_type} | {quality}"
            )


            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )


            font_scale = 0.65

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
                output,

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

                color,

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

                (
                    255,
                    255,
                    255
                ),

                thickness,

                cv2.LINE_AA
            )


    # ========================================================
    # OVERALL QUALITY
    # ========================================================

    if len(detections) == 0:

        overall_quality = (
            "NO FIBER DETECTED"
        )

    elif has_bad:

        overall_quality = "BAD QUALITY"

    else:

        overall_quality = "GOOD QUALITY"


    return (
        output,
        detections,
        overall_quality
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
# VIDEO PROCESSING
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


    raw_output = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name
    )


    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )


    writer = cv2.VideoWriter(
        raw_output,

        fourcc,

        fps,

        (width, height)
    )


    last_boxes = []

    detected_types = {}

    has_bad = False

    has_good = False


    # ========================================================
    # FRAME LOOP
    # ========================================================

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


        image_area = (
            width * height
        )


        # ----------------------------------------------------
        # DETECT
        # ----------------------------------------------------

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

                    color = (
                        0,
                        0,
                        255
                    )

                else:

                    has_good = True

                    color = (
                        0,
                        180,
                        0
                    )


                current_boxes.append(
                    {
                        "box":
                            (
                                x1,
                                y1,
                                x2,
                                y2
                            ),

                        "fiber_type":
                            fiber_type,

                        "quality":
                            quality,

                        "color":
                            color
                    }
                )


                key = (
                    fiber_type,
                    quality
                )


                detected_types[key] = True


        # ----------------------------------------------------
        # KEEP BOXES VISIBLE
        # ----------------------------------------------------

        if len(current_boxes) > 0:

            last_boxes = current_boxes


        boxes_to_draw = (
            current_boxes
            if len(current_boxes) > 0
            else last_boxes
        )


        processed = frame.copy()


        # ====================================================
        # DRAW BOXES
        # ====================================================

        for detection in boxes_to_draw:

            x1, y1, x2, y2 = (
                detection["box"]
            )


            fiber_type = (
                detection["fiber_type"]
            )


            quality = (
                detection["quality"]
            )


            color = (
                detection["color"]
            )


            cv2.rectangle(
                processed,

                (x1, y1),

                (x2, y2),

                color,

                5
            )


            label = (
                f"{fiber_type} | {quality}"
            )


            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )


            font_scale = 0.65

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
                processed,

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

                color,

                -1
            )


            cv2.putText(
                processed,

                label,

                (
                    x1 + 6,
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


        writer.write(
            processed
        )


    cap.release()

    writer.release()


    # ========================================================
    # FINAL QUALITY
    # ========================================================

    if len(detected_types) == 0:

        overall_quality = (
            "NO FIBER DETECTED"
        )

    elif has_bad:

        overall_quality = "BAD QUALITY"

    else:

        overall_quality = "GOOD QUALITY"


    final_video = (
        convert_video_for_browser(
            raw_output
        )
    )


    return (
        final_video,
        detected_types,
        overall_quality
    )


# ============================================================
# RESULT TABLE
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
            (
                fiber,
                quality
            )
        )


    html = """
    <table class="fiber-table">

        <tr>
            <th>Fiber Type</th>
            <th>Quality</th>
        </tr>
    """


    for fiber, quality in rows:

        if quality == "Good Quality":

            quality_html = (
                '<span class="good-cell">'
                '🟢 Good Quality'
                '</span>'
            )

        else:

            quality_html = (
                '<span class="bad-cell">'
                '🔴 Bad Quality'
                '</span>'
            )


        html += f"""
        <tr>

            <td>
                {fiber}
            </td>

            <td>
                {quality_html}
            </td>

        </tr>
        """


    html += """
    </table>
    """


    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# VIDEO RESULT TABLE
# ============================================================

def show_video_table(
    detected_types
):

    if len(detected_types) == 0:

        st.info(
            "No fiber detected by the model."
        )

        return


    html = """
    <table class="fiber-table">

        <tr>
            <th>Fiber Type</th>
            <th>Quality</th>
        </tr>
    """


    for fiber, quality in (
        detected_types.keys()
    ):

        if quality == "Good Quality":

            quality_html = (
                '<span class="good-cell">'
                '🟢 Good Quality'
                '</span>'
            )

        else:

            quality_html = (
                '<span class="bad-cell">'
                '🔴 Bad Quality'
                '</span>'
            )


        html += f"""
        <tr>

            <td>
                {fiber}
            </td>

            <td>
                {quality_html}
            </td>

        </tr>
        """


    html += """
    </table>
    """


    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# ============================================================
# HOME PAGE
# ============================================================
# ============================================================

if st.session_state.page == "home":

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                """
                <h2 style="
                    text-align:center;
                    color:#263238;
                    margin-top:18px;
                ">
                    AI Career for Women
                    <br>
                    (AICW)
                </h2>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
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


        st.write("")


        if st.button(
            "🔍  PREDICT",
            use_container_width=True
        ):

            st.session_state.page = (
                "inspection"
            )

            st.rerun()


    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

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
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div style="
                    line-height:1.65;
                    font-size:15px;
                    color:#263238;
                ">

                <p>
                YarnX is an AI-powered yarn quality inspection
                system designed to automatically detect and
                identify yarn fiber defects using Computer
                Vision and Deep Learning.
                </p>

                <p>
                The system accepts yarn images, camera input,
                and videos for inspection. A trained YOLO model
                analyzes the yarn and identifies fiber regions
                by drawing bounding boxes around detected fibers.
                </p>

                <p>
                The system displays the detected fiber type,
                confidence score, and estimated quality result
                as GOOD or BAD. This helps reduce manual
                inspection effort and supports faster and more
                accurate yarn quality assessment.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # TEAM
    # --------------------------------------------------------

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
                "**Md. Abdul Aziz**"
            )


            st.markdown(
                '<div class="info-title">'
                'DESIGNATION'
                '</div>',
                unsafe_allow_html=True
            )


            st.write(
                "**Co Lead & Trainer AICW**"
            )


# ============================================================
# ============================================================
# INSPECTION PAGE
# ============================================================
# ============================================================

else:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BACK
    # --------------------------------------------------------

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.session_state.image_result = None
        st.session_state.image_defects = []
        st.session_state.image_quality = None

        st.session_state.video_output = None
        st.session_state.video_defects = {}
        st.session_state.video_quality = None

        st.rerun()


    st.write("")


    # ========================================================
    # BALANCED COLUMNS
    # ========================================================

    input_col, result_col = st.columns(
        [48, 52],
        gap="large"
    )


    # ========================================================
    # INPUT COLUMN
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
                    width=280
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
                            detections,
                            quality
                        ) = process_image(
                            image
                        )


                    st.session_state.image_result = (
                        result_image
                    )


                    st.session_state.image_defects = (
                        detections
                    )


                    st.session_state.image_quality = (
                        quality
                    )


                    st.session_state.video_output = None

                    st.session_state.video_defects = {}

                    st.session_state.video_quality = None


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
                    width=280
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
                            detections,
                            quality
                        ) = process_image(
                            image
                        )


                    st.session_state.image_result = (
                        result_image
                    )


                    st.session_state.image_defects = (
                        detections
                    )


                    st.session_state.image_quality = (
                        quality
                    )


                    st.session_state.video_output = None


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
                    start_time=0,
                    width=360
                )


                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        input_file = (
                            tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".mp4"
                            )
                        )


                        input_file.write(
                            uploaded_video.getvalue()
                        )


                        input_file.close()


                        (
                            video_result,
                            detected_types,
                            quality
                        ) = process_video(
                            input_file.name
                        )


                    st.session_state.video_output = (
                        video_result
                    )


                    st.session_state.video_defects = (
                        detected_types
                    )


                    st.session_state.video_quality = (
                        quality
                    )


                    st.session_state.image_result = None

                    st.session_state.image_defects = []

                    st.session_state.image_quality = None


                    st.rerun()


    # ========================================================
    # RESULT COLUMN
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
                width=350
            )


            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            if (
                st.session_state.image_quality
                == "GOOD QUALITY"
            ):

                st.markdown(
                    """
                    <div class="good-quality">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif (
                st.session_state.image_quality
                == "BAD QUALITY"
            ):

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


            # ------------------------------------------------
            # DETECTED FIBERS
            # ------------------------------------------------

            st.markdown(
                "### Detected Fibers"
            )


            show_fiber_table(
                st.session_state.image_defects
            )


        # ====================================================
        # VIDEO RESULT
        # ====================================================

        elif (
            st.session_state.video_output
            is not None
        ):

            st.write(
                "**ANALYZED VIDEO**"
            )


            st.video(
                st.session_state.video_output,
                format="video/mp4",
                start_time=0,
                width=380
            )


            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            if (
                st.session_state.video_quality
                == "GOOD QUALITY"
            ):

                st.markdown(
                    """
                    <div class="good-quality">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif (
                st.session_state.video_quality
                == "BAD QUALITY"
            ):

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


            # ------------------------------------------------
            # VIDEO TABLE
            # ------------------------------------------------

            st.markdown(
                "### Detected Fibers"
            )


            show_video_table(
                st.session_state.video_defects
            )


        # ====================================================
        # INITIAL MESSAGE
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
