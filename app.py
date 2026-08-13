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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "best.pt"),
    os.path.join(BASE_DIR, "best (6).pt"),
    os.path.join(BASE_DIR, "best (1).pt"),
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

            if file in [
                "best.pt",
                "best (6).pt",
                "best (1).pt"
            ]:

                return os.path.join(
                    root,
                    file
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
            "❌ best.pt model not found. "
            "Keep best.pt inside the project folder."
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

if "image_quality" not in st.session_state:
    st.session_state.image_quality = None

if "image_detections" not in st.session_state:
    st.session_state.image_detections = []

if "video_result" not in st.session_state:
    st.session_state.video_result = None

if "video_quality" not in st.session_state:
    st.session_state.video_quality = None

if "video_detections" not in st.session_state:
    st.session_state.video_detections = []


# ============================================================
# QUALITY THRESHOLD
# ============================================================

# Confidence below 70%  -> GOOD
# Confidence 70% or more -> BAD

QUALITY_THRESHOLD = 0.70


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


    /* ======================================================
       TITLE
       ====================================================== */

    .main-title {

        width: 100%;

        min-height: 70px;

        box-sizing: border-box;

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
            0 5px 15px
            rgba(106,27,154,0.14);

        margin-bottom: 22px;
    }


    /* ======================================================
       BUTTON
       ====================================================== */

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
            0 5px 14px
            rgba(106,27,154,0.25);
    }


    .stButton > button:hover {

        color: white;
    }


    /* ======================================================
       SECTION HEADING
       ====================================================== */

    .section-heading {

        font-size: 28px;

        font-weight: 800;

        color: #263238;

        margin-bottom: 15px;
    }


    /* ======================================================
       QUALITY
       ====================================================== */

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


    /* ======================================================
       DETECTION CARD
       ====================================================== */

    .detection-card {

        border-radius: 10px;

        padding: 10px;

        margin-top: 8px;

        font-size: 16px;

        background: #fafafa;

        border: 1px solid #dddddd;
    }


    /* ======================================================
       INFO
       ====================================================== */

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

        return "LOOP FIBER"

    if "protrud" in name:

        return "PROTRUDING FIBER"

    return name.upper()


# ============================================================
# QUALITY DECISION
# ============================================================

def get_quality(
    confidence
):

    if confidence >= QUALITY_THRESHOLD:

        return "BAD"

    return "GOOD"


# ============================================================
# DRAW BOX
# ============================================================

def draw_box(
    image,
    x1,
    y1,
    x2,
    y2,
    fiber_type,
    quality,
    confidence
):

    # --------------------------------------------------------
    # GOOD = GREEN
    # BAD = RED
    # --------------------------------------------------------

    if quality == "GOOD":

        color = (
            0,
            190,
            0
        )

    else:

        color = (
            0,
            0,
            255
        )


    # --------------------------------------------------------
    # BOX
    # --------------------------------------------------------

    cv2.rectangle(
        image,

        (x1, y1),

        (x2, y2),

        color,

        5
    )


    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    label = (
        f"{fiber_type} "
        f"{confidence * 100:.1f}%"
    )


    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )


    font_scale = 0.70

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
        text_height + 18
    )


    # --------------------------------------------------------
    # LABEL BACKGROUND
    # --------------------------------------------------------

    cv2.rectangle(
        image,

        (
            x1,

            label_y
            - text_height
            - 13
        ),

        (
            x1
            + text_width
            + 14,

            label_y + 5
        ),

        color,

        -1
    )


    # --------------------------------------------------------
    # LABEL TEXT
    # --------------------------------------------------------

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

def process_image(
    image
):

    image_array = np.array(
        image
    )


    result = model.predict(
        source=image_array,

        conf=0.25,

        verbose=False
    )[0]


    output = image_array.copy()


    detections = []


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


            # Only known fiber classes

            if fiber_type not in [
                "LOOP FIBER",
                "PROTRUDING FIBER"
            ]:

                continue


            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            quality = get_quality(
                confidence
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
                        confidence
                }
            )


            # ------------------------------------------------
            # DRAW BOTH GOOD AND BAD
            # ------------------------------------------------

            draw_box(
                output,

                x1,
                y1,
                x2,
                y2,

                fiber_type,

                quality,

                confidence
            )


    # ========================================================
    # OVERALL QUALITY
    # ========================================================

    if len(detections) == 0:

        overall_quality = "GOOD"

    elif any(
        d["quality"] == "BAD"
        for d in detections
    ):

        overall_quality = "BAD"

    else:

        overall_quality = "GOOD"


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


    # Last boxes are retained so that
    # boxes do not disappear immediately.

    last_detections = []


    all_detections = []


    while True:

        ret, frame = cap.read()


        if not ret:

            break


        result = model.predict(
            source=frame,

            conf=0.25,

            verbose=False
        )[0]


        current_detections = []


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


                if fiber_type not in [
                    "LOOP FIBER",
                    "PROTRUDING FIBER"
                ]:

                    continue


                quality = get_quality(
                    confidence
                )


                current_detections.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,

                        fiber_type,

                        quality,

                        confidence
                    )
                )


                all_detections.append(
                    {
                        "fiber_type":
                            fiber_type,

                        "quality":
                            quality,

                        "confidence":
                            confidence
                    }
                )


        # ----------------------------------------------------
        # KEEP LAST DETECTIONS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # DRAW BOXES
        # ----------------------------------------------------

        for (
            x1,
            y1,
            x2,
            y2,

            fiber_type,

            quality,

            confidence

        ) in detections_to_draw:

            draw_box(
                processed,

                x1,
                y1,
                x2,
                y2,

                fiber_type,

                quality,

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


    # ========================================================
    # FINAL QUALITY
    # ========================================================

    if len(all_detections) == 0:

        overall_quality = "GOOD"

    elif any(
        d["quality"] == "BAD"
        for d in all_detections
    ):

        overall_quality = "BAD"

    else:

        overall_quality = "GOOD"


    return (
        final_video,
        all_detections,
        overall_quality
    )


# ============================================================
# SHOW IMAGE DETECTIONS
# ============================================================

def show_image_detections(
    detections
):

    if len(detections) == 0:

        st.info(
            "No fiber detected."
        )

        return


    shown = set()


    for detection in detections:

        key = (
            detection["fiber_type"],

            detection["quality"],

            round(
                detection["confidence"],
                2
            )
        )


        if key in shown:

            continue


        shown.add(key)


        confidence = (
            detection["confidence"]
            * 100
        )


        if detection["quality"] == "GOOD":

            st.markdown(
                f"""
                <div class="detection-card">

                🟢
                <b>
                {detection["fiber_type"]}
                </b>

                <br>

                Quality:
                <b>GOOD QUALITY</b>

                <br>

                Confidence:
                <b>{confidence:.2f}%</b>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="detection-card">

                🔴
                <b>
                {detection["fiber_type"]}
                </b>

                <br>

                Quality:
                <b>BAD QUALITY</b>

                <br>

                Confidence:
                <b>{confidence:.2f}%</b>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# SHOW VIDEO DETECTIONS
# ============================================================

def show_video_detections(
    detections
):

    if len(detections) == 0:

        st.info(
            "No fiber detected."
        )

        return


    # Keep highest confidence for each
    # fiber + quality combination.

    best = {}


    for detection in detections:

        key = (
            detection["fiber_type"],
            detection["quality"]
        )


        confidence = (
            detection["confidence"]
        )


        if (
            key not in best
            or confidence > best[key]
        ):

            best[key] = confidence


    for (
        fiber_type,
        quality
    ), confidence in best.items():

        confidence_percent = (
            confidence * 100
        )


        if quality == "GOOD":

            st.markdown(
                f"""
                <div class="detection-card">

                🟢
                <b>{fiber_type}</b>

                <br>

                Quality:
                <b>GOOD QUALITY</b>

                <br>

                Confidence:
                <b>{confidence_percent:.2f}%</b>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="detection-card">

                🔴
                <b>{fiber_type}</b>

                <br>

                Quality:
                <b>BAD QUALITY</b>

                <br>

                Confidence:
                <b>{confidence_percent:.2f}%</b>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# ============================================================
# HOME PAGE
# ============================================================
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
    # LEFT
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
    # RIGHT
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

                The system accepts yarn images, camera input,
                and videos. A trained YOLO model analyzes
                the yarn and identifies fiber regions.

                </p>

                <p style="
                    font-size:15px;
                    line-height:1.7;
                ">

                YarnX provides GOOD or BAD quality results
                and displays detected fiber regions with
                colored bounding boxes and confidence scores.

                </p>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # TEAM
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
# ============================================================
# INSPECTION PAGE
# ============================================================
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

        st.session_state.image_detections = []

        st.session_state.video_result = None

        st.session_state.video_quality = None

        st.session_state.video_detections = []

        st.rerun()


    st.write("")


    # ========================================================
    # INPUT / RESULT
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


                # SMALL INPUT

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
                            detections,
                            quality
                        ) = process_image(
                            image
                        )


                    st.session_state.image_result = (
                        result_image
                    )

                    st.session_state.image_detections = (
                        detections
                    )

                    st.session_state.image_quality = (
                        quality
                    )

                    st.session_state.video_result = (
                        None
                    )

                    st.session_state.video_detections = (
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
                            detections,
                            quality
                        ) = process_image(
                            image
                        )


                    st.session_state.image_result = (
                        result_image
                    )

                    st.session_state.image_detections = (
                        detections
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
                            detections,
                            quality
                        ) = process_video(
                            temp_video.name
                        )


                    st.session_state.video_result = (
                        output_video
                    )

                    st.session_state.video_detections = (
                        detections
                    )

                    st.session_state.video_quality = (
                        quality
                    )

                    st.session_state.image_result = (
                        None
                    )

                    st.session_state.image_detections = (
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
                "**ANALYZED IMAGE**"
            )


            # SMALL OUTPUT

            st.image(
                st.session_state.image_result,

                width=330
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
            # DETECTIONS
            # =================================================

            st.write(
                "### Detected Fiber"
            )


            show_image_detections(
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


            # SMALL OUTPUT VIDEO

            st.video(
                st.session_state.video_result,

                width=340
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


            # =================================================
            # VIDEO DETECTIONS
            # =================================================

            st.write(
                "### Detected Fiber"
            )


            show_video_detections(
                st.session_state.video_detections
            )


        # ====================================================
        # BEFORE ANALYSIS
        # ====================================================

        else:

            st.info(
                "Upload an image or video "
                "and click Analyze."
            )
