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

        st.error(
            f"❌ Model file not found: {MODEL_PATH}"
        )

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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       PAGE TOP SPACE
       ======================================================== */

    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1rem !important;
    }


    /* ========================================================
       YARNX TITLE BOX
       ======================================================== */

    .main-title {

        width: 100%;
        height: 72px;

        box-sizing: border-box;

        border: 2px solid #6a1b9a;
        border-radius: 14px;

        margin: 0 0 18px 0;
        padding: 0 20px;

        display: flex;
        align-items: center;
        justify-content: center;

        text-align: center;

        font-size: 28px;
        font-weight: 800;

        line-height: 1.2;

        color: #4a148c;

        background: linear-gradient(
            90deg,
            #f3e5f5 0%,
            #e3f2fd 50%,
            #fce4ec 100%
        );

        box-shadow:
            0 2px 8px rgba(90, 50, 130, 0.08);
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        width: 100%;

        min-height: 46px;

        border: none !important;

        border-radius: 13px !important;

        font-size: 16px !important;

        font-weight: 800 !important;

        color: white !important;

        background: linear-gradient(
            135deg,
            #6a1b9a 0%,
            #3949ab 50%,
            #1976d2 100%
        ) !important;

        box-shadow:
            0 5px 14px rgba(70, 55, 150, 0.25) !important;

        transition: all 0.2s ease-in-out !important;
    }


    .stButton > button:hover {

        color: white !important;

        background: linear-gradient(
            135deg,
            #7b1fa2 0%,
            #3f51b5 50%,
            #1e88e5 100%
        ) !important;

        transform: translateY(-2px);

        box-shadow:
            0 8px 18px rgba(70, 55, 150, 0.30) !important;
    }


    .stButton > button:focus,
    .stButton > button:active {

        color: white !important;

        border: none !important;

        outline: none !important;
    }


    /* ========================================================
       FIRST PAGE AICW CARD
       ======================================================== */

    .aicw-title {

        text-align: center;

        font-size: 31px;

        font-weight: 700;

        color: #24324a;

        line-height: 1.35;

        margin-top: 55px;
    }


    .capstone-title {

        text-align: center;

        font-size: 25px;

        font-weight: 700;

        color: #24324a;

        margin-top: 42px;

        margin-bottom: 45px;
    }


    .project-heading {

        color: #4a148c;

        font-size: 28px;

        font-weight: 800;

        margin-bottom: 16px;
    }


    /* ========================================================
       TEAM / EMAIL / GUIDE
       ======================================================== */

    .team-item {

        font-size: 15px;

        color: #24324a;

        margin: 9px 0;

        line-height: 1.5;
    }


    .email-item {

        font-size: 14px;

        color: #315bb5;

        margin: 10px 0;

        line-height: 1.4;

        word-break: break-word;
    }


    .guide-value {

        font-size: 15px;

        color: #24324a;

        margin-top: 6px;

        margin-bottom: 16px;
    }


    /* ========================================================
       SECOND PAGE MEDIA BOX
       ======================================================== */

    .media-box {

        width: 100%;

        max-width: 390px;

        height: 220px;

        margin: 6px auto 10px auto;

        border: 2px solid #90caf9;

        border-radius: 12px;

        background: #f5faff;

        display: flex;

        align-items: center;

        justify-content: center;

        overflow: hidden;

        box-sizing: border-box;
    }


    .media-box.output-box {

        border-color: #ce93d8;

        background: #fcf5ff;
    }


    .media-box img {

        max-width: 100%;

        max-height: 100%;

        width: auto;

        height: auto;

        object-fit: contain;

        display: block;
    }


    .media-box video {

        width: 100%;

        height: 100%;

        object-fit: contain;

        display: block;
    }


    .media-title {

        font-size: 13px;

        font-weight: 800;

        color: #24324a;

        margin: 4px 0 4px 0;
    }


    /* ========================================================
       QUALITY
       ======================================================== */

    .good-quality {

        border: 2px solid #2e7d32;

        border-radius: 12px;

        padding: 8px;

        text-align: center;

        font-size: 20px;

        font-weight: bold;

        color: #1b5e20;

        background: #e8f5e9;

        margin-top: 8px;
    }


    .bad-quality {

        border: 2px solid #c62828;

        border-radius: 12px;

        padding: 8px;

        text-align: center;

        font-size: 20px;

        font-weight: bold;

        color: #b71c1c;

        background: #ffebee;

        margin-top: 8px;
    }


    /* ========================================================
       DEFECT CARD
       ======================================================== */

    .defect-card {

        border: 1px solid #ef9a9a;

        border-radius: 9px;

        padding: 7px 10px;

        margin-top: 5px;

        background: #fff8f8;

        color: #24324a;

        font-size: 14px;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DISPLAY IMAGE INSIDE FIXED BOX
# ============================================================

def display_image_box(
    image,
    output=False
):

    if isinstance(image, Image.Image):

        pil_image = image.convert("RGB")

    elif isinstance(image, np.ndarray):

        if image.ndim == 3:

            pil_image = Image.fromarray(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )
            )

        else:

            pil_image = Image.fromarray(
                image
            )

    else:

        return


    # Resize only for display.
    # Original image remains unchanged.

    display_image = pil_image.copy()

    display_image.thumbnail(
        (370, 205),
        Image.Resampling.LANCZOS
    )


    border_class = (
        "media-box output-box"
        if output
        else "media-box"
    )


    st.markdown(
        f"""
        <div class="{border_class}">

            <img
                src="data:image/jpeg;base64,{image_to_base64(display_image)}"
            >

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMAGE TO BASE64
# ============================================================

def image_to_base64(image):

    import io
    import base64

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# VIDEO TO BROWSER PLAYABLE MP4
# ============================================================

def convert_video_to_browser_format(
    input_path
):

    try:

        import imageio_ffmpeg

        ffmpeg_path = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

    except Exception:

        return input_path


    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name


    command = [

        ffmpeg_path,

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
# DISPLAY VIDEO INSIDE FIXED BOX
# ============================================================

def display_video_box(
    video_path,
    output=False
):

    try:

        with open(
            video_path,
            "rb"
        ) as video_file:

            video_bytes = (
                video_file.read()
            )


        import base64

        encoded_video = (
            base64.b64encode(
                video_bytes
            ).decode("utf-8")
        )


        border_class = (
            "media-box output-box"
            if output
            else "media-box"
        )


        st.markdown(
            f"""
            <div class="{border_class}">

                <video
                    controls
                    preload="metadata"
                >

                    <source
                        src="data:video/mp4;base64,{encoded_video}"
                        type="video/mp4"
                    >

                    Your browser does not support
                    video playback.

                </video>

            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            f"❌ Video display error: {e}"
        )


# ============================================================
# YOLO IMAGE DETECTION
# ============================================================

def detect_image(image):

    image_array = np.array(
        image.convert("RGB")
    )


    result = model.predict(
        source=image_array,

        conf=0.15,

        verbose=False
    )[0]


    output = image_array.copy()

    defects = []


    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

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


        defects.append(
            {
                "name": defect_name,

                "confidence": confidence
            }
        )


        # Bounding box

        cv2.rectangle(
            output,

            (x1, y1),

            (x2, y2),

            (255, 0, 0),

            5
        )


        # Label

        label = (
            f"{defect_name} "
            f"{confidence * 100:.1f}%"
        )


        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = 0.70

        thickness = 2


        text_size, baseline = (
            cv2.getTextSize(
                label,

                font,

                scale,

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

            scale,

            (255, 255, 255),

            thickness,

            cv2.LINE_AA
        )


    return output, defects


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

        return None, {}


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


    raw_output = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name


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

        return None, {}


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


                if (
                    defect_name
                    not in all_defects
                ):

                    all_defects[
                        defect_name
                    ] = confidence

                elif (
                    confidence
                    >
                    all_defects[
                        defect_name
                    ]
                ):

                    all_defects[
                        defect_name
                    ] = confidence


        if len(current_boxes) > 0:

            last_boxes = current_boxes


        boxes_to_draw = (
            current_boxes
            if len(current_boxes) > 0
            else last_boxes
        )


        processed_frame = (
            frame.copy()
        )


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

                (255, 0, 0),

                5
            )


            label = (
                f"{name} "
                f"{confidence * 100:.1f}%"
            )


            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            scale = 0.70

            thickness = 2


            text_size, baseline = (
                cv2.getTextSize(
                    label,

                    font,

                    scale,

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

                (255, 0, 0),

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

                scale,

                (255, 255, 255),

                thickness,

                cv2.LINE_AA
            )


        writer.write(
            processed_frame
        )


    cap.release()

    writer.release()


    # Convert to browser playable H.264

    final_output = (
        convert_video_to_browser_format(
            raw_output
        )
    )


    return final_output, all_defects


# ============================================================
# FIRST PAGE
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
    # TOP SECTION
    # ========================================================

    left, right = st.columns(
        [35, 65],

        gap="small"
    )


    # ========================================================
    # AICW CARD
    # ========================================================

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                """
                <div class="aicw-title">

                    AI Career for Women

                    <br>

                    (AICW)

                </div>
                """,

                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div class="capstone-title">

                    Capstone Project

                </div>
                """,

                unsafe_allow_html=True
            )


        st.write("")


        if st.button(
            "🔍 PREDICT",

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
                <div class="project-heading">

                    Project Description

                </div>
                """,

                unsafe_allow_html=True
            )


            st.write(
                """
                YarnX is an AI-powered yarn quality
                inspection system designed to automatically
                detect and identify yarn defects using
                Computer Vision and Deep Learning.
                """
            )


            st.write(
                """
                The system accepts yarn images, camera
                input, and videos for inspection. A trained
                YOLO model analyzes the yarn and identifies
                defective regions by drawing bounding boxes
                around detected defects.
                """
            )


            st.write(
                """
                The system displays the detected defect,
                confidence score, and final quality result
                as GOOD or BAD. This helps reduce manual
                inspection effort and supports faster and
                more accurate yarn quality assessment.
                """
            )


    # ========================================================
    # TEAM / GMAIL / GUIDE
    # ========================================================

    st.write("")

    st.write("")


    team, gmail, guide = st.columns(
        [1.35, 1.25, 0.9],

        gap="medium"
    )


    # ========================================================
    # TEAM
    # ========================================================

    with team:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 👩‍💻 TEAM MEMBERS"
            )


            st.markdown(
                '<div class="team-item">'
                '1. Gutti.Pavani Devi Priya'
                '</div>',

                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="team-item">'
                '2. Somasani.Sasi Priya'
                '</div>',

                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="team-item">'
                '3. Galidevara.Rama Devi'
                '</div>',

                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="team-item">'
                '4. Rambala.Harshitha Sai Lakshmi'
                '</div>',

                unsafe_allow_html=True
            )


    # ========================================================
    # GMAIL
    # ========================================================

    with gmail:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 📧 GMAIL"
            )


            st.markdown(
                '<div class="email-item">'
                'gutthipavanidevipriya@gmail.com'
                '</div>',

                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="email-item">'
                'Sasipriya8090@gmail.com'
                '</div>',

                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="email-item">'
                'ramadevigalidevara0@gmail.com'
                '</div>',

                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="email-item">'
                'harshitharambala3@gmail.com'
                '</div>',

                unsafe_allow_html=True
            )


    # ========================================================
    # GUIDE
    # ========================================================

    with guide:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🎓 GUIDE NAME"
            )


            st.markdown(
                '<div class="guide-value">'
                'Md. Abdul Aziz'
                '</div>',

                unsafe_allow_html=True
            )


            st.markdown(
                "### DESIGNATION"
            )


            st.markdown(
                '<div class="guide-value">'
                'Co Lead & Trainer AICW'
                '</div>',

                unsafe_allow_html=True
            )


# ============================================================
# SECOND PAGE
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
    # BACK BUTTON
    # ========================================================

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.session_state.image_output = None

        st.session_state.image_defects = []

        st.session_state.video_output = None

        st.session_state.video_defects = {}

        st.rerun()


    st.write("")


    # ========================================================
    # EQUAL COLUMNS
    # ========================================================

    input_col, output_col = st.columns(
        [1, 1],

        gap="medium"
    )


    # ========================================================
    # INPUT
    # ========================================================

    with input_col:

        st.subheader(
            "📥 INPUT"
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


                st.markdown(
                    '<div class="media-title">'
                    'INPUT PREVIEW'
                    '</div>',

                    unsafe_allow_html=True
                )


                display_image_box(
                    image,

                    output=False
                )


                if st.button(
                    "🔍 Analyze Image",

                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        output_image, defects = (
                            detect_image(
                                image
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


                st.markdown(
                    '<div class="media-title">'
                    'CAMERA PREVIEW'
                    '</div>',

                    unsafe_allow_html=True
                )


                display_image_box(
                    image,

                    output=False
                )


                if st.button(
                    "🔍 Analyze Camera",

                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        output_image, defects = (
                            detect_image(
                                image
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

                input_video_file = (
                    tempfile.NamedTemporaryFile(
                        delete=False,

                        suffix=".mp4"
                    )
                )


                input_video_file.write(
                    uploaded_video.getvalue()
                )


                input_video_file.close()


                st.markdown(
                    '<div class="media-title">'
                    'INPUT VIDEO'
                    '</div>',

                    unsafe_allow_html=True
                )


                display_video_box(
                    input_video_file.name,

                    output=False
                )


                if st.button(
                    "🔍 Analyze Video",

                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        output_video, defects = (
                            process_video(
                                input_video_file.name
                            )
                        )


                    if output_video is None:

                        st.error(
                            "❌ Video processing failed."
                        )

                    else:

                        st.session_state.video_output = (
                            output_video
                        )


                        st.session_state.video_defects = (
                            defects
                        )


                        st.session_state.image_output = None

                        st.session_state.image_defects = []


                        st.rerun()


    # ========================================================
    # OUTPUT
    # ========================================================

    with output_col:

        st.subheader(
            "🤖 INSPECTION RESULT"
        )


        # ====================================================
        # IMAGE OUTPUT
        # ====================================================

        if (
            st.session_state.image_output
            is not None
        ):

            st.markdown(
                '<div class="media-title">'
                'ANALYZED IMAGE'
                '</div>',

                unsafe_allow_html=True
            )


            display_image_box(
                st.session_state.image_output,

                output=True
            )


            defects = (
                st.session_state.image_defects
            )


            if len(defects) > 0:

                st.markdown(
                    """
                    <div class="bad-quality">
                        ❌ BAD QUALITY
                    </div>
                    """,

                    unsafe_allow_html=True
                )


                st.write(
                    "### Detected Defects"
                )


                for defect in defects:

                    # NORMAL STREAMLIT TEXT
                    # NO HTML TAGS

                    with st.container(
                        border=True
                    ):

                        st.write(
                            f"🔴 Defect: {defect['name']}"
                        )

                        st.write(
                            f"📊 Confidence: "
                            f"{defect['confidence'] * 100:.2f}%"
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
        # VIDEO OUTPUT
        # ====================================================

        elif (
            st.session_state.video_output
            is not None
        ):

            st.markdown(
                '<div class="media-title">'
                'ANALYZED VIDEO'
                '</div>',

                unsafe_allow_html=True
            )


            # PLAYABLE FIXED VIDEO BOX

            display_video_box(
                st.session_state.video_output,

                output=True
            )


            defects = (
                st.session_state.video_defects
            )


            if len(defects) > 0:

                st.markdown(
                    """
                    <div class="bad-quality">
                        ❌ BAD QUALITY
                    </div>
                    """,

                    unsafe_allow_html=True
                )


                st.write(
                    "### Detected Defects"
                )


                for name, confidence in (
                    defects.items()
                ):

                    with st.container(
                        border=True
                    ):

                        st.write(
                            f"🔴 Defect: {name}"
                        )

                        st.write(
                            f"📊 Confidence: "
                            f"{confidence * 100:.2f}%"
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
        # NO RESULT
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
