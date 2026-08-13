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
# PAGE CONFIGURATION
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
       PAGE
       ======================================================== */

    .block-container {

        padding-top: 2.0rem !important;

        padding-bottom: 1.5rem !important;

        max-width: 100% !important;
    }


    /* ========================================================
       YARNX TITLE
       ======================================================== */

    .yarnx-title {

        width: 100%;

        min-height: 78px;

        box-sizing: border-box;

        border: 2px solid #6a1b9a;

        border-radius: 14px;

        display: flex;

        align-items: center;

        justify-content: center;

        padding: 12px 25px;

        margin: 0 0 18px 0;

        text-align: center;

        font-size: 28px;

        font-weight: 800;

        line-height: 1.25;

        color: #43248a;

        background: linear-gradient(
            90deg,
            #f3e5f5 0%,
            #e3f2fd 50%,
            #fce4ec 100%
        );

        overflow: visible;
    }


    /* ========================================================
       TOP CARDS
       ======================================================== */

    .project-card {

        border: 1px solid #d9d9d9;

        border-radius: 10px;

        background: white;

        box-sizing: border-box;
    }


    /* ========================================================
       AICW CARD
       ======================================================== */

    .aicw-content {

        min-height: 265px;

        display: flex;

        flex-direction: column;

        align-items: center;

        justify-content: space-between;

        text-align: center;

        padding: 22px 15px;
    }


    .aicw-title {

        font-size: 32px;

        font-weight: 700;

        color: #24324a;

        line-height: 1.2;

        margin: 0;
    }


    .aicw-subtitle {

        font-size: 30px;

        font-weight: 700;

        color: #24324a;

        line-height: 1.2;

        margin-top: 10px;
    }


    .capstone-title {

        font-size: 25px;

        font-weight: 700;

        color: #24324a;

        margin: 0;
    }


    /* ========================================================
       PROJECT DESCRIPTION
       ======================================================== */

    .project-heading {

        font-size: 30px;

        font-weight: 700;

        color: #4a148c;

        margin-bottom: 18px;
    }


    .project-text {

        font-size: 16px;

        color: #24324a;

        line-height: 1.75;

        margin-bottom: 16px;
    }


    /* ========================================================
       PREDICT BUTTON
       ======================================================== */

    .predict-area .stButton > button {

        width: 100%;

        height: 46px;

        border: 1px solid #d2d5dc !important;

        border-radius: 10px !important;

        background: #f4f6f9 !important;

        color: #24324a !important;

        font-size: 15px !important;

        font-weight: 600 !important;

        box-shadow: none !important;
    }


    .predict-area .stButton > button:hover {

        background: #eef0f5 !important;

        border: 1px solid #c5c9d2 !important;

        color: #24324a !important;

        transform: none !important;

        box-shadow: none !important;
    }


    /* ========================================================
       BACK BUTTON
       ======================================================== */

    .back-area .stButton > button {

        width: auto;

        min-width: 78px;

        height: 44px;

        border-radius: 10px !important;

        border: none !important;

        background: linear-gradient(
            135deg,
            #6a1b9a,
            #1976d2
        ) !important;

        color: white !important;

        font-weight: 700 !important;

        box-shadow:
            0 5px 12px rgba(70, 55, 150, 0.20) !important;
    }


    /* ========================================================
       ANALYZE BUTTON
       ======================================================== */

    .analyze-area .stButton > button {

        width: 100%;

        min-height: 48px;

        border: none !important;

        border-radius: 13px !important;

        background: linear-gradient(
            135deg,
            #6a1b9a,
            #1976d2
        ) !important;

        color: white !important;

        font-weight: 700 !important;

        box-shadow:
            0 5px 14px rgba(70, 55, 150, 0.25) !important;
    }


    /* ========================================================
       QUALITY RESULT
       ======================================================== */

    .good-quality {

        width: 100%;

        box-sizing: border-box;

        border: 2px solid #4f7f42;

        border-radius: 12px;

        padding: 10px;

        text-align: center;

        font-size: 20px;

        font-weight: 700;

        color: #2e6330;

        background: #edf7ea;

        margin-top: 12px;
    }


    .bad-quality {

        width: 100%;

        box-sizing: border-box;

        border: 2px solid #c62828;

        border-radius: 12px;

        padding: 10px;

        text-align: center;

        font-size: 20px;

        font-weight: 700;

        color: #b71c1c;

        background: #fff0f1;

        margin-top: 12px;
    }


    /* ========================================================
       DEFECT CARD
       ======================================================== */

    .defect-card {

        width: 100%;

        box-sizing: border-box;

        border: 1px solid #ef9a9a;

        border-radius: 10px;

        padding: 14px 16px;

        margin-top: 9px;

        background: #fffafa;

        color: #24324a;

        font-size: 16px;

        line-height: 1.7;
    }


    .defect-name {

        color: #24324a;

        font-weight: 700;
    }


    .confidence-value {

        color: #24324a;

        font-weight: 600;
    }


    /* ========================================================
       IMAGE / VIDEO BOX
       ======================================================== */

    .media-label {

        font-size: 15px;

        font-weight: 700;

        color: #24324a;

        margin-bottom: 8px;
    }


    /* ========================================================
       BOTTOM INFORMATION
       ======================================================== */

    .bottom-heading {

        font-size: 17px;

        font-weight: 700;

        color: #24324a;

        margin-bottom: 18px;
    }


    .bottom-text {

        font-size: 15px;

        color: #24324a;

        line-height: 2.0;

        margin: 0;
    }


    .email-text {

        font-size: 15px;

        color: #315bb5;

        line-height: 2.0;

        margin: 0;

        word-break: break-word;
    }


    .guide-name {

        font-size: 15px;

        color: #24324a;

        margin-bottom: 22px;
    }


    .designation {

        font-size: 15px;

        color: #24324a;
    }


    .designation-heading {

        font-size: 15px;

        font-weight: 700;

        color: #24324a;

        margin-bottom: 12px;
    }


    /* ========================================================
       RADIO
       ======================================================== */

    div[data-testid="stRadio"] label {

        font-weight: 600;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE FUNCTION
# ============================================================

def show_title():

    st.markdown(
        """
        <div class="yarnx-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMAGE DISPLAY
#
# IMPORTANT:
# We DO NOT convert RGB -> BGR here.
# Uploaded PIL image is already RGB.
# ============================================================

def show_image(
    image,
    width=400
):

    if isinstance(image, np.ndarray):

        # Image arrays in this app are RGB.
        image_to_show = Image.fromarray(
            image.astype(np.uint8)
        )

    elif isinstance(image, Image.Image):

        image_to_show = image.convert("RGB")

    else:

        image_to_show = image


    st.image(
        image_to_show,
        width=width
    )


# ============================================================
# DRAW YOLO BOXES ON RGB IMAGE
#
# IMPORTANT:
# OpenCV normally uses BGR.
# But this image is RGB.
# Therefore RED = (255, 0, 0)
# ============================================================

def draw_yolo_boxes_rgb(
    image_rgb,
    result
):

    output = image_rgb.copy()

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


        defect_name = str(
            model.names[class_id]
        )


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


        # ====================================================
        # RED BOUNDING BOX
        # RGB RED = (255, 0, 0)
        # ====================================================

        cv2.rectangle(
            output,

            (x1, y1),

            (x2, y2),

            (255, 0, 0),

            5
        )


        # ====================================================
        # LABEL
        # ====================================================

        label = (
            f"{defect_name} "
            f"{confidence * 100:.1f}%"
        )


        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.70

        thickness = 2


        text_size, baseline = (
            cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )
        )


        text_width, text_height = text_size


        label_y = max(
            y1,
            text_height + 15
        )


        # ====================================================
        # RED LABEL BACKGROUND
        # ====================================================

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

                label_y
                + 4
            ),

            (255, 0, 0),

            -1
        )


        # ====================================================
        # WHITE LABEL TEXT
        # ====================================================

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
# FIRST PAGE
# ============================================================

if st.session_state.page == "home":


    # ========================================================
    # YARNX TITLE
    # ========================================================

    show_title()


    # ========================================================
    # TOP TWO CARDS
    # ========================================================

    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # ========================================================
    # AICW
    # ========================================================

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                """
                <div class="aicw-content">

                    <div>

                        <div class="aicw-title">
                            AI Career for Women
                        </div>

                        <div class="aicw-subtitle">
                            (AICW)
                        </div>

                    </div>


                    <div class="capstone-title">
                        Capstone Project
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown(
            '<div class="predict-area">',
            unsafe_allow_html=True
        )


        if st.button(
            "PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "inspection"

            st.rerun()


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


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
                    GOOD or BAD. This helps reduce manual
                    inspection effort and supports faster and more
                    accurate yarn quality assessment.
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # SPACE
    # ========================================================

    st.write("")


    # ========================================================
    # BOTTOM INFORMATION
    # ========================================================

    team, gmail, guide = st.columns(
        [1.35, 1.25, 0.9],
        gap="medium"
    )


    # ========================================================
    # TEAM MEMBERS
    # ========================================================

    with team:

        st.markdown(
            """
            <div class="bottom-heading">
                TEAM MEMBERS
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="bottom-text">

                1. &nbsp; Gutti.Pavani Devi Priya<br><br>

                2. &nbsp; Somasani.Sasi Priya<br><br>

                3. &nbsp; Galidevara.Rama Devi<br><br>

                4. &nbsp; Rambala.Harshitha Sai Lakshmi

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # GMAIL
    # ========================================================

    with gmail:

        st.markdown(
            """
            <div class="bottom-heading">
                GMAIL
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="email-text">

                <a href="mailto:gutthipavanidevipriya@gmail.com">
                    gutthipavanidevipriya@gmail.com
                </a>
                <br><br>

                <a href="mailto:Sasipriya8090@gmail.com">
                    Sasipriya8090@gmail.com
                </a>
                <br><br>

                <a href="mailto:ramadevigalidevara0@gmail.com">
                    ramadevigalidevara0@gmail.com
                </a>
                <br><br>

                <a href="mailto:harshitharambala3@gmail.com">
                    harshitharambala3@gmail.com
                </a>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # GUIDE
    # ========================================================

    with guide:

        st.markdown(
            """
            <div class="bottom-heading">
                GUIDE NAME
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="guide-name">
                Md. Abdul Aziz
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="designation-heading">
                DESIGNATION
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="designation">
                Co Lead & Trainer AICW
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SECOND PAGE
# ============================================================

else:


    # ========================================================
    # TITLE
    # ========================================================

    show_title()


    # ========================================================
    # BACK BUTTON
    # ========================================================

    st.markdown(
        '<div class="back-area">',
        unsafe_allow_html=True
    )


    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.session_state.image_output = None

        st.session_state.image_defects = []

        st.session_state.video_output = None

        st.session_state.video_defects = {}

        st.rerun()


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    st.write("")


    # ========================================================
    # INPUT + OUTPUT
    # ========================================================

    left, right = st.columns(
        [1, 1],
        gap="medium"
    )


    # ========================================================
    # INPUT SIDE
    # ========================================================

    with left:

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
        # IMAGE INPUT
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
                    '<div class="media-label">'
                    'INPUT PREVIEW'
                    '</div>',
                    unsafe_allow_html=True
                )


                # =================================================
                # ACTUAL ORIGINAL IMAGE
                # =================================================

                show_image(
                    image,
                    width=400
                )


                st.write("")


                st.markdown(
                    '<div class="analyze-area">',
                    unsafe_allow_html=True
                )


                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        # PIL RGB -> NumPy RGB
                        image_rgb = np.array(
                            image
                        )


                        result = model.predict(
                            source=image_rgb,

                            conf=0.15,

                            verbose=False
                        )[0]


                    # IMPORTANT:
                    # draw_yolo_boxes_rgb keeps
                    # original image colors.

                    output_image, defects = (
                        draw_yolo_boxes_rgb(
                            image_rgb,
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


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


        # ====================================================
        # CAMERA INPUT
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
                    '<div class="media-label">'
                    'CAMERA PREVIEW'
                    '</div>',
                    unsafe_allow_html=True
                )


                show_image(
                    image,
                    width=400
                )


                st.write("")


                st.markdown(
                    '<div class="analyze-area">',
                    unsafe_allow_html=True
                )


                if st.button(
                    "🔍 Analyze Camera",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        image_rgb = np.array(
                            image
                        )


                        result = model.predict(
                            source=image_rgb,

                            conf=0.15,

                            verbose=False
                        )[0]


                    output_image, defects = (
                        draw_yolo_boxes_rgb(
                            image_rgb,
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


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


        # ====================================================
        # VIDEO INPUT
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

                # =================================================
                # SAVE ORIGINAL INPUT VIDEO
                # =================================================

                preview_file = (
                    tempfile.NamedTemporaryFile(
                        delete=False,

                        suffix=".mp4"
                    )
                )


                preview_file.write(
                    uploaded_video.getvalue()
                )


                preview_file.close()


                st.markdown(
                    '<div class="media-label">'
                    'INPUT VIDEO'
                    '</div>',
                    unsafe_allow_html=True
                )


                # Actual video

                st.video(
                    preview_file.name
                )


                st.write("")


                st.markdown(
                    '<div class="analyze-area">',
                    unsafe_allow_html=True
                )


                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        # -----------------------------------------
                        # SAVE INPUT VIDEO
                        # -----------------------------------------

                        input_temp = (
                            tempfile.NamedTemporaryFile(
                                delete=False,

                                suffix=".mp4"
                            )
                        )


                        input_temp.write(
                            uploaded_video.getvalue()
                        )


                        input_temp.close()


                        input_path = (
                            input_temp.name
                        )


                        # -----------------------------------------
                        # OPEN VIDEO
                        # -----------------------------------------

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


                        # -----------------------------------------
                        # OUTPUT VIDEO
                        # -----------------------------------------

                        output_temp = (
                            tempfile.NamedTemporaryFile(
                                delete=False,

                                suffix=".mp4"
                            )
                        )


                        output_temp.close()


                        raw_output = (
                            output_temp.name
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


                        if not writer.isOpened():

                            cap.release()

                            st.error(
                                "❌ Unable to create output video."
                            )

                            st.stop()


                        all_defects = {}


                        last_boxes = []


                        # -----------------------------------------
                        # PROCESS VIDEO
                        # -----------------------------------------

                        while True:

                            ret, frame = (
                                cap.read()
                            )


                            if not ret:
                                break


                            # Video frame from OpenCV
                            # is BGR.

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


                                    x1, y1, x2, y2 = (
                                        coords
                                    )


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


                                    defect_name = str(
                                        model.names[
                                            class_id
                                        ]
                                    )


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


                                    # Keep highest confidence

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

                                last_boxes = (
                                    current_boxes
                                )


                            if len(current_boxes) > 0:

                                boxes_to_draw = (
                                    current_boxes
                                )

                            else:

                                boxes_to_draw = (
                                    last_boxes
                                )


                            processed_frame = (
                                frame.copy()
                            )


                            # -------------------------------------
                            # DRAW VIDEO BOXES
                            #
                            # Video is BGR.
                            # RED = (0, 0, 255)
                            # -------------------------------------

                            for detection in boxes_to_draw:

                                x1, y1, x2, y2 = (
                                    detection["box"]
                                )


                                name = (
                                    detection["name"]
                                )


                                confidence = (
                                    detection["confidence"]
                                )


                                cv2.rectangle(
                                    processed_frame,

                                    (x1, y1),

                                    (x2, y2),

                                    (0, 0, 255),

                                    5
                                )


                                label = (
                                    f"{name} "
                                    f"{confidence * 100:.1f}%"
                                )


                                font = (
                                    cv2.FONT_HERSHEY_SIMPLEX
                                )


                                font_scale = 0.70

                                thickness = 2


                                text_size, baseline = (
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


                        # -----------------------------------------
                        # RELEASE VIDEO
                        # -----------------------------------------

                        cap.release()

                        writer.release()


                        # -----------------------------------------
                        # BROWSER COMPATIBILITY
                        # -----------------------------------------

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


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


    # ========================================================
    # INSPECTION RESULT
    # ========================================================

    with right:

        st.subheader(
            "🤖 INSPECTION RESULT"
        )


        # ====================================================
        # IMAGE RESULT
        # ====================================================

        if (
            st.session_state.image_output
            is not None
        ):

            st.markdown(
                '<div class="media-label">'
                'ANALYZED IMAGE'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # ACTUAL IMAGE
            #
            # image_output is RGB.
            # NO COLOR CONVERSION.
            # =================================================

            show_image(
                st.session_state.image_output,

                width=500
            )


            defects = (
                st.session_state.image_defects
            )


            # =================================================
            # BAD QUALITY
            # =================================================

            if len(defects) > 0:

                st.markdown(
                    """
                    <div class="bad-quality">
                        ❌ BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                    "### Detected Defects"
                )


                # =================================================
                # CLEAN DEFECT DETAILS
                #
                # NO RAW HTML TAGS
                # =================================================

                for defect in defects:

                    defect_name = str(
                        defect["name"]
                    )


                    confidence = (
                        defect["confidence"]
                        * 100
                    )


                    st.markdown(
                        f"""
                        <div class="defect-card">

                            🔴
                            <span class="defect-name">
                                Defect:
                            </span>

                            {defect_name}

                            <br><br>

                            📊
                            <span class="defect-name">
                                Confidence:
                            </span>

                            <span class="confidence-value">
                                {confidence:.2f}%
                            </span>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # =================================================
            # GOOD QUALITY
            # =================================================

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

        elif (
            st.session_state.video_output
            is not None
        ):

            st.markdown(
                '<div class="media-label">'
                'ANALYZED VIDEO'
                '</div>',
                unsafe_allow_html=True
            )


            # Actual analyzed video

            st.video(
                st.session_state.video_output
            )


            defects = (
                st.session_state.video_defects
            )


            # =================================================
            # BAD QUALITY
            # =================================================

            if len(defects) > 0:

                st.markdown(
                    """
                    <div class="bad-quality">
                        ❌ BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                    "### Detected Defects"
                )


                for name, confidence in (
                    defects.items()
                ):

                    name = str(name)

                    confidence_percent = (
                        confidence * 100
                    )


                    # Clean HTML rendered card

                    st.markdown(
                        f"""
                        <div class="defect-card">

                            🔴
                            <span class="defect-name">
                                Defect:
                            </span>

                            {name}

                            <br><br>

                            📊
                            <span class="defect-name">
                                Confidence:
                            </span>

                            <span class="confidence-value">
                                {confidence_percent:.2f}%
                            </span>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # =================================================
            # GOOD QUALITY
            # =================================================

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
        # DEFAULT RESULT
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
