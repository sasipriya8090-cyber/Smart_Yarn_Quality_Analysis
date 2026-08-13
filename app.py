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
# MODEL CONFIGURATION
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

        padding-bottom: 1.5rem !important;

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

        margin: 0 0 20px 0;

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

        min-height: 48px;

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


    .stButton > button:focus {

        color: white !important;

        border: none !important;

        outline: none !important;

    }


    /* ========================================================
       AICW CARD
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


    /* ========================================================
       PROJECT DESCRIPTION
       ======================================================== */

    .project-heading {

        color: #4a148c;

        font-size: 28px;

        font-weight: 800;

        margin-bottom: 18px;

    }


    /* ========================================================
       TEAM / GUIDE
       ======================================================== */

    .team-item {

        font-size: 15px;

        color: #24324a;

        margin: 10px 0;

        line-height: 1.6;

    }


    .email-item {

        font-size: 14px;

        color: #315bb5;

        margin: 13px 0;

        line-height: 1.5;

        word-break: break-word;

    }


    .guide-value {

        font-size: 15px;

        color: #24324a;

        margin-top: 8px;

        margin-bottom: 22px;

    }


    /* ========================================================
       QUALITY RESULT
       ======================================================== */

    .good-quality {

        border: 2px solid #2e7d32;

        border-radius: 12px;

        padding: 10px;

        text-align: center;

        font-size: 21px;

        font-weight: bold;

        color: #1b5e20;

        background: #e8f5e9;

        margin-top: 12px;

    }


    .bad-quality {

        border: 2px solid #c62828;

        border-radius: 12px;

        padding: 10px;

        text-align: center;

        font-size: 21px;

        font-weight: bold;

        color: #b71c1c;

        background: #ffebee;

        margin-top: 12px;

    }


    /* ========================================================
       DEFECT CARD
       ======================================================== */

    .defect-card {

        border: 1px solid #ef9a9a;

        border-radius: 9px;

        padding: 9px 12px;

        margin-top: 7px;

        background: #fff8f8;

        font-size: 15px;

    }


    /* ========================================================
       IMAGE CONTAINER
       ======================================================== */

    .image-label {

        font-size: 14px;

        font-weight: 700;

        color: #24324a;

        margin-bottom: 5px;

    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DRAW YOLO BOUNDING BOXES
# ============================================================

def draw_yolo_boxes(frame, result):

    output = frame.copy()

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


        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            output,

            (x1, y1),

            (x2, y2),

            (0, 0, 255),

            5
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"{defect_name} "
            f"{confidence * 100:.1f}%"
        )


        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.70

        thickness = 2


        text_size, baseline = cv2.getTextSize(
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

            (0, 0, 255),

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
# VIDEO PROCESSING
# ============================================================

def process_video(input_path):

    cap = cv2.VideoCapture(input_path)


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


    output_file = tempfile.NamedTemporaryFile(
        delete=False,

        suffix=".mp4"
    )


    output_path = output_file.name

    output_file.close()


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        output_path,

        fourcc,

        fps,

        (width, height)
    )


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


                defect_name = model.names[class_id]


                current_boxes.append(
                    {
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        "name": defect_name,

                        "confidence": confidence
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

            confidence = detection["confidence"]


            # BOX

            cv2.rectangle(
                processed_frame,

                (x1, y1),

                (x2, y2),

                (0, 0, 255),

                5
            )


            # LABEL

            label = (
                f"{name} "
                f"{confidence * 100:.1f}%"
            )


            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.70

            thickness = 2


            text_size, baseline = cv2.getTextSize(
                label,

                font,

                font_scale,

                thickness
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
                    label_y - text_height - 12
                ),

                (
                    x1 + text_width + 12,
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


    return output_path, all_defects


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
    # AICW BOX
    # ========================================================

    with left:

        with st.container(border=True):

            st.markdown(
                '<div class="aicw-title">'
                'AI Career for Women'
                '<br>'
                '(AICW)'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="capstone-title">'
                'Capstone Project'
                '</div>',
                unsafe_allow_html=True
            )


        st.write("")


        # PREDICT

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "inspection"

            st.rerun()


    # ========================================================
    # PROJECT DESCRIPTION BOX
    # ========================================================

    with right:

        with st.container(border=True):

            st.markdown(
                '<div class="project-heading">'
                'Project Description'
                '</div>',
                unsafe_allow_html=True
            )


            st.write(
                """
                YarnX is an AI-powered yarn quality inspection
                system designed to automatically detect and
                identify yarn defects using Computer Vision
                and Deep Learning.
                """
            )


            st.write(
                """
                The system accepts yarn images, camera input,
                and videos for inspection. A trained YOLO model
                analyzes the yarn and identifies defective
                regions by drawing bounding boxes around
                detected defects.
                """
            )


            st.write(
                """
                The system displays the detected defect,
                confidence score, and final quality result as
                GOOD or BAD. This helps reduce manual inspection
                effort and supports faster and more accurate
                yarn quality assessment.
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
    # TEAM MEMBERS
    # ========================================================

    with team:

        with st.container(border=True):

            st.markdown(
                "### 👩‍💻 TEAM MEMBERS"
            )


            st.markdown(
                '<div class="team-item">'
                '1. &nbsp; Gutti.Pavani Devi Priya'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="team-item">'
                '2. &nbsp; Somasani.Sasi Priya'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="team-item">'
                '3. &nbsp; Galidevara.Rama Devi'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="team-item">'
                '4. &nbsp; Rambala.Harshitha Sai Lakshmi'
                '</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # GMAIL
    # ========================================================

    with gmail:

        with st.container(border=True):

            st.markdown(
                "### 📧 GMAIL"
            )


            st.markdown(
                '<div class="email-item">'
                '<a href="mailto:gutthipavanidevipriya@gmail.com">'
                'gutthipavanidevipriya@gmail.com'
                '</a>'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="email-item">'
                '<a href="mailto:Sasipriya8090@gmail.com">'
                'Sasipriya8090@gmail.com'
                '</a>'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="email-item">'
                '<a href="mailto:ramadevigalidevara0@gmail.com">'
                'ramadevigalidevara0@gmail.com'
                '</a>'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="email-item">'
                '<a href="mailto:harshitharambala3@gmail.com">'
                'harshitharambala3@gmail.com'
                '</a>'
                '</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # GUIDE
    # ========================================================

    with guide:

        with st.container(border=True):

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
    # EQUAL INPUT + OUTPUT
    # ========================================================

    input_col, output_col = st.columns(
        [1, 1],

        gap="medium"
    )


    # ========================================================
    # INPUT SIDE
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
                    '<div class="image-label">'
                    'INPUT PREVIEW'
                    '</div>',
                    unsafe_allow_html=True
                )


                # IMPORTANT:
                # Direct st.image is used.
                # No base64 / HTML image display.

                st.image(
                    image,

                    width=400
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
                    '<div class="image-label">'
                    'CAMERA PREVIEW'
                    '</div>',
                    unsafe_allow_html=True
                )


                st.image(
                    image,

                    width=400
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

                video_input_file = (
                    tempfile.NamedTemporaryFile(
                        delete=False,

                        suffix=".mp4"
                    )
                )


                video_input_file.write(
                    uploaded_video.getvalue()
                )


                video_input_file.close()


                st.markdown(
                    '<div class="image-label">'
                    'INPUT VIDEO'
                    '</div>',
                    unsafe_allow_html=True
                )


                # ACTUAL VIDEO

                st.video(
                    uploaded_video.getvalue()
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
                                video_input_file.name
                            )
                        )


                    if output_video is None:

                        st.error(
                            "❌ Unable to process video."
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


                        st.success(
                            "✅ Video analysis completed."
                        )


                        st.rerun()


    # ========================================================
    # OUTPUT SIDE
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
                '<div class="image-label">'
                'ANALYZED IMAGE'
                '</div>',
                unsafe_allow_html=True
            )


            # IMPORTANT:
            # Direct st.image.
            # No base64 code.

            st.image(
                st.session_state.image_output,

                width=500
            )


            defects = (
                st.session_state.image_defects
            )


            # BAD

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

                    st.markdown(
                        f"""
                        <div class="defect-card">

                            🔴 <b>Defect:</b>
                            {defect["name"]}

                            &nbsp;&nbsp;

                            📊 <b>Confidence:</b>
                            {defect["confidence"] * 100:.2f}%

                        </div>
                        """,

                        unsafe_allow_html=True
                    )


            # GOOD

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
                '<div class="image-label">'
                'ANALYZED VIDEO'
                '</div>',
                unsafe_allow_html=True
            )


            # ACTUAL PROCESSED VIDEO

            try:

                with open(
                    st.session_state.video_output,

                    "rb"
                ) as video_file:

                    video_bytes = (
                        video_file.read()
                    )


                st.video(
                    video_bytes
                )


            except Exception as e:

                st.error(
                    f"❌ Unable to display video: {e}"
                )


            defects = (
                st.session_state.video_defects
            )


            # BAD

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

                    st.markdown(
                        f"""
                        <div class="defect-card">

                            🔴 <b>Defect:</b>
                            {name}

                            &nbsp;&nbsp;

                            📊 <b>Confidence:</b>
                            {confidence * 100:.2f}%

                        </div>
                        """,

                        unsafe_allow_html=True
                    )


            # GOOD

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
        # INITIAL MESSAGE
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
