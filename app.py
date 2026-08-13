import os
import tempfile
import subprocess
import base64

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

    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False

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

    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }


    /* ======================================================
       TITLE
       ====================================================== */

    .main-title {
        border: 2px solid #6a1b9a;
        border-radius: 14px;
        padding: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        color: #4a148c;
        background: linear-gradient(
            90deg,
            #f3e5f5,
            #e3f2fd,
            #fce4ec
        );
        margin-bottom: 14px;
    }


    /* ======================================================
       QUALITY
       ====================================================== */

    .good-quality {
        border: 2px solid #2e7d32;
        border-radius: 12px;
        padding: 9px;
        text-align: center;
        font-size: 21px;
        font-weight: bold;
        color: #1b5e20;
        background: #e8f5e9;
        margin-top: 10px;
    }


    .bad-quality {
        border: 2px solid #c62828;
        border-radius: 12px;
        padding: 9px;
        text-align: center;
        font-size: 21px;
        font-weight: bold;
        color: #b71c1c;
        background: #ffebee;
        margin-top: 10px;
    }


    /* ======================================================
       MEDIA BOX
       ====================================================== */

    .input-media-box {
        width: 320px;
        height: 210px;
        margin: 8px auto 12px auto;
        border: 2px solid #90caf9;
        border-radius: 12px;
        background: #f5faff;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }


    .output-media-box {
        width: 420px;
        height: 270px;
        margin: 8px auto 12px auto;
        border: 2px solid #ce93d8;
        border-radius: 12px;
        background: #fcf5ff;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }


    .input-media-box img {
        max-width: 300px !important;
        max-height: 190px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
    }


    .output-media-box img {
        max-width: 400px !important;
        max-height: 240px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
    }


    .defect-card {
        border: 1px solid #ef9a9a;
        border-radius: 9px;
        padding: 7px 10px;
        margin-top: 5px;
        background: #fff8f8;
    }


    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DRAW YOLO BOXES
# ============================================================

def draw_yolo_boxes(frame, result):

    output = frame.copy()

    defects = []


    if result.boxes is None:

        return output, defects


    if len(result.boxes) == 0:

        return output, defects


    for box in result.boxes:

        # -----------------------------------------------
        # COORDINATES
        # -----------------------------------------------

        coords = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        x1, y1, x2, y2 = coords


        # -----------------------------------------------
        # CONFIDENCE
        # -----------------------------------------------

        confidence = float(
            box.conf[0]
            .cpu()
            .item()
        )


        # -----------------------------------------------
        # CLASS
        # -----------------------------------------------

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


        # -----------------------------------------------
        # RED BOX
        # -----------------------------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            6
        )


        # -----------------------------------------------
        # LABEL
        # -----------------------------------------------

        label = (
            f"{defect_name} "
            f"{confidence * 100:.1f}%"
        )


        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.75

        thickness = 2


        (
            text_size,
            baseline
        ) = cv2.getTextSize(
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


        # -----------------------------------------------
        # LABEL BACKGROUND
        # -----------------------------------------------

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

            (0, 0, 255),

            -1
        )


        # -----------------------------------------------
        # LABEL TEXT
        # -----------------------------------------------

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
# RESIZE IMAGE
# ============================================================

def resize_for_display(
    image,
    max_width=400,
    max_height=240
):

    if isinstance(image, np.ndarray):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            image
        )


    image = image.copy()


    image.thumbnail(
        (
            max_width,
            max_height
        ),
        Image.Resampling.LANCZOS
    )


    return image


# ============================================================
# CONVERT VIDEO
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
# FIXED SIZE VIDEO PLAYER
# ============================================================

def show_video(
    video_path,
    width=300,
    height=190
):

    try:

        with open(
            video_path,
            "rb"
        ) as video_file:

            video_bytes = (
                video_file.read()
            )


        encoded_video = (
            base64.b64encode(
                video_bytes
            ).decode("utf-8")
        )


        html = f"""
        <div style="
            width:{width}px;
            height:{height}px;
            margin:8px auto 12px auto;
            border:2px solid #90caf9;
            border-radius:12px;
            background:#f5faff;
            display:flex;
            align-items:center;
            justify-content:center;
            overflow:hidden;
        ">

            <video
                controls
                style="
                    width:{width}px;
                    max-width:{width}px;
                    max-height:{height}px;
                    height:auto;
                    object-fit:contain;
                "
            >

                <source
                    src="data:video/mp4;base64,{encoded_video}"
                    type="video/mp4"
                >

            </video>

        </div>
        """


        st.components.v1.html(
            html,
            height=height + 35
        )


    except Exception as e:

        st.error(
            f"❌ Unable to display video: {e}"
        )


# ============================================================
# ============================================================
# FIRST PAGE
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
        gap="small"
    )


    # ========================================================
    # LEFT
    # ========================================================

    with left:

        with st.container(border=True):

            st.markdown(
                """
                <h2 style="text-align:center;">
                AI Career for Women (AICW)
                </h2>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <h3 style="text-align:center;">
                Capstone Project
                </h3>
                """,
                unsafe_allow_html=True
            )


            st.write("")
            st.write("")
            st.write("")
            st.write("")


        st.write("")


        if st.button(
            "PREDICT",
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

        with st.container(border=True):

            st.subheader(
                "Project Description"
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
    # TEAM DETAILS
    # ========================================================

    st.write("")


    team_col, mail_col, guide_col = st.columns(
        [1.35, 1.25, 0.9],
        gap="small"
    )


    with team_col:

        st.markdown(
            "**TEAM MEMBERS**"
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


    with mail_col:

        st.markdown(
            "**GMAIL**"
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

        st.markdown(
            "**GUIDE NAME**"
        )

        st.write(
            "Md. Abdul Aziz"
        )

        st.markdown(
            "**DESIGNATION**"
        )

        st.write(
            "Co Lead & Trainer AICW"
        )


# ============================================================
# ============================================================
# SECOND PAGE
# ============================================================
# ============================================================

elif st.session_state.page == "inspection":

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


    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # ========================================================
    # INPUT
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
                    "**INPUT PREVIEW**"
                )


                preview = resize_for_display(
                    image,
                    300,
                    190
                )


                st.markdown(
                    '<div class="input-media-box">',
                    unsafe_allow_html=True
                )


                st.image(
                    preview,
                    width=300
                )


                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
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


                st.write(
                    "**CAMERA PREVIEW**"
                )


                preview = resize_for_display(
                    image,
                    300,
                    190
                )


                st.markdown(
                    '<div class="input-media-box">',
                    unsafe_allow_html=True
                )


                st.image(
                    preview,
                    width=300
                )


                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
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

        elif input_type == "🎥 Video":

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

                # -------------------------------------------
                # SAVE INPUT VIDEO
                # -------------------------------------------

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


                st.write(
                    "**INPUT VIDEO**"
                )


                # EXACT SMALL INPUT VIDEO
                show_video(
                    preview_file.name,
                    300,
                    190
                )


                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        # ==================================
                        # SAVE INPUT
                        # ==================================

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


                        # ==================================
                        # OPEN VIDEO
                        # ==================================

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


                        # ==================================
                        # OUTPUT VIDEO
                        # ==================================

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


                        # ==================================
                        # DEFECTS
                        # ==================================

                        all_defects = {}


                        # ==================================
                        # LAST KNOWN BOX
                        #
                        # Once detected, it stays until
                        # video ends.
                        # ==================================

                        last_boxes = []


                        # ==================================
                        # PROCESS EVERY FRAME
                        # ==================================

                        while True:

                            ret, frame = (
                                cap.read()
                            )


                            if not ret:

                                break


                            # IMPORTANT:
                            # NO model.track()
                            #
                            # This avoids the tap/tracker
                            # error.

                            result = model.predict(
                                source=frame,
                                conf=0.10,
                                verbose=False
                            )[0]


                            current_boxes = []


                            # =================================
                            # GET DETECTIONS
                            # =================================

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


                                    defect_name = (
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


                                    # Save highest confidence

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


                            # =================================
                            # PERSIST BOX UNTIL VIDEO END
                            # =================================

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


                            # =================================
                            # DRAW
                            # =================================

                            processed_frame = (
                                frame.copy()
                            )


                            for detection in (
                                boxes_to_draw
                            ):

                                x1, y1, x2, y2 = (
                                    detection["box"]
                                )


                                name = (
                                    detection["name"]
                                )


                                confidence = (
                                    detection["confidence"]
                                )


                                # RED BOX

                                cv2.rectangle(
                                    processed_frame,

                                    (x1, y1),

                                    (x2, y2),

                                    (0, 0, 255),

                                    6
                                )


                                # LABEL

                                label = (
                                    f"{name} "
                                    f"{confidence * 100:.1f}%"
                                )


                                font = (
                                    cv2
                                    .FONT_HERSHEY_SIMPLEX
                                )


                                font_scale = 0.75

                                thickness = 2


                                (
                                    text_size,
                                    baseline
                                ) = cv2.getTextSize(
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


                                # Label background

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


                                # Label text

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


                            # Write frame

                            writer.write(
                                processed_frame
                            )


                        # ==================================
                        # RELEASE
                        # ==================================

                        cap.release()

                        writer.release()


                        # ==================================
                        # BROWSER COMPATIBLE
                        # ==================================

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


                        st.success(
                            "✅ Video analysis completed."
                        )


                        st.rerun()


    # ========================================================
    # OUTPUT
    # ========================================================

    with right:

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

            st.write(
                "**ANALYZED IMAGE**"
            )


            result_image = (
                resize_for_display(
                    st.session_state.image_output,
                    400,
                    240
                )
            )


            st.markdown(
                '<div class="output-media-box">',
                unsafe_allow_html=True
            )


            st.image(
                result_image,
                width=400
            )


            st.markdown(
                '</div>',
                unsafe_allow_html=True
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

            st.write(
                "**ANALYZED VIDEO**"
            )


            # EXACT SMALL OUTPUT VIDEO

            show_video(
                st.session_state.video_output,
                400,
                240
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
        # WAITING
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
