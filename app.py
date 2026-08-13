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
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 1rem;
}


/* ============================================================
   MAIN TITLE
   ============================================================ */

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


/* ============================================================
   VIOLET + BLUE BUTTONS
   ============================================================ */

.stButton > button {

    width: 100%;
    min-height: 46px;

    border: 0 !important;
    border-radius: 12px !important;

    font-weight: 800 !important;
    font-size: 16px !important;

    color: white !important;

    background: linear-gradient(
        135deg,
        #6a1b9a 0%,
        #3949ab 50%,
        #1976d2 100%
    ) !important;

    box-shadow:
        0 5px 14px rgba(70, 55, 150, 0.22) !important;

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
    border: 0 !important;

    box-shadow:
        0 5px 14px rgba(70, 55, 150, 0.24) !important;
}


/* ============================================================
   INFORMATION CARDS
   ============================================================ */

.info-card {

    min-height: 220px;

    padding: 20px 22px;

    border: 1px solid #d7d9e8;
    border-radius: 16px;

    background: linear-gradient(
        145deg,
        #ffffff 0%,
        #f8f9ff 100%
    );

    box-shadow:
        0 5px 18px rgba(60, 50, 120, 0.08);

    box-sizing: border-box;
}


.info-card h3 {

    margin: 0 0 16px 0;

    color: #32146f;

    font-size: 20px;
    font-weight: 800;
}


.team-item {

    margin: 9px 0;

    color: #24324a;

    font-size: 15px;

    line-height: 1.45;
}


.email-item {

    margin: 9px 0;

    font-size: 14px;

    line-height: 1.45;

    word-break: break-word;
}


.email-item a {

    color: #315bb5;

    text-decoration: none;
}


.email-item a:hover {

    text-decoration: underline;
}


.guide-label {

    margin-top: 14px;
    margin-bottom: 5px;

    color: #5b5f73;

    font-size: 13px;
    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 0.5px;
}


.guide-value {

    color: #24324a;

    font-size: 15px;

    line-height: 1.5;
}


/* ============================================================
   QUALITY
   ============================================================ */

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


/* ============================================================
   DEFECT CARD
   ============================================================ */

.defect-card {

    border: 1px solid #ef9a9a;

    border-radius: 9px;

    padding: 7px 10px;

    margin-top: 5px;

    background: #fff8f8;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FIXED IMAGE DISPLAY
# ============================================================

def show_fixed_image(
    image,
    width=400,
    height=240,
    border_color="#90caf9",
    background="#f5faff"
):

    if isinstance(image, np.ndarray):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(image)

    image = image.copy()

    image.thumbnail(
        (
            width - 12,
            height - 12
        ),
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

    html = f"""
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
        box-sizing:border-box;
    ">

        <img
            src="data:image/png;base64,{encoded}"
            style="
                max-width:{width - 12}px;
                max-height:{height - 12}px;
                width:auto;
                height:auto;
                object-fit:contain;
                display:block;
            "
        >

    </div>
    """

    st.components.v1.html(
        html,
        height=height + 30
    )


# ============================================================
# FIXED VIDEO DISPLAY
# ============================================================

def show_fixed_video(
    video_path,
    width=400,
    height=240,
    border_color="#90caf9",
    background="#f5faff"
):

    try:

        with open(
            video_path,
            "rb"
        ) as f:

            video_bytes = f.read()

        encoded = base64.b64encode(
            video_bytes
        ).decode("utf-8")

        html = f"""
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
        ">

            <video
                controls
                style="
                    width:{width - 4}px;
                    max-width:{width - 4}px;
                    max-height:{height - 4}px;
                    height:auto;
                    object-fit:contain;
                "
            >

                <source
                    src="data:video/mp4;base64,{encoded}"
                    type="video/mp4"
                >

            </video>

        </div>
        """

        st.components.v1.html(
            html,
            height=height + 30
        )

    except Exception as e:

        st.error(
            f"❌ Unable to display video: {e}"
        )


# ============================================================
# DRAW YOLO BOXES
# ============================================================

def draw_yolo_boxes(
    frame,
    result
):

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

        defects.append({
            "name": defect_name,
            "confidence": confidence,
            "box": (
                x1,
                y1,
                x2,
                y2
            )
        })


        # RED BOUNDING BOX

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            6
        )


        # LABEL

        label = (
            f"{defect_name} "
            f"{confidence * 100:.1f}%"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.75

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


        # LABEL BACKGROUND

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


        # LABEL TEXT

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

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown("""
    <div class="main-title">
        🧶 YarnX – The Future of Yarn Inspection
    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # MAIN CONTENT
    # --------------------------------------------------------

    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # --------------------------------------------------------
    # LEFT PROJECT CARD
    # --------------------------------------------------------

    with left:

        with st.container(border=True):

            st.markdown("""
            <h2 style="text-align:center;">
                AI Career for Women
                <br>
                (AICW)
            </h2>
            """, unsafe_allow_html=True)

            st.markdown("""
            <h3 style="text-align:center;">
                Capstone Project
            </h3>
            """, unsafe_allow_html=True)

            st.write("")
            st.write("")
            st.write("")
            st.write("")


        st.write("")


        # PREDICT BUTTON

        if st.button(
            "PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "inspection"

            st.rerun()


    # --------------------------------------------------------
    # PROJECT DESCRIPTION
    # --------------------------------------------------------

    with right:

        with st.container(border=True):

            st.subheader(
                "Project Description"
            )


            st.write("""
            YarnX is an AI-powered yarn quality inspection
            system designed to automatically detect and
            identify yarn defects using Computer Vision
            and Deep Learning.
            """)


            st.write("""
            The system accepts yarn images, camera input,
            and videos for inspection. A trained YOLO model
            analyzes the yarn and identifies defective
            regions by drawing bounding boxes around
            detected defects.
            """)


            st.write("""
            The system displays the detected defect,
            confidence score, and final quality result as
            GOOD or BAD. This helps reduce manual inspection
            effort and supports faster and more accurate
            yarn quality assessment.
            """)


    # ========================================================
    # TEAM / GMAIL / GUIDE
    # ========================================================

    st.write("")


    team, gmail, guide = st.columns(
        [1.35, 1.25, 0.9],
        gap="medium"
    )


    # --------------------------------------------------------
    # TEAM MEMBERS
    # --------------------------------------------------------

    with team:

        st.markdown("""
        <div class="info-card">

            <h3>👥 TEAM MEMBERS</h3>

            <div class="team-item">
                1. Gutti.Pavani Devi Priya
            </div>

            <div class="team-item">
                2. Somasani.Sasi Priya
            </div>

            <div class="team-item">
                3. Galidevara.Rama Devi
            </div>

            <div class="team-item">
                4. Rambala.Harshitha Sai Lakshmi
            </div>

        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # GMAIL
    # --------------------------------------------------------

    with gmail:

        st.markdown("""
        <div class="info-card">

            <h3>✉️ GMAIL</h3>

            <div class="email-item">
                <a href="mailto:gutthipavanidevipriya@gmail.com">
                    gutthipavanidevipriya@gmail.com
                </a>
            </div>

            <div class="email-item">
                <a href="mailto:Sasipriya8090@gmail.com">
                    Sasipriya8090@gmail.com
                </a>
            </div>

            <div class="email-item">
                <a href="mailto:ramadevigalidevara0@gmail.com">
                    ramadevigalidevara0@gmail.com
                </a>
            </div>

            <div class="email-item">
                <a href="mailto:harshitharambala3@gmail.com">
                    harshitharambala3@gmail.com
                </a>
            </div>

        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # GUIDE
    # --------------------------------------------------------

    with guide:

        st.markdown("""
        <div class="info-card">

            <h3>🎓 GUIDE</h3>

            <div class="guide-label">
                Guide Name
            </div>

            <div class="guide-value">
                Md. Abdul Aziz
            </div>

            <div class="guide-label">
                Designation
            </div>

            <div class="guide-value">
                Co Lead &amp; Trainer AICW
            </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# SECOND PAGE
# ============================================================

else:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown("""
    <div class="main-title">
        🧶 YarnX – The Future of Yarn Inspection
    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # BACK BUTTON
    # --------------------------------------------------------

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = "home"

        st.session_state.image_output = None
        st.session_state.image_defects = []

        st.session_state.video_output = None
        st.session_state.video_defects = {}

        st.rerun()


    # ========================================================
    # EQUAL INPUT + OUTPUT COLUMNS
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


                st.write(
                    "**INPUT PREVIEW**"
                )


                show_fixed_image(
                    image,

                    width=400,
                    height=240,

                    border_color="#90caf9",
                    background="#f5faff"
                )


                # ANALYZE IMAGE BUTTON

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


                st.write(
                    "**CAMERA PREVIEW**"
                )


                show_fixed_image(
                    image,

                    width=400,
                    height=240,

                    border_color="#90caf9",
                    background="#f5faff"
                )


                # ANALYZE CAMERA BUTTON

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

                # ------------------------------------------------
                # PREVIEW VIDEO
                # ------------------------------------------------

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


                show_fixed_video(
                    preview_file.name,

                    width=400,
                    height=240,

                    border_color="#90caf9",
                    background="#f5faff"
                )


                # ------------------------------------------------
                # ANALYZE VIDEO BUTTON
                # ------------------------------------------------

                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        # ----------------------------------------
                        # SAVE INPUT VIDEO
                        # ----------------------------------------

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


                        # ----------------------------------------
                        # OPEN VIDEO
                        # ----------------------------------------

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


                        # ----------------------------------------
                        # OUTPUT VIDEO
                        # ----------------------------------------

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


                        # ----------------------------------------
                        # DEFECT STORAGE
                        # ----------------------------------------

                        all_defects = {}

                        last_boxes = []


                        # ----------------------------------------
                        # PROCESS EVERY FRAME
                        # ----------------------------------------

                        while True:

                            ret, frame = (
                                cap.read()
                            )


                            if not ret:
                                break


                            # ------------------------------------
                            # YOLO PREDICTION
                            # ------------------------------------

                            result = model.predict(
                                source=frame,
                                conf=0.10,
                                verbose=False
                            )[0]


                            current_boxes = []


                            # ------------------------------------
                            # DETECTIONS
                            # ------------------------------------

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


                                    current_boxes.append({

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
                                    })


                                    # SAVE HIGHEST CONFIDENCE

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


                            # ------------------------------------
                            # CONTINUE LAST BOX
                            # ------------------------------------

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


                            # ------------------------------------
                            # DRAW BOXES
                            # ------------------------------------

                            processed_frame = (
                                frame.copy()
                            )


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
                                    cv2.FONT_HERSHEY_SIMPLEX
                                )


                                font_scale = 0.75

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


                                # LABEL BACKGROUND

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


                                # LABEL TEXT

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


                            # ------------------------------------
                            # SAVE FRAME
                            # ------------------------------------

                            writer.write(
                                processed_frame
                            )


                        # ----------------------------------------
                        # RELEASE
                        # ----------------------------------------

                        cap.release()

                        writer.release()


                        # ----------------------------------------
                        # CONVERT TO BROWSER MP4
                        # ----------------------------------------

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
    # INSPECTION RESULT
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


            show_fixed_image(
                st.session_state.image_output,

                width=500,
                height=300,

                border_color="#ce93d8",
                background="#fcf5ff"
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


            show_fixed_video(
                st.session_state.video_output,

                width=500,
                height=300,

                border_color="#ce93d8",
                background="#fcf5ff"
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
        # WAITING MESSAGE
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
