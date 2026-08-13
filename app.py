import os
import tempfile
import subprocess
import base64
import textwrap
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
# HTML HELPER
# ============================================================

def render_html(html):
    st.markdown(
        textwrap.dedent(html).strip(),
        unsafe_allow_html=True
    )


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   COLORFUL BACKGROUND
   ============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 8% 12%,
            rgba(186, 104, 200, 0.20) 0%,
            transparent 27%
        ),
        radial-gradient(
            circle at 92% 15%,
            rgba(66, 165, 245, 0.20) 0%,
            transparent 29%
        ),
        radial-gradient(
            circle at 18% 88%,
            rgba(77, 182, 172, 0.16) 0%,
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 88%,
            rgba(244, 143, 177, 0.18) 0%,
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f7efff 0%,
            #edf7ff 45%,
            #fff1f7 100%
        );

    min-height: 100vh;
}


/* ============================================================
   PAGE SIZE
   ============================================================ */

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
}


/* ============================================================
   MAIN TITLE
   ============================================================ */

.main-title {
    width: 100%;
    height: 72px;

    box-sizing: border-box;

    border: 2px solid #6a1b9a;
    border-radius: 14px;

    display: flex;
    align-items: center;
    justify-content: center;

    padding: 0 20px;

    margin-top: 0;
    margin-bottom: 20px;

    transform: translateY(18px);

    overflow: hidden;

    text-align: center;

    font-size: 28px;
    font-weight: 800;

    color: #43218a;

    background:
        linear-gradient(
            90deg,
            rgba(243,229,245,0.96),
            rgba(227,242,253,0.96),
            rgba(252,228,236,0.96)
        );

    box-shadow:
        0 5px 16px rgba(83,52,120,0.12);
}


/* ============================================================
   GENERAL TEXT
   ============================================================ */

.stApp,
.stApp p,
.stApp span,
.stApp label {
    color: #24324a;
}


/* ============================================================
   HEADINGS
   ============================================================ */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    font-weight: 800;
    letter-spacing: 0.2px;
    text-decoration: none !important;
}


/* ============================================================
   REMOVE HEADING UNDERLINES
   ============================================================ */

h1,
h2,
h3,
h4,
h5,
h6 {
    text-decoration: none !important;
}


/* ============================================================
   REMOVE STREAMLIT HEADING LINK ICON
   ============================================================ */

h1 a,
h2 a,
h3 a,
h4 a,
h5 a,
h6 a {
    display: none !important;
}


/* ============================================================
   CARD STYLE
   ============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {

    border-radius: 16px !important;

    border: 1px solid rgba(120,100,160,0.22) !important;

    background:
        rgba(255,255,255,0.50) !important;

    box-shadow:
        0 5px 18px rgba(74,55,120,0.08) !important;

    backdrop-filter: blur(7px);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}


div[data-testid="stVerticalBlockBorderWrapper"]:hover {

    transform: translateY(-2px);

    box-shadow:
        0 9px 24px rgba(74,55,120,0.12) !important;
}


/* ============================================================
   CARD HEADINGS
   ============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] h2 {
    color: #43218a;
}


div[data-testid="stVerticalBlockBorderWrapper"] h3 {
    color: #263957;
}


/* ============================================================
   CARD TEXT
   ============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] p {

    color: #263952;

    line-height: 1.6;
}


/* ============================================================
   AICW SPECIAL CARD
   ============================================================ */

.aicw-card {

    width: 100%;

    min-height: 255px;

    border-radius: 16px;

    padding: 24px 22px;

    box-sizing: border-box;

    background:
        linear-gradient(
            145deg,
            rgba(246,236,255,0.96),
            rgba(231,243,255,0.90)
        );

    border:
        1px solid rgba(106,27,154,0.20);

    box-shadow:
        0 8px 24px rgba(106,27,154,0.10);

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 15px;
}


.aicw-text {

    flex: 1;

    min-width: 0;
}


.aicw-title {

    font-size: 31px;

    font-weight: 850;

    color: #243957;

    line-height: 1.15;

    margin-bottom: 20px;
}


.aicw-subtitle {

    font-size: 27px;

    font-weight: 800;

    color: #263957;

    margin-bottom: 22px;
}


.aicw-capstone {

    font-size: 22px;

    font-weight: 800;

    color: #263957;
}


/* ============================================================
   WOMAN ILLUSTRATION
   ============================================================ */

.woman-visual {

    width: 112px;

    height: 112px;

    flex-shrink: 0;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    position: relative;

    background:
        linear-gradient(
            145deg,
            #eadcff,
            #dceeff
        );

    border:
        2px solid rgba(106,27,154,0.16);

    box-shadow:
        0 8px 20px rgba(75,55,140,0.12);

    font-size: 63px;
}


/* ============================================================
   YARN BADGE
   ============================================================ */

.yarn-badge {

    position: absolute;

    right: -5px;

    bottom: 2px;

    width: 40px;

    height: 40px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #ffffff,
            #f3e5f5
        );

    box-shadow:
        0 4px 10px rgba(70,50,120,0.15);

    font-size: 22px;
}


/* ============================================================
   PROJECT DESCRIPTION
   ============================================================ */

.project-card {

    background:
        linear-gradient(
            145deg,
            rgba(237,247,255,0.94),
            rgba(246,239,255,0.90)
        );
}


/* ============================================================
   PROJECT DESCRIPTION HEADING
   ============================================================ */

.project-heading {

    color: #43218a;

    font-size: 28px;

    font-weight: 850;

    margin-bottom: 16px;
}


/* ============================================================
   PROJECT TEXT
   ============================================================ */

.project-text {

    color: #263952;

    font-size: 15px;

    line-height: 1.65;

    margin-bottom: 12px;
}


/* ============================================================
   EMAIL LINKS
   ============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] a {

    color: #315bb5 !important;

    font-weight: 600;

    text-decoration: none !important;
}


/* ============================================================
   BUTTONS - WHITE TEXT FIX
   ============================================================ */

.stButton > button,
.stButton > button *,
button[kind="secondary"],
button[kind="secondary"] *,
button[kind="primary"],
button[kind="primary"] * {

    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    font-weight: 800 !important;
}


/* ============================================================
   BUTTON DESIGN
   ============================================================ */

.stButton > button {

    width: 100% !important;

    min-height: 48px !important;

    border: none !important;

    border-radius: 13px !important;

    font-size: 16px !important;

    font-weight: 800 !important;

    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    background:
        linear-gradient(
            135deg,
            #6a1b9a 0%,
            #3949ab 50%,
            #1976d2 100%
        ) !important;

    box-shadow:
        0 6px 16px rgba(70,55,150,0.25) !important;

    transition:
        all 0.2s ease-in-out !important;
}


/* ============================================================
   BUTTON HOVER
   ============================================================ */

.stButton > button:hover {

    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    background:
        linear-gradient(
            135deg,
            #7b1fa2 0%,
            #3f51b5 50%,
            #1e88e5 100%
        ) !important;

    transform: translateY(-2px);

    box-shadow:
        0 9px 20px rgba(70,55,150,0.30) !important;
}


/* ============================================================
   BUTTON FOCUS
   ============================================================ */

.stButton > button:focus,
.stButton > button:active {

    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    border: none !important;

    outline: none !important;

    box-shadow:
        0 6px 16px rgba(70,55,150,0.25) !important;
}


/* ============================================================
   BUTTON TEXT
   ============================================================ */

.stButton > button p,
.stButton > button span,
.stButton > button div {

    color: #ffffff !important;

    -webkit-text-fill-color: #ffffff !important;

    font-weight: 800 !important;
}


/* ============================================================
   RADIO
   ============================================================ */

div[data-testid="stRadio"] label {

    font-weight: 600;

    color: #263957;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

section[data-testid="stFileUploaderDropzone"] {

    background:
        rgba(245,248,255,0.80) !important;

    border-radius: 12px !important;
}


/* ============================================================
   INFO MESSAGE
   ============================================================ */

div[data-testid="stAlert"] {

    border-radius: 12px !important;
}


/* ============================================================
   GOOD QUALITY
   ============================================================ */

.good-quality {

    border: 2px solid #2e7d32;

    border-radius: 12px;

    padding: 9px;

    text-align: center;

    font-size: 21px;

    font-weight: bold;

    color: #1b5e20;

    background:
        linear-gradient(
            135deg,
            #e8f5e9,
            #f1f8e9
        );

    margin-top: 10px;

    box-shadow:
        0 4px 12px rgba(46,125,50,0.10);
}


/* ============================================================
   BAD QUALITY
   ============================================================ */

.bad-quality {

    border: 2px solid #c62828;

    border-radius: 12px;

    padding: 9px;

    text-align: center;

    font-size: 21px;

    font-weight: bold;

    color: #b71c1c;

    background:
        linear-gradient(
            135deg,
            #ffebee,
            #fff5f5
        );

    margin-top: 10px;

    box-shadow:
        0 4px 12px rgba(198,40,40,0.10);
}


/* ============================================================
   DEFECT CARD
   ============================================================ */

.defect-card {

    border: 1px solid #ef9a9a;

    border-radius: 10px;

    padding: 10px 12px;

    margin-top: 7px;

    background:
        linear-gradient(
            135deg,
            rgba(255,248,248,0.96),
            rgba(255,242,245,0.92)
        );

    box-shadow:
        0 3px 9px rgba(180,60,80,0.06);
}


/* ============================================================
   IMAGE / VIDEO CONTAINERS
   ============================================================ */

.media-container {

    background:
        rgba(255,255,255,0.55);

    border-radius: 14px;
}


/* ============================================================
   SMOOTH UI
   ============================================================ */

* {
    box-sizing: border-box;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# IMAGE DISPLAY
# ============================================================

def show_fixed_image(
    image,
    width=400,
    height=240,
    border_color="#90caf9",
    background="#f5faff"
):

    if isinstance(image, np.ndarray):

        image = Image.fromarray(
            image.astype(np.uint8)
        )

    elif not isinstance(image, Image.Image):

        image = Image.fromarray(
            np.array(image).astype(np.uint8)
        )

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

    render_html(f"""
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
    """)

    st.write("")


# ============================================================
# VIDEO DISPLAY
# ============================================================

def show_fixed_video(
    video_path,
    width=400,
    height=240,
    border_color="#90caf9",
    background="#f5faff"
):

    try:

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        encoded = base64.b64encode(
            video_bytes
        ).decode("utf-8")

        render_html(f"""
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
        """)

        st.write("")

    except Exception as e:

        st.error(
            f"❌ Unable to display video: {e}"
        )


# ============================================================
# YOLO IMAGE BOXES
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


        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            6
        )


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

            font_scale,

            (255, 255, 255),

            thickness,

            cv2.LINE_AA
        )


    return output, defects


# ============================================================
# VIDEO CONVERSION
# ============================================================

def convert_video_for_browser(input_path):

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

    render_html("""
    <div class="main-title">
        🧶 YarnX – The Future of Yarn Inspection
    </div>
    """)


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

        render_html("""
        <div class="aicw-card">

            <div class="aicw-text">

                <div class="aicw-title">
                    AI Career for Women
                </div>

                <div class="aicw-subtitle">
                    (AICW)
                </div>

                <div class="aicw-capstone">
                    Capstone Project
                </div>

            </div>

            <div class="woman-visual">

                👩‍💻

                <div class="yarn-badge">
                    🧶
                </div>

            </div>

        </div>
        """)


        st.write("")


        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "inspection"

            st.rerun()


    # ========================================================
    # PROJECT DESCRIPTION
    # ========================================================

    with right:

        with st.container(border=True):

            render_html("""
            <div class="project-heading">
                Project Description
            </div>
            """)


            render_html("""
            <div class="project-text">
                YarnX is an AI-powered yarn quality inspection
                system designed to automatically detect and
                identify yarn defects using Computer Vision
                and Deep Learning.
            </div>
            """)


            render_html("""
            <div class="project-text">
                The system accepts yarn images, camera input,
                and videos for inspection. A trained YOLO model
                analyzes the yarn and identifies defective
                regions by drawing bounding boxes around
                detected defects.
            </div>
            """)


            render_html("""
            <div class="project-text">
                The system displays the detected defect,
                confidence score, and final quality result as
                GOOD or BAD. This helps reduce manual inspection
                effort and supports faster and more accurate
                yarn quality assessment.
            </div>
            """)


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


    # ========================================================
    # GMAIL
    # ========================================================

    with gmail:

        with st.container(border=True):

            st.markdown(
                "### 📧 GMAIL"
            )

            st.markdown(
                '<span style="color:#315bb5;">'
                'gutthipavanidevipriya@gmail.com'
                '</span>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<span style="color:#315bb5;">'
                'Sasipriya8090@gmail.com'
                '</span>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<span style="color:#315bb5;">'
                'ramadevigalidevara0@gmail.com'
                '</span>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<span style="color:#315bb5;">'
                'harshitharambala3@gmail.com'
                '</span>',
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

            st.write(
                "Md. Abdul Aziz"
            )

            st.markdown(
                "### DESIGNATION"
            )

            st.write(
                "Co Lead & Trainer AICW"
            )


# ============================================================
# SECOND PAGE
# ============================================================

else:

    render_html("""
    <div class="main-title">
        🧶 YarnX – The Future of Yarn Inspection
    </div>
    """)


    # ========================================================
    # BACK
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
    # INPUT + OUTPUT
    # ========================================================

    left, right = st.columns(
        [1, 1],
        gap="medium"
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


                show_fixed_image(
                    image,
                    width=400,
                    height=240,
                    border_color="#90caf9",
                    background="#f5faff"
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


                show_fixed_image(
                    image,
                    width=400,
                    height=240,
                    border_color="#90caf9",
                    background="#f5faff"
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


                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

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


                        input_path = input_temp.name


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


                        output_temp = (
                            tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".mp4"
                            )
                        )

                        output_temp.close()

                        raw_output = output_temp.name


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


                        # PROCESS VIDEO

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


                                    defect_name = (
                                        model.names[class_id]
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

                                confidence = (
                                    detection["confidence"]
                                )


                                cv2.rectangle(
                                    processed_frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (0, 0, 255),
                                    6
                                )


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


                        cap.release()

                        writer.release()


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
        # IMAGE RESULT
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

                render_html("""
                <div class="bad-quality">
                    ❌ BAD QUALITY
                </div>
                """)


                st.write(
                    "### Detected Defects"
                )


                for defect in defects:

                    render_html(f"""
                    <div class="defect-card">
                        🔴 <strong>Defect:</strong> {defect["name"]}
                        <span style="margin-left:24px;">
                            📊 <strong>Confidence:</strong>
                            {defect["confidence"] * 100:.2f}%
                        </span>
                    </div>
                    """)


            else:

                render_html("""
                <div class="good-quality">
                    ✅ GOOD QUALITY
                </div>
                """)


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

                render_html("""
                <div class="bad-quality">
                    ❌ BAD QUALITY
                </div>
                """)


                st.write(
                    "### Detected Defects"
                )


                for name, confidence in (
                    defects.items()
                ):

                    render_html(f"""
                    <div class="defect-card">
                        🔴 <strong>Defect:</strong> {name}
                        <span style="margin-left:24px;">
                            📊 <strong>Confidence:</strong>
                            {confidence * 100:.2f}%
                        </span>
                    </div>
                    """)


            else:

                render_html("""
                <div class="good-quality">
                    ✅ GOOD QUALITY
                </div>
                """)


        # ====================================================
        # DEFAULT
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
