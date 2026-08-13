import os
import cv2
import tempfile
import subprocess
import numpy as np
import streamlit as st
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
# CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    max-width: 1450px !important;
    padding-top: 18px !important;
    padding-bottom: 10px !important;
}


/* ================= TITLE ================= */

.yarn-title {
    width: 100%;
    min-height: 68px;

    display: flex;
    align-items: center;
    justify-content: center;

    box-sizing: border-box;

    border: 2px solid #673ab7;
    border-radius: 16px;

    background: linear-gradient(
        90deg,
        #eee9fa,
        #eef2ff,
        #f8edf5
    );

    color: #392080;

    font-size: 29px;
    font-weight: 800;

    text-align: center;

    margin-bottom: 16px;

    box-shadow: 0 4px 14px rgba(80, 40, 140, 0.12);
}


/* ================= HEADINGS ================= */

.section-title {
    font-size: 27px;
    font-weight: 800;
    color: #263238;
    margin-bottom: 8px;
}

.sub-title {
    font-size: 15px;
    font-weight: 800;
    color: #263238;
    margin-bottom: 6px;
}


/* ================= BUTTONS ================= */

div.stButton > button {
    width: 100% !important;

    min-height: 44px !important;

    border: none !important;
    border-radius: 11px !important;

    background: linear-gradient(
        90deg,
        #6a1b9a,
        #3949ab
    ) !important;

    color: white !important;

    font-size: 15px !important;
    font-weight: 700 !important;

    box-shadow: 0 4px 12px rgba(70, 40, 150, 0.20);
}

div.stButton > button:hover {
    color: white !important;

    background: linear-gradient(
        90deg,
        #5e178a,
        #303f9f
    ) !important;
}


/* ================= HOME CARDS ================= */

.home-card {
    border: 1px solid #d7dce5;
    border-radius: 14px;

    background: white;

    padding: 18px;

    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

.aicw-box {
    min-height: 90px;

    display: flex;
    align-items: center;
    justify-content: center;

    text-align: center;

    border-radius: 10px;

    background: #f7f8fb;

    color: #263238;

    font-size: 17px;
    font-weight: 600;
}

.capstone-box {
    min-height: 52px;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-top: 20px;

    border-radius: 10px;

    background: #f7f8fb;

    color: #263238;

    font-size: 16px;
    font-weight: 600;
}

.project-heading {
    font-size: 27px;
    font-weight: 800;

    color: #4a2390;

    margin-bottom: 12px;
}

.project-text {
    font-size: 15px;
    line-height: 1.45;

    color: #263238;

    margin-bottom: 9px;
}


/* ================= INFO CARDS ================= */

.info-card {
    border: 1px solid #d8dce5;
    border-radius: 12px;

    background: white;

    padding: 14px;

    min-height: 145px;
}

.info-heading {
    color: #4a2390;

    font-size: 15px;
    font-weight: 800;

    margin-bottom: 8px;
}

.info-text {
    color: #263238;

    font-size: 13px;

    margin-bottom: 5px;
}


/* ================= QUALITY ================= */

.good-box {
    width: 100%;

    box-sizing: border-box;

    padding: 11px;

    margin-top: 9px;

    border: 2px solid #4caf50;
    border-radius: 11px;

    background: #edf8ed;

    color: #1b5e20;

    text-align: center;

    font-size: 18px;
    font-weight: 800;
}

.bad-box {
    width: 100%;

    box-sizing: border-box;

    padding: 11px;

    margin-top: 9px;

    border: 2px solid #e53935;
    border-radius: 11px;

    background: #fff0f0;

    color: #b71c1c;

    text-align: center;

    font-size: 18px;
    font-weight: 800;
}


/* ================= DEFECT ================= */

.defect-box {
    border: 1px solid #ef9a9a;

    border-radius: 10px;

    background: #fff7f7;

    padding: 12px;

    margin-top: 8px;

    color: #263238;

    font-size: 15px;
}


/* ================= MEDIA BOX ================= */

.media-box {
    width: 100%;

    height: 235px;

    display: flex;

    align-items: center;

    justify-content: center;

    overflow: hidden;

    border: 1px solid #d5dae2;

    border-radius: 10px;

    background: #f4f6f9;

    margin-bottom: 8px;
}


/* ================= RESPONSIVE ================= */

@media (max-width: 900px) {

    .yarn-title {
        font-size: 22px;
        min-height: 60px;
    }

    .section-title {
        font-size: 23px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

def show_title():

    st.markdown(
        """
        <div class="yarn-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FIND MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def find_model():

    model_names = [
        "best.pt",
        "best (6).pt",
        "best (5).pt",
        "best (4).pt",
        "best (3).pt",
        "best (2).pt",
        "best (1).pt"
    ]

    # Check current folder first
    for name in model_names:

        path = os.path.join(
            BASE_DIR,
            name
        )

        if os.path.isfile(path):
            return path

    # Check common folders
    folders = [
        os.path.join(
            BASE_DIR,
            "model"
        ),

        os.path.join(
            BASE_DIR,
            "trained_model",
            "weights"
        ),

        os.path.join(
            BASE_DIR,
            "weights"
        )
    ]

    for folder in folders:

        for name in model_names:

            path = os.path.join(
                folder,
                name
            )

            if os.path.isfile(path):
                return path

    # Search recursively
    for root, dirs, files in os.walk(
        BASE_DIR
    ):

        for file in files:

            if file.lower().endswith(".pt"):

                return os.path.join(
                    root,
                    file
                )

    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = find_model()

    if model_path is None:

        st.error(
            "❌ best.pt model not found."
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

if "image_input" not in st.session_state:
    st.session_state.image_input = None

if "image_output" not in st.session_state:
    st.session_state.image_output = None

if "image_quality" not in st.session_state:
    st.session_state.image_quality = None

if "image_defects" not in st.session_state:
    st.session_state.image_defects = []

if "video_input" not in st.session_state:
    st.session_state.video_input = None

if "video_output" not in st.session_state:
    st.session_state.video_output = None

if "video_quality" not in st.session_state:
    st.session_state.video_quality = None

if "video_defects" not in st.session_state:
    st.session_state.video_defects = []


# ============================================================
# CLASS NAME
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

def get_fiber_type(name):

    name = (
        name
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

    red = (0, 0, 255)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        red,
        3
    )

    label = (
        f"PROTRUDING FIBER "
        f"{confidence * 100:.1f}%"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.55

    thickness = 2

    text_size, _ = cv2.getTextSize(
        label,
        font,
        scale,
        thickness
    )

    text_w = text_size[0]
    text_h = text_size[1]

    top = max(
        y1,
        text_h + 8
    )

    cv2.rectangle(
        image,

        (
            x1,
            top - text_h - 8
        ),

        (
            x1 + text_w + 8,
            top + 3
        ),

        red,

        -1
    )

    cv2.putText(
        image,

        label,

        (
            x1 + 4,
            top - 4
        ),

        font,

        scale,

        (255, 255, 255),

        thickness,

        cv2.LINE_AA
    )


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(image):

    image_np = np.array(
        image
    )

    results = model.predict(
        source=image_np,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    output = image_np.copy()

    defects = []

    if result.boxes is not None:

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

            # --------------------------------------------
            # LOOP FIBER
            # --------------------------------------------
            # Loop fiber is NOT shown as defect.
            # --------------------------------------------

            if fiber_type == "LOOP FIBER":

                continue

            # --------------------------------------------
            # PROTRUDING FIBER
            # --------------------------------------------

            if fiber_type == "PROTRUDING FIBER":

                draw_red_box(
                    output,
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence
                )

                defects.append(
                    {
                        "type":
                            "PROTRUDING FIBER",

                        "confidence":
                            confidence
                    }
                )

    if len(defects) > 0:

        quality = "BAD"

    else:

        quality = "GOOD"

    return (
        output,
        quality,
        defects
    )


# ============================================================
# CONVERT VIDEO TO BROWSER FRIENDLY MP4
# ============================================================

def convert_video_for_browser(
    input_video
):

    output_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_video
    ]

    try:

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        return output_video

    except Exception:

        return input_video


# ============================================================
# ANALYZE VIDEO
# ============================================================

def analyze_video(
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

    temp_output = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    writer = cv2.VideoWriter(
        temp_output,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height
        )
    )

    all_defects = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        processed = frame.copy()

        if result.boxes is not None:

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

                # Ignore loop fiber
                if fiber_type == "LOOP FIBER":
                    continue

                # Detect protruding fiber
                if fiber_type == "PROTRUDING FIBER":

                    draw_red_box(
                        processed,
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence
                    )

                    all_defects.append(
                        {
                            "type":
                                "PROTRUDING FIBER",

                            "confidence":
                                confidence
                        }
                    )

        writer.write(
            processed
        )

    cap.release()
    writer.release()

    if len(all_defects) > 0:

        quality = "BAD"

    else:

        quality = "GOOD"

    browser_video = (
        convert_video_for_browser(
            temp_output
        )
    )

    return (
        browser_video,
        quality,
        all_defects
    )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    show_title()

    left, right = st.columns(
        [0.85, 1.45],
        gap="medium"
    )

    # --------------------------------------------------------
    # LEFT CARD
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="home-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="aicw-box">
                AI Career for Women<br>
                (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="capstone-box">
                Capstone Project
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🔍  PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "inspection"

            st.rerun()

    # --------------------------------------------------------
    # PROJECT DESCRIPTION
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="home-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="project-heading">
                Project Description
            </div>

            <div class="project-text">
                YarnX is an AI-powered yarn quality
                inspection system designed to
                automatically detect yarn fiber defects
                using Computer Vision and Deep Learning.
            </div>

            <div class="project-text">
                The system accepts yarn images,
                camera input and videos. A trained YOLO
                model analyzes the yarn and identifies
                fiber regions.
            </div>

            <div class="project-text">
                YarnX provides a simple GOOD or BAD
                result and highlights detected
                protruding fiber defects with red
                bounding boxes and confidence scores.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # INFORMATION CARDS
    # --------------------------------------------------------

    team, gmail, guide = st.columns(
        [1.15, 1.25, 0.85],
        gap="medium"
    )

    with team:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-heading">
                    👩‍💻 TEAM MEMBERS
                </div>

                <div class="info-text">
                    1. Gutti.Pavani Devi Priya
                </div>

                <div class="info-text">
                    2. Somasani.Sasi Priya
                </div>

                <div class="info-text">
                    3. Galidevara.Rama Devi
                </div>

                <div class="info-text">
                    4. Rambala.Harshitha Sai Lakshmi
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with gmail:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-heading">
                    📧 GMAIL
                </div>

                <div class="info-text">
                    gutthipavanidevipriya@gmail.com
                </div>

                <div class="info-text">
                    Sasipriya8090@gmail.com
                </div>

                <div class="info-text">
                    ramadevi.galidevara0@gmail.com
                </div>

                <div class="info-text">
                    harshitharambala3@gmail.com
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with guide:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-heading">
                    👨‍🏫 GUIDE NAME
                </div>

                <div class="info-text">
                    Md. Abdul Aziz
                </div>

                <div class="info-heading">
                    DESIGNATION
                </div>

                <div class="info-text">
                    Co Lead & Trainer AICW
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# INSPECTION PAGE
# ============================================================

else:

    show_title()

    # --------------------------------------------------------
    # BACK
    # --------------------------------------------------------

    if st.button(
        "⬅  Back",
        use_container_width=False
    ):

        st.session_state.page = "home"

        st.session_state.image_input = None
        st.session_state.image_output = None
        st.session_state.image_quality = None
        st.session_state.image_defects = []

        st.session_state.video_input = None
        st.session_state.video_output = None
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
            '<div class="section-title">📥 INPUT</div>',
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
                key="upload_image"
            )

            if uploaded_image is not None:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")

                st.session_state.image_input = image

                st.markdown(
                    '<div class="sub-title">INPUT IMAGE</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    width=320
                )

                if st.button(
                    "🔍  Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        (
                            output,
                            quality,
                            defects
                        ) = analyze_image(
                            image
                        )

                    st.session_state.image_output = output

                    st.session_state.image_quality = quality

                    st.session_state.image_defects = defects

                    st.rerun()

        # ====================================================
        # CAMERA
        # ====================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Capture Yarn"
            )

            if camera_image is not None:

                image = Image.open(
                    camera_image
                ).convert("RGB")

                st.session_state.image_input = image

                st.markdown(
                    '<div class="sub-title">INPUT IMAGE</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    width=320
                )

                if st.button(
                    "🔍  Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        (
                            output,
                            quality,
                            defects
                        ) = analyze_image(
                            image
                        )

                    st.session_state.image_output = output

                    st.session_state.image_quality = quality

                    st.session_state.image_defects = defects

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
                key="upload_video"
            )

            if uploaded_video is not None:

                video_bytes = (
                    uploaded_video.getvalue()
                )

                st.session_state.video_input = video_bytes

                st.markdown(
                    '<div class="sub-title">INPUT VIDEO</div>',
                    unsafe_allow_html=True
                )

                st.video(
                    video_bytes
                )

                if st.button(
                    "🔍  Analyze Video",
                    use_container_width=True
                ):

                    input_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    )

                    input_file.write(
                        video_bytes
                    )

                    input_file.close()

                    with st.spinner(
                        "Analyzing video..."
                    ):

                        (
                            output_video,
                            quality,
                            defects
                        ) = analyze_video(
                            input_file.name
                        )

                    st.session_state.video_output = output_video

                    st.session_state.video_quality = quality

                    st.session_state.video_defects = defects

                    st.rerun()


    # ========================================================
    # RESULT
    # ========================================================

    with result_col:

        st.markdown(
            '<div class="section-title">🎯 RESULT</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # IMAGE RESULT
        # ====================================================

        if st.session_state.image_output is not None:

            st.markdown(
                '<div class="sub-title">OUTPUT IMAGE</div>',
                unsafe_allow_html=True
            )

            st.image(
                st.session_state.image_output,
                width=320
            )

            # ------------------------------------------------
            # GOOD
            # ------------------------------------------------

            if (
                st.session_state.image_quality
                == "GOOD"
            ):

                st.markdown(
                    """
                    <div class="good-box">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # BAD
            # ------------------------------------------------

            else:

                st.markdown(
                    """
                    <div class="bad-box">
                        🔴 BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### 🔴 Detected Defect"
                )

                # Show every detected protruding fiber
                for i, defect in enumerate(
                    st.session_state.image_defects,
                    start=1
                ):

                    confidence = (
                        defect["confidence"]
                    )

                    st.markdown(
                        f"""
                        <div class="defect-box">
                            🔴 <b>PROTRUDING FIBER</b>
                            <br><br>
                            Confidence:
                            <b>{confidence * 100:.2f}%</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # ====================================================
        # VIDEO RESULT
        # ====================================================

        elif st.session_state.video_output is not None:

            st.markdown(
                '<div class="sub-title">OUTPUT VIDEO</div>',
                unsafe_allow_html=True
            )

            st.video(
                st.session_state.video_output
            )

            # ------------------------------------------------
            # GOOD
            # ------------------------------------------------

            if (
                st.session_state.video_quality
                == "GOOD"
            ):

                st.markdown(
                    """
                    <div class="good-box">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # BAD
            # ------------------------------------------------

            else:

                st.markdown(
                    """
                    <div class="bad-box">
                        🔴 BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### 🔴 Detected Defect"
                )

                if st.session_state.video_defects:

                    max_confidence = max(
                        d["confidence"]
                        for d in
                        st.session_state.video_defects
                    )

                    st.markdown(
                        f"""
                        <div class="defect-box">
                            🔴 <b>PROTRUDING FIBER</b>
                            <br><br>
                            Confidence:
                            <b>
                            {max_confidence * 100:.2f}%
                            </b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # ====================================================
        # INITIAL RESULT
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
