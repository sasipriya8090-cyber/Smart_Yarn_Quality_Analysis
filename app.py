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
# MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "best.pt"),
    os.path.join(BASE_DIR, "best (6).pt"),
    os.path.join(BASE_DIR, "model", "best.pt"),
    os.path.join(BASE_DIR, "trained_model", "weights", "best.pt"),
    os.path.join(BASE_DIR, "weights", "best.pt"),
]


def find_model():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return p

    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith(".pt"):
                return os.path.join(root, f)

    return None


_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


@st.cache_resource
def load_model():
    model_path = find_model()

    if model_path is None:
        st.error("❌ best.pt not found.")
        st.stop()

    return YOLO(model_path)


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
    st.session_state.video_defects = []


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
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
    padding-top: 0.8rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 1500px !important;
}


/* =========================================================
   MAIN TITLE
   ========================================================= */

.yarnx-title {
    width: 100%;
    height: 72px;

    display: flex;
    align-items: center;
    justify-content: center;

    box-sizing: border-box;

    border: 2px solid #6a1b9a;
    border-radius: 16px;

    background: linear-gradient(
        90deg,
        #eee7f8 0%,
        #eaf2ff 50%,
        #f8eaf2 100%
    );

    color: #402080;

    font-size: 30px;
    font-weight: 800;

    margin: 0 0 16px 0;

    box-shadow:
        0 5px 16px rgba(82, 36, 145, 0.15);

    white-space: nowrap;
}


/* =========================================================
   SECTION HEADINGS
   ========================================================= */

.section-title {
    font-size: 27px;
    font-weight: 800;
    color: #263238;

    margin-top: 3px;
    margin-bottom: 12px;
}


/* =========================================================
   HOME PAGE CARDS
   ========================================================= */

.home-card {
    min-height: 250px;

    border: 1px solid #d8d8df;
    border-radius: 14px;

    padding: 22px;

    background: #ffffff;

    box-shadow:
        0 2px 10px rgba(0,0,0,0.04);
}

.home-left-title {
    text-align: center;

    font-size: 25px;
    font-weight: 800;

    color: #263238;

    line-height: 1.35;

    margin-top: 30px;
}

.home-project {
    text-align: center;

    font-size: 21px;
    font-weight: 800;

    color: #37474f;

    margin-top: 32px;
}

.project-title {
    font-size: 27px;
    font-weight: 800;

    color: #4a148c;

    margin-bottom: 18px;
}

.project-text {
    font-size: 15px;
    line-height: 1.65;

    color: #263238;

    margin-bottom: 12px;
}


/* =========================================================
   PREDICT BUTTON
   ========================================================= */

.predict-btn > button {
    height: 52px;

    border: none !important;
    border-radius: 13px !important;

    background: linear-gradient(
        90deg,
        #6a1b9a,
        #3949ab
    ) !important;

    color: white !important;

    font-size: 16px !important;
    font-weight: 800 !important;

    box-shadow:
        0 5px 14px rgba(76, 38, 150, 0.25);
}


/* =========================================================
   HOME INFORMATION CARDS
   ========================================================= */

.info-card {
    min-height: 185px;

    border: 1px solid #dadde5;
    border-radius: 13px;

    padding: 17px 20px;

    background: #ffffff;
}

.info-heading {
    font-size: 17px;
    font-weight: 800;

    color: #402080;

    margin-bottom: 12px;
}

.info-item {
    font-size: 15px;

    color: #263238;

    margin-bottom: 10px;
}


/* =========================================================
   BACK BUTTON
   ========================================================= */

.back-btn > button {
    height: 43px;

    border: none !important;
    border-radius: 12px !important;

    background: linear-gradient(
        90deg,
        #6a1b9a,
        #3949ab
    ) !important;

    color: white !important;

    font-size: 15px !important;
    font-weight: 800 !important;
}


/* =========================================================
   ANALYZE BUTTON
   ========================================================= */

.analyze-btn > button {
    height: 48px;

    border: none !important;
    border-radius: 12px !important;

    background: linear-gradient(
        90deg,
        #6a1b9a,
        #3949ab
    ) !important;

    color: white !important;

    font-size: 15px !important;
    font-weight: 800 !important;
}


/* =========================================================
   IMAGE DISPLAY BOX
   ========================================================= */

.media-box {
    width: 320px;
    height: 245px;

    display: flex;
    align-items: center;
    justify-content: center;

    overflow: hidden;

    border: 1px solid #cfd8dc;
    border-radius: 12px;

    background: #f7f9fc;

    margin-top: 5px;
    margin-bottom: 12px;
}


/* =========================================================
   QUALITY
   ========================================================= */

.good-box {
    width: 100%;

    box-sizing: border-box;

    border: 2px solid #388e3c;
    border-radius: 12px;

    background: #eaf6ea;

    padding: 11px;

    text-align: center;

    color: #1b5e20;

    font-size: 19px;
    font-weight: 800;

    margin: 10px 0;
}

.bad-box {
    width: 100%;

    box-sizing: border-box;

    border: 2px solid #d32f2f;
    border-radius: 12px;

    background: #ffeded;

    padding: 11px;

    text-align: center;

    color: #b71c1c;

    font-size: 19px;
    font-weight: 800;

    margin: 10px 0;
}


/* =========================================================
   DEFECT CARD
   ========================================================= */

.defect-box {
    width: 100%;

    box-sizing: border-box;

    border: 1px solid #ef9a9a;
    border-radius: 10px;

    background: #fff8f8;

    padding: 10px 14px;

    margin: 7px 0;

    color: #263238;

    font-size: 15px;
}


/* =========================================================
   SMALL LABEL
   ========================================================= */

.media-label {
    font-size: 16px;
    font-weight: 800;

    color: #263238;

    margin-top: 5px;
    margin-bottom: 7px;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    margin-bottom: 8px;
}


/* =========================================================
   RADIO
   ========================================================= */

[data-testid="stRadio"] {
    margin-bottom: 5px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 900px) {

    .yarnx-title {
        height: auto;
        min-height: 65px;

        font-size: 22px;

        white-space: normal;

        text-align: center;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def get_class_name(class_id):
    try:
        return str(model.names[class_id]).lower().strip()
    except Exception:
        return "unknown"


def get_fiber_type(class_name):

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
        4
    )

    label = (
        f"PROTRUDING FIBER "
        f"{confidence * 100:.1f}%"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.58
    thickness = 2

    text_size, _ = cv2.getTextSize(
        label,
        font,
        scale,
        thickness
    )

    tw, th = text_size

    label_y = max(
        y1,
        th + 12
    )

    cv2.rectangle(
        image,
        (x1, label_y - th - 9),
        (x1 + tw + 10, label_y + 4),
        red,
        -1
    )

    cv2.putText(
        image,
        label,
        (x1 + 5, label_y - 5),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS IMAGE
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

    protruding_found = False

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

            # Loop fiber is not shown in final
            # result according to your requirement.

            if fiber_type == "LOOP FIBER":
                continue

            # Protruding fiber = defect

            if fiber_type == "PROTRUDING FIBER":

                protruding_found = True

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
                        "fiber": "PROTRUDING FIBER",
                        "confidence": confidence
                    }
                )

    quality = (
        "BAD"
        if protruding_found
        else "GOOD"
    )

    return output, defects, quality


# ============================================================
# VIDEO
# ============================================================

def convert_video_for_browser(input_path):

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

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


def process_video(input_path):

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    defects = []

    has_bad = False

    last_boxes = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )[0]

        current_boxes = []

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

                if fiber_type == "LOOP FIBER":
                    continue

                if fiber_type == "PROTRUDING FIBER":

                    has_bad = True

                    current_boxes.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence
                        )
                    )

                    defects.append(
                        {
                            "fiber":
                                "PROTRUDING FIBER",
                            "confidence":
                                confidence
                        }
                    )

        if current_boxes:
            last_boxes = current_boxes

        frame_boxes = (
            current_boxes
            if current_boxes
            else last_boxes
        )

        processed = frame.copy()

        for (
            x1,
            y1,
            x2,
            y2,
            confidence
        ) in frame_boxes:

            draw_red_box(
                processed,
                x1,
                y1,
                x2,
                y2,
                confidence
            )

        writer.write(processed)

    cap.release()
    writer.release()

    final_video = convert_video_for_browser(
        output_path
    )

    quality = (
        "BAD"
        if has_bad
        else "GOOD"
    )

    return final_video, defects, quality


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
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    show_title()

    # --------------------------------------------------------
    # MAIN HOME ROW
    # --------------------------------------------------------

    left, right = st.columns(
        [0.75, 1.45],
        gap="medium"
    )

    with left:

        st.markdown(
            """
            <div class="home-card">

                <div class="home-left-title">

                    AI Career for Women
                    <br>
                    (AICW)

                </div>

                <div class="home-project">
                    Capstone Project
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            '<div class="predict-btn">',
            unsafe_allow_html=True
        )

        predict = st.button(
            "🔍  PREDICT",
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        if predict:

            st.session_state.page = "inspection"

            st.rerun()

    with right:

        st.markdown(
            """
            <div class="home-card">

                <div class="project-title">
                    Project Description
                </div>

                <div class="project-text">

                    YarnX is an AI-powered yarn quality
                    inspection system designed to
                    automatically detect yarn fiber
                    defects using Computer Vision and
                    Deep Learning.

                </div>

                <div class="project-text">

                    The system accepts yarn images,
                    camera input and videos. A trained
                    YOLO model analyzes the yarn and
                    identifies fiber regions.

                </div>

                <div class="project-text">

                    YarnX provides a simple GOOD or BAD
                    result and highlights detected
                    protruding fiber defects with red
                    bounding boxes and confidence scores.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # INFORMATION ROW
    # --------------------------------------------------------

    team_col, email_col, guide_col = st.columns(
        [1.15, 1.25, 0.8],
        gap="medium"
    )

    with team_col:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-heading">
                    👩‍💻 TEAM MEMBERS
                </div>

                <div class="info-item">
                    1. Gutti.Pavani Devi Priya
                </div>

                <div class="info-item">
                    2. Somasani.Sasi Priya
                </div>

                <div class="info-item">
                    3. Galidevara.Rama Devi
                </div>

                <div class="info-item">
                    4. Rambala.Harshitha Sai Lakshmi
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with email_col:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-heading">
                    📧 GMAIL
                </div>

                <div class="info-item">
                    gutthipavanidevipriya@gmail.com
                </div>

                <div class="info-item">
                    Sasipriya8090@gmail.com
                </div>

                <div class="info-item">
                    ramadevi.galidevara0@gmail.com
                </div>

                <div class="info-item">
                    harshitharambala3@gmail.com
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with guide_col:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-heading">
                    👨‍🏫 GUIDE NAME
                </div>

                <div class="info-item">
                    Md. Abdul Aziz
                </div>

                <div class="info-heading"
                     style="margin-top:18px;">
                    DESIGNATION
                </div>

                <div class="info-item">
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

    st.markdown(
        '<div class="back-btn">',
        unsafe_allow_html=True
    )

    back = st.button(
        "⬅ Back"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    if back:

        st.session_state.page = "home"

        st.session_state.image_result = None
        st.session_state.image_quality = None
        st.session_state.image_defects = []

        st.session_state.video_result = None
        st.session_state.video_quality = None
        st.session_state.video_defects = []

        st.rerun()

    st.write("")

    # ========================================================
    # EQUAL TWO COLUMNS
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
            """
            <div class="section-title">
                📥 INPUT
            </div>
            """,
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
                key="image_upload"
            )

            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")

                st.markdown(
                    '<div class="media-label">INPUT IMAGE</div>',
                    unsafe_allow_html=True
                )

                # FIXED SMALL SIZE

                st.image(
                    image,
                    width=320
                )

                st.markdown(
                    '<div class="analyze-btn">',
                    unsafe_allow_html=True
                )

                analyze = st.button(
                    "🔍  Analyze Image",
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                if analyze:

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        (
                            result_image,
                            defects,
                            quality
                        ) = process_image(image)

                    st.session_state.image_result = (
                        result_image
                    )

                    st.session_state.image_quality = (
                        quality
                    )

                    st.session_state.image_defects = (
                        defects
                    )

                    st.session_state.video_result = None
                    st.session_state.video_quality = None
                    st.session_state.video_defects = []

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
                    '<div class="media-label">INPUT IMAGE</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    width=320
                )

                st.markdown(
                    '<div class="analyze-btn">',
                    unsafe_allow_html=True
                )

                analyze = st.button(
                    "🔍  Analyze Camera",
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                if analyze:

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        (
                            result_image,
                            defects,
                            quality
                        ) = process_image(image)

                    st.session_state.image_result = (
                        result_image
                    )

                    st.session_state.image_quality = (
                        quality
                    )

                    st.session_state.image_defects = (
                        defects
                    )

                    st.session_state.video_result = None
                    st.session_state.video_quality = None
                    st.session_state.video_defects = []

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

                st.markdown(
                    '<div class="media-label">INPUT VIDEO</div>',
                    unsafe_allow_html=True
                )

                # FIXED SMALL SIZE

                st.video(
                    uploaded_video,
                    width=320
                )

                st.markdown(
                    '<div class="analyze-btn">',
                    unsafe_allow_html=True
                )

                analyze = st.button(
                    "🔍  Analyze Video",
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                if analyze:

                    with st.spinner(
                        "Analyzing video..."
                    ):

                        temp_video = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
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

                    st.session_state.video_quality = (
                        quality
                    )

                    st.session_state.video_defects = (
                        defects
                    )

                    st.session_state.image_result = None
                    st.session_state.image_quality = None
                    st.session_state.image_defects = []

                    st.rerun()

    # ========================================================
    # RESULT
    # ========================================================

    with result_col:

        st.markdown(
            """
            <div class="section-title">
                🎯 RESULT
            </div>
            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # IMAGE RESULT
        # ====================================================

        if st.session_state.image_result is not None:

            st.markdown(
                '<div class="media-label">OUTPUT IMAGE</div>',
                unsafe_allow_html=True
            )

            # SAME WIDTH AS INPUT

            st.image(
                st.session_state.image_result,
                width=320
            )

            quality = (
                st.session_state.image_quality
            )

            # ------------------------------------------------
            # GOOD
            # ------------------------------------------------

            if quality == "GOOD":

                st.markdown(
                    """
                    <div class="good-box">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    "No defect detected."
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
                    "### 🔴 Detected Defect",
                    unsafe_allow_html=True
                )

                shown = set()

                for defect in (
                    st.session_state.image_defects
                ):

                    fiber = defect["fiber"]

                    confidence = (
                        defect["confidence"]
                    )

                    key = (
                        fiber,
                        round(confidence, 2)
                    )

                    if key in shown:
                        continue

                    shown.add(key)

                    st.markdown(
                        f"""
                        <div class="defect-box">

                            🔴
                            <b>{fiber}</b>

                            <br>

                            Confidence:
                            <b>
                            {confidence * 100:.2f}%
                            </b>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # ====================================================
        # VIDEO RESULT
        # ====================================================

        elif st.session_state.video_result is not None:

            st.markdown(
                '<div class="media-label">OUTPUT VIDEO</div>',
                unsafe_allow_html=True
            )

            st.video(
                st.session_state.video_result,
                width=320
            )

            quality = (
                st.session_state.video_quality
            )

            # ------------------------------------------------
            # GOOD
            # ------------------------------------------------

            if quality == "GOOD":

                st.markdown(
                    """
                    <div class="good-box">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    "No defect detected."
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
                    "### 🔴 Detected Defect",
                    unsafe_allow_html=True
                )

                if st.session_state.video_defects:

                    highest = max(
                        d["confidence"]
                        for d in
                        st.session_state.video_defects
                    )

                    st.markdown(
                        f"""
                        <div class="defect-box">

                            🔴
                            <b>PROTRUDING FIBER</b>

                            <br>

                            Confidence:
                            <b>
                            {highest * 100:.2f}%
                            </b>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # ====================================================
        # NO RESULT YET
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
