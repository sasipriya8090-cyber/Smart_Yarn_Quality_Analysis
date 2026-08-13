import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE SETTINGS
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
    padding-top: 12px !important;
    padding-bottom: 10px !important;
    max-width: 1450px !important;
}


/* ==========================================================
   TITLE
   ========================================================== */

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
        #eee8fa,
        #eef2ff,
        #f8edf5
    );

    color: #392080;

    font-size: 29px;
    font-weight: 800;

    text-align: center;

    margin-bottom: 15px;

    box-shadow: 0 4px 14px rgba(80, 40, 140, 0.12);
}


/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

.section-heading {
    font-size: 27px;
    font-weight: 800;
    color: #263238;

    margin-top: 0;
    margin-bottom: 10px;
}


/* ==========================================================
   HOME CARDS
   ========================================================== */

.home-card {
    border: 1px solid #d8dce5;
    border-radius: 14px;

    background: white;

    padding: 18px;

    min-height: 235px;

    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

.aicw-box {
    height: 92px;

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
    height: 55px;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-top: 24px;

    border-radius: 10px;

    background: #f7f8fb;

    color: #263238;

    font-size: 16px;
    font-weight: 600;
}

.project-heading {
    font-size: 28px;
    font-weight: 800;

    color: #4a2390;

    margin-bottom: 14px;
}

.project-text {
    font-size: 15px;
    line-height: 1.45;

    color: #263238;

    margin-bottom: 10px;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

div.stButton > button {
    border: none !important;

    border-radius: 12px !important;

    background: linear-gradient(
        90deg,
        #6a1b9a,
        #3949ab
    ) !important;

    color: white !important;

    font-weight: 700 !important;

    height: 46px;

    box-shadow: 0 4px 12px rgba(70, 40, 150, 0.20);

    transition: 0.2s;
}

div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #5e178a,
        #303f9f
    ) !important;

    color: white !important;
}


/* ==========================================================
   INFO CARDS
   ========================================================== */

.info-card {
    border: 1px solid #d8dce5;
    border-radius: 12px;

    background: white;

    padding: 15px;

    min-height: 150px;
}

.info-heading {
    color: #4a2390;

    font-size: 16px;
    font-weight: 800;

    margin-bottom: 10px;
}

.info-text {
    color: #263238;

    font-size: 14px;

    margin-bottom: 7px;
}


/* ==========================================================
   MEDIA
   ========================================================== */

.media-title {
    font-size: 15px;

    font-weight: 800;

    color: #263238;

    margin-top: 6px;
    margin-bottom: 6px;
}


/* ==========================================================
   QUALITY BADGES
   ========================================================== */

.good-quality {
    width: 100%;

    box-sizing: border-box;

    padding: 11px;

    margin-top: 9px;
    margin-bottom: 10px;

    border: 2px solid #4caf50;

    border-radius: 11px;

    background: #edf8ed;

    color: #1b5e20;

    text-align: center;

    font-size: 18px;

    font-weight: 800;
}

.bad-quality {
    width: 100%;

    box-sizing: border-box;

    padding: 11px;

    margin-top: 9px;
    margin-bottom: 10px;

    border: 2px solid #e53935;

    border-radius: 11px;

    background: #fff0f0;

    color: #b71c1c;

    text-align: center;

    font-size: 18px;

    font-weight: 800;
}


/* ==========================================================
   DEFECT BOX
   ========================================================== */

.defect-card {
    border: 1px solid #ef9a9a;

    border-radius: 10px;

    background: #fff8f8;

    padding: 11px 14px;

    margin-top: 7px;

    color: #263238;

    font-size: 15px;
}


/* ==========================================================
   FILE UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {
    margin-top: -4px;
    margin-bottom: 4px;
}


/* ==========================================================
   RADIO
   ========================================================== */

[data-testid="stRadio"] {
    margin-bottom: 5px;
}


/* ==========================================================
   IMAGE CONTAINER
   ========================================================== */

.fixed-media {
    width: 320px;
    height: 220px;

    display: flex;
    align-items: center;
    justify-content: center;

    overflow: hidden;

    border-radius: 10px;

    background: #f4f6f9;

    border: 1px solid #d4d9e0;
}


/* ==========================================================
   SMALL SCREEN
   ========================================================== */

@media (max-width: 900px) {

    .yarn-title {
        font-size: 22px;
        min-height: 62px;
    }

    .section-heading {
        font-size: 23px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL SEARCH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_model():

    possible_models = [

        os.path.join(
            BASE_DIR,
            "best.pt"
        ),

        os.path.join(
            BASE_DIR,
            "best (6).pt"
        ),

        os.path.join(
            BASE_DIR,
            "model",
            "best.pt"
        ),

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

    for path in possible_models:

        if os.path.exists(path):
            return path

    for root, dirs, files in os.walk(BASE_DIR):

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
# DRAW RED PROTRUDING FIBER BOX
# ============================================================

def draw_protruding_box(
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

    font_scale = 0.55

    thickness = 2

    text_size, _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    text_width = text_size[0]
    text_height = text_size[1]

    label_y = max(
        y1,
        text_height + 8
    )

    cv2.rectangle(
        image,

        (
            x1,
            label_y - text_height - 8
        ),

        (
            x1 + text_width + 8,
            label_y + 3
        ),

        red,

        -1
    )

    cv2.putText(
        image,

        label,

        (
            x1 + 4,
            label_y - 4
        ),

        font,

        font_scale,

        (255, 255, 255),

        thickness,

        cv2.LINE_AA
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image):

    image_array = np.array(image)

    results = model.predict(
        source=image_array,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    output = image_array.copy()

    defects = []

    protruding_found = False

    if result.boxes is not None:

        for box in result.boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = coordinates

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

            # =================================================
            # LOOP FIBER
            # =================================================
            #
            # Loop fiber is detected internally,
            # but is NOT displayed as a defect.
            #

            if fiber_type == "LOOP FIBER":

                continue

            # =================================================
            # PROTRUDING FIBER
            # =================================================

            if fiber_type == "PROTRUDING FIBER":

                protruding_found = True

                draw_protruding_box(
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

    if protruding_found:

        quality = "BAD"

    else:

        quality = "GOOD"

    return (
        output,
        defects,
        quality
    )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(input_path):

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

    writer = cv2.VideoWriter(
        output_path,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height
        )
    )

    protruding_found = False

    defects = []

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

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                x1, y1, x2, y2 = coordinates

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

                # Protruding fiber = BAD

                if fiber_type == "PROTRUDING FIBER":

                    protruding_found = True

                    draw_protruding_box(
                        processed,

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

        writer.write(
            processed
        )

    cap.release()

    writer.release()

    if protruding_found:

        quality = "BAD"

    else:

        quality = "GOOD"

    return (
        output_path,
        defects,
        quality
    )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    show_title()

    left_col, right_col = st.columns(
        [0.8, 1.45],
        gap="medium"
    )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    with left_col:

        st.markdown(
            """
            <div class="home-card">

                <div class="aicw-box">

                    AI Career for Women
                    <br>
                    (AICW)

                </div>

                <div class="capstone-box">

                    Capstone Project

                </div>

            </div>
            """,
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
    # RIGHT
    # --------------------------------------------------------

    with right_col:

        st.markdown(
            """
            <div class="home-card">

                <div class="project-heading">
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
    # INFORMATION CARDS
    # --------------------------------------------------------

    team_col, mail_col, guide_col = st.columns(
        [1.15, 1.25, 0.85],
        gap="medium"
    )

    with team_col:

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

    with mail_col:

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

    with guide_col:

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
        "⬅  Back"
    ):

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
    # TWO EQUAL COLUMNS
    # ========================================================

    input_col, result_col = st.columns(
        [1, 1],
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
                key="image_file"
            )

            if uploaded_image is not None:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")

                st.markdown(
                    """
                    <div class="media-title">
                        INPUT IMAGE
                    </div>
                    """,
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
                            result_image,
                            defects,
                            quality
                        ) = analyze_image(
                            image
                        )

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

                    st.rerun()

        # ====================================================
        # CAMERA INPUT
        # ====================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Capture Yarn"
            )

            if camera_image is not None:

                image = Image.open(
                    camera_image
                ).convert("RGB")

                st.markdown(
                    """
                    <div class="media-title">
                        INPUT IMAGE
                    </div>
                    """,
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
                            result_image,
                            defects,
                            quality
                        ) = analyze_image(
                            image
                        )

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
                key="video_file"
            )

            if uploaded_video is not None:

                st.markdown(
                    """
                    <div class="media-title">
                        INPUT VIDEO
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.video(
                    uploaded_video
                )

                if st.button(
                    "🔍  Analyze Video",
                    use_container_width=True
                ):

                    temp_input = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    )

                    temp_input.write(
                        uploaded_video.getvalue()
                    )

                    temp_input.close()

                    with st.spinner(
                        "Analyzing video..."
                    ):

                        (
                            output_video,
                            defects,
                            quality
                        ) = analyze_video(
                            temp_input.name
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

        if st.session_state.image_result is not None:

            st.markdown(
                """
                <div class="media-title">
                    OUTPUT IMAGE
                </div>
                """,
                unsafe_allow_html=True
            )

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
                    <div class="good-quality">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    "No protruding fiber defect detected."
                )

            # ------------------------------------------------
            # BAD
            # ------------------------------------------------

            else:

                st.markdown(
                    """
                    <div class="bad-quality">
                        🔴 BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### 🔴 Detected Defect"
                )

                displayed = set()

                for defect in (
                    st.session_state.image_defects
                ):

                    defect_type = (
                        defect["type"]
                    )

                    confidence = (
                        defect["confidence"]
                    )

                    unique_key = (
                        defect_type,
                        round(
                            confidence,
                            2
                        )
                    )

                    if unique_key in displayed:
                        continue

                    displayed.add(
                        unique_key
                    )

                    st.markdown(
                        f"""
                        <div class="defect-card">

                        🔴 <b>{defect_type}</b>

                        <br><br>

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
                """
                <div class="media-title">
                    OUTPUT VIDEO
                </div>
                """,
                unsafe_allow_html=True
            )

            st.video(
                st.session_state.video_result
            )

            quality = (
                st.session_state.video_quality
            )

            # ------------------------------------------------
            # GOOD VIDEO
            # ------------------------------------------------

            if quality == "GOOD":

                st.markdown(
                    """
                    <div class="good-quality">
                        🟢 GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    "No protruding fiber defect detected."
                )

            # ------------------------------------------------
            # BAD VIDEO
            # ------------------------------------------------

            else:

                st.markdown(
                    """
                    <div class="bad-quality">
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
                        <div class="defect-card">

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
        # NO RESULT
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
