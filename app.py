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
# MODEL FINDER
# ============================================================

MODEL_NAMES = [
    "best (6).pt",
    "best.pt",
    "Copy of best.pt"
]


def find_model():

    locations = [
        ".",
        "./model",
        "./trained_model",
        "./trained_model/weights",
        "./weights",
        "/content",
        "/content/model",
        "/content/trained_model",
        "/content/trained_model/weights"
    ]

    for location in locations:

        for name in MODEL_NAMES:

            path = os.path.join(
                location,
                name
            )

            if os.path.exists(path):
                return path

    # Recursive search
    for root, dirs, files in os.walk("."):

        for name in MODEL_NAMES:

            if name in files:

                return os.path.join(
                    root,
                    name
                )

    return None


# ============================================================
# TORCH LOAD COMPATIBILITY
# ============================================================

_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):

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
            "❌ Model file not found. "
            "Please check that best.pt or best (6).pt "
            "is present in the repository."
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

if "image_defects" not in st.session_state:
    st.session_state.image_defects = []

if "image_quality" not in st.session_state:
    st.session_state.image_quality = None

if "video_output" not in st.session_state:
    st.session_state.video_output = None

if "video_defects" not in st.session_state:
    st.session_state.video_defects = {}

if "video_quality" not in st.session_state:
    st.session_state.video_quality = None


# ============================================================
# QUALITY SETTINGS
# ============================================================

# These are heuristic values because the existing YOLO
# model has only:
#
# 0 -> loop_fiber
# 1 -> protruding_fiber
#
# It was not trained with Good/Bad classes.

BAD_CONFIDENCE = 0.50
BAD_AREA_RATIO = 0.03


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 0.8rem;
        max-width: 1500px;
    }

    /* Main title */

    .main-title {
        border: 2px solid #6a1b9a;
        border-radius: 16px;

        padding: 12px;

        text-align: center;

        font-size: 29px;
        font-weight: 800;

        color: #4a148c;

        background: linear-gradient(
            90deg,
            #ede7f6,
            #e3f2fd,
            #fce4ec
        );

        margin-bottom: 15px;
    }


    /* Buttons */

    .stButton > button {

        min-height: 45px;

        border-radius: 12px;

        border: none;

        font-size: 16px;

        font-weight: 800;

        color: white;

        background: linear-gradient(
            90deg,
            #6a1b9a,
            #8e24aa,
            #3949ab
        );

        box-shadow:
            0 5px 14px
            rgba(106, 27, 154, 0.28);

        transition: 0.2s;
    }


    .stButton > button:hover {

        color: white;

        transform: translateY(-2px);

        box-shadow:
            0 7px 18px
            rgba(106, 27, 154, 0.38);
    }


    /* Result quality */

    .good-quality {

        border: 2px solid #2e7d32;

        border-radius: 12px;

        padding: 8px;

        text-align: center;

        font-size: 19px;

        font-weight: 800;

        color: #1b5e20;

        background: #e8f5e9;

        margin-top: 8px;
    }


    .bad-quality {

        border: 2px solid #c62828;

        border-radius: 12px;

        padding: 8px;

        text-align: center;

        font-size: 19px;

        font-weight: 800;

        color: #b71c1c;

        background: #ffebee;

        margin-top: 8px;
    }


    /* Team cards */

    .team-card {

        border-radius: 14px;

        padding: 12px;

        background: #fafafa;

        border: 1px solid #dddddd;
    }


    /* Input / Result headings */

    .section-heading {

        font-size: 27px;

        font-weight: 800;

        color: #263238;

        margin-bottom: 10px;
    }


    </style>
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

    return name.upper()


# ============================================================
# QUALITY HEURISTIC
# ============================================================

def calculate_quality(
    confidence,
    area_ratio
):

    if (
        confidence >= BAD_CONFIDENCE
        or area_ratio >= BAD_AREA_RATIO
    ):

        return "BAD"

    return "GOOD"


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image):

    image_array = np.array(
        image
    )

    result = model.predict(
        source=image_array,
        conf=0.10,
        verbose=False
    )[0]

    output = image_array.copy()

    detections = []

    image_height, image_width = (
        output.shape[:2]
    )

    image_area = (
        image_height * image_width
    )

    has_bad = False


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


            box_width = max(
                0,
                x2 - x1
            )

            box_height = max(
                0,
                y2 - y1
            )


            box_area = (
                box_width
                * box_height
            )


            area_ratio = (
                box_area / image_area
                if image_area > 0
                else 0
            )


            quality = calculate_quality(
                confidence,
                area_ratio
            )


            if quality == "BAD":

                has_bad = True

                box_color = (
                    0,
                    0,
                    255
                )

            else:

                box_color = (
                    0,
                    180,
                    0
                )


            detections.append(
                {
                    "fiber_type":
                        fiber_type,

                    "quality":
                        quality,

                    "confidence":
                        confidence,

                    "box":
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                }
            )


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                box_color,
                5
            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (
                f"{fiber_type} | {quality}"
            )


            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            font_scale = 0.65

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

                box_color,

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

                (
                    255,
                    255,
                    255
                ),

                thickness,

                cv2.LINE_AA
            )


    # --------------------------------------------------------
    # FINAL QUALITY
    # --------------------------------------------------------

    if len(detections) == 0:

        overall_quality = "GOOD"

    elif has_bad:

        overall_quality = "BAD"

    else:

        overall_quality = "GOOD"


    return (
        output,
        detections,
        overall_quality
    )


# ============================================================
# BROWSER VIDEO CONVERSION
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


    output_path = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name
    )


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
# PROCESS VIDEO
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


    raw_output = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name
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


    last_boxes = []

    all_defects = {}

    has_bad = False

    has_good = False


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


                class_name = get_class_name(
                    class_id
                )


                fiber_type = get_fiber_type(
                    class_name
                )


                box_width = max(
                    0,
                    x2 - x1
                )

                box_height = max(
                    0,
                    y2 - y1
                )


                image_area = (
                    width * height
                )


                box_area = (
                    box_width
                    * box_height
                )


                area_ratio = (
                    box_area / image_area
                    if image_area > 0
                    else 0
                )


                quality = calculate_quality(
                    confidence,
                    area_ratio
                )


                if quality == "BAD":

                    has_bad = True

                    box_color = (
                        0,
                        0,
                        255
                    )

                else:

                    has_good = True

                    box_color = (
                        0,
                        180,
                        0
                    )


                current_boxes.append(
                    {
                        "box":
                            (
                                x1,
                                y1,
                                x2,
                                y2
                            ),

                        "fiber_type":
                            fiber_type,

                        "quality":
                            quality,

                        "confidence":
                            confidence,

                        "color":
                            box_color
                    }
                )


                key = (
                    f"{fiber_type} | "
                    f"{quality}"
                )


                if key not in all_defects:

                    all_defects[key] = (
                        confidence
                    )

                else:

                    all_defects[key] = max(
                        all_defects[key],
                        confidence
                    )


        # ----------------------------------------------------
        # Keep boxes visible when detection temporarily drops
        # ----------------------------------------------------

        if len(current_boxes) > 0:

            last_boxes = current_boxes


        if len(current_boxes) > 0:

            boxes_to_draw = current_boxes

        else:

            boxes_to_draw = last_boxes


        processed = frame.copy()


        # ----------------------------------------------------
        # DRAW VIDEO BOXES
        # ----------------------------------------------------

        for detection in boxes_to_draw:

            x1, y1, x2, y2 = (
                detection["box"]
            )

            fiber_type = (
                detection["fiber_type"]
            )

            quality = (
                detection["quality"]
            )

            box_color = (
                detection["color"]
            )


            cv2.rectangle(
                processed,

                (x1, y1),

                (x2, y2),

                box_color,

                5
            )


            label = (
                f"{fiber_type} | {quality}"
            )


            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            font_scale = 0.65

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
                text_height + 15
            )


            cv2.rectangle(
                processed,

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

                box_color,

                -1
            )


            cv2.putText(
                processed,

                label,

                (
                    x1 + 6,
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


        writer.write(
            processed
        )


    cap.release()

    writer.release()


    # --------------------------------------------------------
    # FINAL VIDEO QUALITY
    # --------------------------------------------------------

    if has_bad:

        overall_quality = "BAD"

    elif has_good:

        overall_quality = "GOOD"

    else:

        overall_quality = "GOOD"


    final_video = (
        convert_video_for_browser(
            raw_output
        )
    )


    return (
        final_video,
        all_defects,
        overall_quality
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
        gap="small"
    )


    # --------------------------------------------------------
    # AICW
    # --------------------------------------------------------

    with left:

        with st.container(border=True):

            st.markdown(
                """
                <h2 style="
                    text-align:center;
                    color:#263238;
                ">
                    AI Career for Women
                    <br>
                    (AICW)
                </h2>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <h3 style="
                    text-align:center;
                    color:#37474f;
                    margin-top:30px;
                ">
                    Capstone Project
                </h3>
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


    # --------------------------------------------------------
    # PROJECT DESCRIPTION
    # --------------------------------------------------------

    with right:

        with st.container(border=True):

            st.markdown(
                """
                <h2 style="
                    color:#4a148c;
                ">
                    Project Description
                </h2>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div style="
                    line-height:1.65;
                    font-size:15px;
                    color:#263238;
                ">

                <p>
                YarnX is an AI-powered yarn quality inspection
                system designed to automatically detect and
                identify yarn fiber defects using Computer
                Vision and Deep Learning.
                </p>

                <p>
                The system accepts yarn images, camera input,
                and videos for inspection. A trained YOLO model
                analyzes the yarn and identifies fiber regions
                by drawing bounding boxes around detected fibers.
                </p>

                <p>
                The system displays the detected fiber type,
                confidence score, and estimated quality result
                as GOOD or BAD. This helps reduce manual
                inspection effort and supports faster and more
                accurate yarn quality assessment.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # TEAM
    # --------------------------------------------------------

    st.write("")


    team_col, email_col, guide_col = st.columns(
        [1.2, 1.3, 0.9],
        gap="medium"
    )


    with team_col:

        with st.container(border=True):

            st.markdown(
                "### 👩‍💻 TEAM MEMBERS"
            )

            st.write(
                "1. **Gutti.Pavani Devi Priya**"
            )

            st.write(
                "2. **Somasani.Sasi Priya**"
            )

            st.write(
                "3. **Galidevara.Rama Devi**"
            )

            st.write(
                "4. **Rambala.Harshitha Sai Lakshmi**"
            )


    with email_col:

        with st.container(border=True):

            st.markdown(
                "### 📧 GMAIL"
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

        with st.container(border=True):

            st.markdown(
                "### 👨‍🏫 GUIDE NAME"
            )

            st.write(
                "**Md. Abdul Aziz**"
            )

            st.markdown(
                "### DESIGNATION"
            )

            st.write(
                "**Co Lead & Trainer AICW**"
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


    if st.button("⬅ Back"):

        st.session_state.page = "home"

        st.session_state.image_result = None
        st.session_state.image_defects = []
        st.session_state.image_quality = None

        st.session_state.video_output = None
        st.session_state.video_defects = {}
        st.session_state.video_quality = None

        st.rerun()


    # ========================================================
    # IMPORTANT:
    # BALANCED SECOND PAGE
    #
    # Earlier:
    # [38, 62]
    #
    # Now:
    # [48, 52]
    #
    # So INPUT becomes wider and RESULT becomes slightly
    # smaller.
    # ========================================================

    left, right = st.columns(
        [48, 52],
        gap="medium"
    )


    # ========================================================
    # INPUT
    # ========================================================

    with left:

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


                # SMALL INPUT IMAGE
                st.image(
                    image,
                    width=280
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

                    st.session_state.image_defects = (
                        detections
                    )

                    st.session_state.image_quality = (
                        quality
                    )

                    st.session_state.video_output = None
                    st.session_state.video_defects = {}
                    st.session_state.video_quality = None

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
                    width=280
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

                    st.session_state.image_defects = (
                        detections
                    )

                    st.session_state.image_quality = (
                        quality
                    )

                    st.session_state.video_output = None

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
                    format="video/mp4",
                    start_time=0
                )


                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        input_file = (
                            tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".mp4"
                            )
                        )


                        input_file.write(
                            uploaded_video.getvalue()
                        )

                        input_file.close()


                        (
                            video_result,
                            defects,
                            quality
                        ) = process_video(
                            input_file.name
                        )


                    st.session_state.video_output = (
                        video_result
                    )

                    st.session_state.video_defects = (
                        defects
                    )

                    st.session_state.video_quality = (
                        quality
                    )

                    st.session_state.image_result = None
                    st.session_state.image_defects = []
                    st.session_state.image_quality = None

                    st.rerun()


    # ========================================================
    # RESULT
    # ========================================================

    with right:

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


            # SMALL OUTPUT IMAGE
            st.image(
                st.session_state.image_result,
                width=350
            )


            if (
                st.session_state.image_quality
                == "GOOD"
            ):

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


            st.write(
                "### Detected Fibers"
            )


            if len(
                st.session_state.image_defects
            ) == 0:

                st.info(
                    "No fiber detected."
                )


            for detection in (
                st.session_state.image_defects
            ):

                if (
                    detection["quality"]
                    == "GOOD"
                ):

                    st.success(
                        f"🟢 Fiber Type: "
                        f"{detection['fiber_type']} | "
                        f"Quality: GOOD | "
                        f"Confidence: "
                        f"{detection['confidence'] * 100:.1f}%"
                    )

                else:

                    st.error(
                        f"🔴 Fiber Type: "
                        f"{detection['fiber_type']} | "
                        f"Quality: BAD | "
                        f"Confidence: "
                        f"{detection['confidence'] * 100:.1f}%"
                    )


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


            # SMALL OUTPUT VIDEO
            st.video(
                st.session_state.video_output,
                format="video/mp4",
                start_time=0
            )


            if (
                st.session_state.video_quality
                == "GOOD"
            ):

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


            st.write(
                "### Detected Fibers"
            )


            if len(
                st.session_state.video_defects
            ) == 0:

                st.info(
                    "No fiber detected."
                )


            for name, confidence in (
                st.session_state.video_defects.items()
            ):

                st.write(
                    f"• **{name}** — "
                    f"{confidence * 100:.1f}%"
                )


        # ====================================================
        # INITIAL RESULT MESSAGE
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
