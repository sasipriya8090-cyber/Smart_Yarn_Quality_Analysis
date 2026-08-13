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


if "image_input" not in st.session_state:
    st.session_state.image_input = None


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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }


    /* ========================================================
       NORMAL YARNX TITLE
       NO BOX
       NO BORDER
       NO BACKGROUND
       ======================================================== */

    .main-title {

        width: 100%;

        text-align: center;

        font-size: 30px;

        font-weight: 800;

        color: #4a148c;

        background: transparent !important;

        border: none !important;

        border-radius: 0 !important;

        padding: 0 !important;

        margin: 0 0 18px 0 !important;

        box-shadow: none !important;

        line-height: 1.25;

        overflow: visible;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {

        height: 50px;

        border-radius: 14px;

        border: none;

        font-size: 17px;

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
            rgba(106, 27, 154, 0.30);
    }


    .stButton > button:hover {

        color: white;

        transform: translateY(-2px);

        box-shadow:
            0 8px 20px
            rgba(106, 27, 154, 0.40);
    }


    /* ========================================================
       QUALITY
       ======================================================== */

    .good-quality {

        border: 2px solid #2e7d32;

        border-radius: 12px;

        padding: 10px;

        text-align: center;

        font-size: 20px;

        font-weight: 800;

        color: #1b5e20;

        background: #e8f5e9;

        margin-top: 8px;
    }


    .bad-quality {

        border: 2px solid #c62828;

        border-radius: 12px;

        padding: 10px;

        text-align: center;

        font-size: 20px;

        font-weight: 800;

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

        padding: 8px 10px;

        margin-top: 6px;

        background: #fff8f8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CLASSIFICATION HELPERS
# ============================================================

def get_class_name(class_id):

    try:

        return str(
            model.names[class_id]
        ).lower().strip()

    except Exception:

        return "unknown"


def classify_detection(class_name):

    name = (
        class_name
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


    # loop_fiber       -> GOOD
    # protruding_fiber -> BAD

    if "protrud" in name:

        return "bad"


    if "loop" in name:

        return "good"


    return "unknown"


# ============================================================
# DRAW IMAGE RESULT
# ============================================================

def process_image(image):

    image_array = np.array(image)


    result = model.predict(
        source=image_array,
        conf=0.25,
        verbose=False
    )[0]


    # IMPORTANT:
    # Start from the ORIGINAL image.
    # This prevents unnecessary colour changes.

    output = image_array.copy()

    detections = []

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


            category = classify_detection(
                class_name
            )


            # GOOD DETECTION:
            # Do not draw box.

            if category == "good":

                continue


            # Unknown detection:
            # Ignore.

            if category != "bad":

                continue


            # BAD DETECTION

            has_bad = True

            display_name = (
                "PROTRUDING FIBER"
            )

            box_color = (
                0,
                0,
                255
            )


            detections.append({

                "name":
                    display_name,

                "original_class":
                    class_name,

                "confidence":
                    confidence,

                "category":
                    "bad",

                "box":
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )

            })


            # ------------------------------------------------
            # RED YOLO BOX
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
                f"{display_name} "
                f"{confidence * 100:.1f}%"
            )


            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            font_scale = 0.70

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

    quality = (
        "BAD"
        if has_bad
        else
        "GOOD"
    )


    return (
        output,
        detections,
        quality
    )


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
# VIDEO PROCESSING
# ============================================================

def process_video(input_path):

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


    output_file = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )
    )


    output_file.close()


    output_path = (
        output_file.name
    )


    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )


    writer = cv2.VideoWriter(

        output_path,

        fourcc,

        fps,

        (
            width,
            height
        )
    )


    last_bad_boxes = []

    all_defects = {}

    has_bad = False


    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = (
            cap.read()
        )


        if not ret:

            break


        result = model.predict(

            source=frame,

            conf=0.25,

            verbose=False

        )[0]


        current_bad_boxes = []


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


                class_name = (
                    get_class_name(
                        class_id
                    )
                )


                category = (
                    classify_detection(
                        class_name
                    )
                )


                # LOOP FIBER = GOOD

                if category == "good":

                    continue


                # UNKNOWN = IGNORE

                if category != "bad":

                    continue


                # PROTRUDING FIBER = BAD

                has_bad = True

                display_name = (
                    "PROTRUDING FIBER"
                )


                box_color = (
                    0,
                    0,
                    255
                )


                current_bad_boxes.append({

                    "box":
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                    "color":
                        box_color,

                    "name":
                        display_name,

                    "confidence":
                        confidence

                })


                if (

                    display_name
                    not in all_defects

                    or

                    confidence
                    >
                    all_defects[
                        display_name
                    ]

                ):

                    all_defects[
                        display_name
                    ] = confidence


        # ----------------------------------------------------
        # KEEP PREVIOUS BOX DURING SMALL DETECTION GAPS
        # ----------------------------------------------------

        if current_bad_boxes:

            last_bad_boxes = (
                current_bad_boxes
            )


        boxes_to_draw = (

            current_bad_boxes

            if current_bad_boxes

            else

            last_bad_boxes

        )


        processed = (
            frame.copy()
        )


        # ----------------------------------------------------
        # DRAW BOXES
        # ----------------------------------------------------

        for detection in boxes_to_draw:

            x1, y1, x2, y2 = (
                detection["box"]
            )


            box_color = (
                detection["color"]
            )


            name = (
                detection["name"]
            )


            confidence = (
                detection["confidence"]
            )


            cv2.rectangle(

                processed,

                (x1, y1),

                (x2, y2),

                box_color,

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


    quality = (
        "BAD"
        if has_bad
        else
        "GOOD"
    )


    browser_video = (
        convert_video_for_browser(
            output_path
        )
    )


    return (
        browser_video,
        all_defects,
        quality
    )


# ============================================================
# HOME PAGE
# ============================================================

if (
    st.session_state.page
    == "home"
):


    # ========================================================
    # NORMAL TITLE ONLY
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
    # AICW
    # ========================================================

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                """
                <h2 style="
                    text-align:center;
                    color:#263238;
                    margin-top:15px;
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
                    margin-top:35px;
                ">
                    Capstone Project
                </h3>
                """,
                unsafe_allow_html=True
            )


            st.write("")

            st.write("")

            st.write("")


        st.write("")


        # ====================================================
        # PREDICT
        # ====================================================

        if st.button(

            "🔍  PREDICT",

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
                <h2 style="
                    color:#4a148c;
                    margin-bottom:18px;
                ">
                    Project Description
                </h2>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div style="
                    line-height:1.7;
                    font-size:15px;
                    color:#263238;
                ">

                <p>
                YarnX is an AI-powered yarn quality inspection
                system designed to automatically detect and
                identify yarn defects using
                <b>Computer Vision</b> and
                <b>Deep Learning</b>.
                </p>

                <p>
                The system accepts yarn images, camera input,
                and videos for inspection. A trained YOLO model
                analyzes the yarn and identifies defective
                regions using bounding boxes.
                </p>

                <p>
                The system displays the detected fiber,
                confidence score, and final quality result
                as <b>GOOD</b> or <b>BAD</b>. This helps
                reduce manual inspection effort and supports
                faster and more accurate yarn quality
                assessment.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # TEAM / EMAIL / GUIDE
    # ========================================================

    st.write("")


    team_col, email_col, guide_col = (
        st.columns(
            [1.2, 1.3, 0.9],
            gap="medium"
        )
    )


    # ========================================================
    # TEAM
    # ========================================================

    with team_col:

        with st.container(
            border=True
        ):

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


    # ========================================================
    # GMAIL
    # ========================================================

    with email_col:

        with st.container(
            border=True
        ):

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
                "ramadevidevigiladevara0@gmail.com"
            )


            st.write(
                "harshitharambala3@gmail.com"
            )


    # ========================================================
    # GUIDE
    # ========================================================

    with guide_col:

        with st.container(
            border=True
        ):

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
# INSPECTION PAGE
# ============================================================

else:


    # ========================================================
    # NORMAL TITLE ONLY
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
    # BACK
    # ========================================================

    if st.button(
        "⬅ Back"
    ):

        st.session_state.page = (
            "home"
        )


        st.session_state.image_input = (
            None
        )


        st.session_state.image_result = (
            None
        )


        st.session_state.image_defects = (
            []
        )


        st.session_state.image_quality = (
            None
        )


        st.session_state.video_output = (
            None
        )


        st.session_state.video_defects = (
            {}
        )


        st.session_state.video_quality = (
            None
        )


        st.rerun()


    # ========================================================
    # INPUT / RESULT
    # ========================================================

    left, right = st.columns(
        [45, 55],
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

        if (
            input_type
            == "🖼️ Image"
        ):

            uploaded_image = (
                st.file_uploader(

                    "Upload Image",

                    type=[
                        "jpg",
                        "jpeg",
                        "png",
                        "webp"
                    ],

                    key="image_upload"

                )
            )


            if uploaded_image:

                image = (
                    Image.open(
                        uploaded_image
                    ).convert("RGB")
                )


                # ORIGINAL IMAGE STORED

                st.session_state.image_input = (
                    image.copy()
                )


                st.write(
                    "**INPUT IMAGE**"
                )


                # ORIGINAL IMAGE
                # No colour conversion.

                st.image(
                    image,
                    width=260
                )


                # ------------------------------------------------
                # ANALYZE IMAGE
                # ------------------------------------------------

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


                    st.session_state.video_output = (
                        None
                    )


                    st.session_state.video_defects = (
                        {}
                    )


                    st.session_state.video_quality = (
                        None
                    )


                    st.rerun()


        # ====================================================
        # CAMERA
        # ====================================================

        elif (
            input_type
            == "📷 Camera"
        ):

            camera_image = (
                st.camera_input(
                    "Capture Yarn"
                )
            )


            if camera_image:

                image = (
                    Image.open(
                        camera_image
                    ).convert("RGB")
                )


                st.session_state.image_input = (
                    image.copy()
                )


                st.write(
                    "**INPUT IMAGE**"
                )


                st.image(
                    image,
                    width=260
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


                    st.session_state.video_output = (
                        None
                    )


                    st.session_state.video_defects = (
                        {}
                    )


                    st.session_state.video_quality = (
                        None
                    )


                    st.rerun()


        # ====================================================
        # VIDEO
        # ====================================================

        else:

            uploaded_video = (
                st.file_uploader(

                    "Upload Video",

                    type=[
                        "mp4",
                        "avi",
                        "mov",
                        "mkv"
                    ],

                    key="video_upload"

                )
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


                st.write(
                    "**INPUT VIDEO**"
                )


                st.video(
                    uploaded_video,
                    width=320
                )


                # ------------------------------------------------
                # ANALYZE VIDEO
                # ------------------------------------------------

                if st.button(

                    "🔍 Analyze Video",

                    use_container_width=True

                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        (
                            video_result,
                            defects,
                            quality
                        ) = process_video(

                            input_video_file.name

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


                    st.session_state.image_result = (
                        None
                    )


                    st.session_state.image_input = (
                        None
                    )


                    st.session_state.image_defects = (
                        []
                    )


                    st.session_state.image_quality = (
                        None
                    )


                    st.rerun()


    # ========================================================
    # RESULT
    # ========================================================

    with right:

        st.subheader(
            "🤖 RESULT"
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


            # ORIGINAL COLOUR + YOLO BOX

            st.image(
                st.session_state.image_result,
                width=330
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
                        🟢 LOOP FIBER
                        <br>
                        ✅ GOOD QUALITY
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
                    <div class="bad-quality">
                        🔴 PROTRUDING FIBER
                        <br>
                        ❌ BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.write(
                    "### 🔴 Detected Defect"
                )


                for detection in (
                    st.session_state.image_defects
                ):

                    st.markdown(
                        f"""
                        <div class="defect-card">

                        🔴 <b>
                        {detection["name"]}
                        </b>

                        &nbsp;&nbsp;

                        📊 Confidence:
                        {detection["confidence"] * 100:.2f}%

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

            st.write(
                "**ANALYZED VIDEO**"
            )


            st.video(
                st.session_state.video_output,
                width=340
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
                        🟢 LOOP FIBER
                        <br>
                        ✅ GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # BAD VIDEO
            # ------------------------------------------------

            else:

                st.markdown(
                    """
                    <div class="bad-quality">
                        🔴 PROTRUDING FIBER
                        <br>
                        ❌ BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.write(
                    "### 🔴 Detected Defect"
                )


                for (
                    name,
                    confidence
                ) in (
                    st.session_state
                    .video_defects
                    .items()
                ):

                    st.markdown(
                        f"""
                        <div class="defect-card">

                        🔴 <b>
                        {name}
                        </b>

                        &nbsp;&nbsp;

                        📊 Confidence:
                        {confidence * 100:.2f}%

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
