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

/* ============================================================
   MAIN PAGE
   ============================================================ */

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 1rem;
}


/* ============================================================
   MAIN TITLE
   ============================================================ */

.main-title {

    border: 2px solid #6a1b9a;

    border-radius: 16px;

    padding: 14px;

    text-align: center;

    font-size: 30px;

    font-weight: 800;

    color: #4a148c;

    background: linear-gradient(
        90deg,
        #ede7f6,
        #e3f2fd,
        #fce4ec
    );

    margin-bottom: 18px;

    box-shadow:
        0 4px 12px rgba(106, 27, 154, 0.12);
}


/* ============================================================
   PREDICT BUTTON
   ============================================================ */

.stButton > button {

    height: 52px;

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
        0 5px 14px rgba(106, 27, 154, 0.30);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}


.stButton > button:hover {

    color: white;

    transform: translateY(-2px);

    box-shadow:
        0 8px 20px rgba(106, 27, 154, 0.40);
}


/* ============================================================
   TEAM CARD
   ============================================================ */

.team-card {

    border: 1px solid #d1c4e9;

    border-radius: 14px;

    padding: 12px 14px;

    margin-bottom: 10px;

    background: linear-gradient(
        135deg,
        #faf7ff,
        #f3f7ff
    );

    box-shadow:
        0 3px 9px rgba(0,0,0,0.06);
}


.team-title {

    font-size: 16px;

    font-weight: 800;

    color: #4a148c;

    margin-bottom: 12px;
}


.team-name {

    font-size: 14px;

    font-weight: 600;

    color: #263238;

    padding: 8px 0;

    border-bottom: 1px solid #eeeeee;
}


.team-name:last-child {
    border-bottom: none;
}


/* ============================================================
   EMAIL CARD
   ============================================================ */

.email-card {

    border: 1px solid #bbdefb;

    border-radius: 14px;

    padding: 12px 14px;

    margin-bottom: 10px;

    background: linear-gradient(
        135deg,
        #f5fbff,
        #eef7ff
    );

    box-shadow:
        0 3px 9px rgba(33,150,243,0.08);
}


.email-title {

    font-size: 16px;

    font-weight: 800;

    color: #1565c0;

    margin-bottom: 12px;
}


.email-item {

    font-size: 13px;

    padding: 8px 0;

    border-bottom: 1px solid #e3f2fd;
}


.email-item:last-child {
    border-bottom: none;
}


/* ============================================================
   GUIDE CARD
   ============================================================ */

.guide-card {

    border: 1px solid #f8bbd0;

    border-radius: 14px;

    padding: 16px;

    background: linear-gradient(
        135deg,
        #fff7fa,
        #fff3e0
    );

    box-shadow:
        0 4px 12px rgba(233,30,99,0.08);
}


.guide-title {

    font-size: 16px;

    font-weight: 800;

    color: #ad1457;

    margin-bottom: 12px;
}


.guide-name {

    font-size: 17px;

    font-weight: 700;

    color: #263238;

    margin-bottom: 15px;
}


.designation-title {

    font-size: 14px;

    font-weight: 800;

    color: #ad1457;

    margin-bottom: 5px;
}


.designation {

    font-size: 14px;

    color: #37474f;

    font-weight: 600;
}


/* ============================================================
   QUALITY RESULT
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
    width=300,
    height=190,
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
    width=300,
    height=190,
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


        # RED BOX

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
# ============================================================
# FIRST PAGE
# ============================================================
# ============================================================

if st.session_state.page == "home":

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TOP SECTION
    # --------------------------------------------------------

    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # ========================================================
    # AICW / CAPSTONE
    # ========================================================

    with left:

        with st.container(border=True):

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
        # PREDICT BUTTON
        # ====================================================

        if st.button(
            "🔍  START YARN INSPECTION",
            use_container_width=True
        ):

            st.session_state.page = "inspection"

            st.rerun()


    # ========================================================
    # PROJECT DESCRIPTION
    # ========================================================

    with right:

        with st.container(border=True):

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
                and videos for inspection. A trained
                <b>YOLO model</b> analyzes the yarn and
                identifies defective regions by drawing
                bounding boxes around detected defects.
                </p>

                <p>
                The system displays the detected defect,
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


    team, gmail, guide = st.columns(
        [1.2, 1.25, 0.9],
        gap="medium"
    )


    # ========================================================
    # TEAM MEMBERS
    # ========================================================

    with team:

        st.markdown(
            """
            <div class="team-card">

                <div class="team-title">
                    👩‍💻 TEAM MEMBERS
                </div>

                <div class="team-name">
                    1. Gutti.Pavani Devi Priya
                </div>

                <div class="team-name">
                    2. Somasani.Sasi Priya
                </div>

                <div class="team-name">
                    3. Galidevara.Rama Devi
                </div>

                <div class="team-name">
                    4. Rambala.Harshitha Sai Lakshmi
                </div>

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
            <div class="email-card">

                <div class="email-title">
                    📧 GMAIL
                </div>

                <div class="email-item">
                    gutthipavanidevipriya@gmail.com
                </div>

                <div class="email-item">
                    Sasipriya8090@gmail.com
                </div>

                <div class="email-item">
                    ramadevigalidevara0@gmail.com
                </div>

                <div class="email-item">
                    harshitharambala3@gmail.com
                </div>

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
            <div class="guide-card">

                <div class="guide-title">
                    👨‍🏫 GUIDE NAME
                </div>

                <div class="guide-name">
                    Md. Abdul Aziz
                </div>

                <div class="designation-title">
                    DESIGNATION
                </div>

                <div class="designation">
                    Co Lead & Trainer AICW
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ============================================================
# SECOND PAGE
# ============================================================
# ============================================================

else:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BACK BUTTON
    # --------------------------------------------------------

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
                    width=300,
                    height=190,
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
                    width=300,
                    height=190,
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
                    width=300,
                    height=190,
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


                        input_path = (
                            input_temp.name
                        )


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


                        # ------------------------------------
                        # PROCESS EVERY FRAME
                        # ------------------------------------

                        while True:

                            ret, frame = (
                                cap.read()
                            )


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


                            # --------------------------------
                            # KEEP LAST BOX
                            # --------------------------------

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


                            # --------------------------------
                            # DRAW BOXES
                            # --------------------------------

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
                                    cv2
                                    .FONT_HERSHEY_SIMPLEX
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
                width=400,
                height=240,
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
                width=400,
                height=240,
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
        # WAITING
        # ====================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
