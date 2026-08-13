import os
import shutil
import subprocess
import tempfile

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
# MODEL PATH
# ============================================================

MODEL_PATH = "best (6).pt"


# ============================================================
# PYTORCH COMPATIBILITY
# ============================================================

_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):

    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


# ============================================================
# LOAD MODEL
# ============================================================

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


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Remove unnecessary top space */

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }


    /* Main title */

    .main-title {
        border: 2px solid #222;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        padding: 10px;
        margin-bottom: 10px;
    }


    /* First page */

    .home-left {
        border: 2px solid #222;
        height: 560px;
        padding: 25px;
        text-align: center;
    }


    .home-right {
        border: 2px solid #222;
        height: 560px;
        padding: 20px;
    }


    .description {
        font-size: 15px;
        line-height: 1.45;
        text-align: justify;
    }


    .team-area {
        border-top: 2px solid #222;
        margin-top: 12px;
        padding-top: 8px;
    }


    .team-heading {
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 5px;
    }


    .team-text {
        font-size: 12px;
        line-height: 1.35;
    }


    /* Second page */

    .input-panel,
    .result-panel {
        border: 2px solid #222;
        min-height: 650px;
        padding: 15px;
    }


    .preview-title {
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 5px;
    }


    /* Fixed preview */

    .preview-container {
        height: 300px;
        width: 100%;
        border: 2px solid #777;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
        background: #fafafa;
    }


    .preview-container img {
        max-width: 100%;
        max-height: 290px;
        object-fit: contain;
    }


    .result-image {
        max-height: 360px !important;
        width: auto !important;
        object-fit: contain;
    }


    /* Quality */

    .good-quality {
        border: 2px solid green;
        padding: 8px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 8px;
    }


    .bad-quality {
        border: 2px solid red;
        padding: 8px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 8px;
    }


    .defect-card {
        border: 1px solid #777;
        padding: 7px;
        margin-top: 5px;
        font-size: 14px;
    }


    /* Buttons */

    div.stButton > button {
        font-weight: bold;
    }


    /* File uploader */

    [data-testid="stFileUploader"] {
        margin-bottom: 5px;
    }


    /* Radio */

    div[role="radiogroup"] {
        gap: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER - GET DEFECTS
# ============================================================

def get_defects(result):

    defects = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            defect_name = model.names[class_id]

            defects.append(
                {
                    "name": defect_name,
                    "confidence": confidence
                }
            )

    return defects


# ============================================================
# HELPER - VIDEO CONVERSION
# ============================================================

def convert_video_to_browser_format(input_path):

    """
    Converts YOLO generated video to H.264/AAC MP4
    so that Streamlit/browser can play it.
    """

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:

        return input_path


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
# PAGE 1 - HOME
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
    # LEFT - AICW
    # ========================================================

    with left:

        st.markdown(
            """
            <div class="home-left">

                <h2>
                    AI Career for Women (AICW)
                </h2>

                <h3>
                    Capstone Project
                </h3>

                <br><br>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        if st.button(
            "PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "inspection"

            st.rerun()


    # ========================================================
    # RIGHT - DESCRIPTION + DETAILS
    # ========================================================

    with right:

        st.markdown(
            """
            <div class="home-right">

                <h2>
                    Project Description
                </h2>

                <div class="description">

                    YarnX is an AI-powered yarn quality
                    inspection system designed to
                    automatically detect and identify
                    yarn defects using Computer Vision
                    and Deep Learning.

                    <br><br>

                    The system accepts yarn images,
                    camera input, and videos for inspection.
                    It uses a trained YOLO model to detect
                    defects, display bounding boxes around
                    defective regions, identify the defect
                    type, and provide a quality result as
                    GOOD or BAD.

                    <br><br>

                    This system helps reduce manual
                    inspection effort, improve detection
                    accuracy, and support faster yarn
                    quality assessment.

                </div>

                <div class="team-area">

                    <div class="team-heading">
                        TEAM MEMBERS
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # Team details

        c1, c2, c3 = st.columns(
            [1.25, 1.1, 0.85],
            gap="small"
        )


        with c1:

            st.markdown(
                """
                <div class="team-text">

                <b>1. Gutti.Pavani<br>
                Devi Priya</b><br><br>

                <b>2. Somasani.Sasi<br>
                Priya</b><br><br>

                <b>3. Galidevara.Rama<br>
                Devi</b><br><br>

                <b>4. Rambala.Harshitha<br>
                Sai Lakshmi</b>

                </div>
                """,
                unsafe_allow_html=True
            )


        with c2:

            st.markdown(
                """
                <div class="team-text">

                gutthipavanidevipriya<br>
                @gmail.com<br><br>

                Sasipriya8090<br>
                @gmail.com<br><br>

                ramadevigalidevara0<br>
                @gmail.com<br><br>

                harshitharambala3<br>
                @gmail.com

                </div>
                """,
                unsafe_allow_html=True
            )


        with c3:

            st.markdown(
                """
                <div class="team-text">

                <b>GUIDE NAME</b><br><br>

                Md. Abdul Aziz<br><br>

                <b>DESIGNATION</b><br><br>

                Co Lead &<br>
                Trainer AICW

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PAGE 2 - INSPECTION
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


    # Back button

    if st.button("⬅ Back to Home"):

        st.session_state.page = "home"

        st.rerun()


    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # ========================================================
    # LEFT - INPUT
    # ========================================================

    with left:

        st.markdown(
            """
            <div class="input-panel">

            <h2>📥 INPUT</h2>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "### Select Input Type:"
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

            uploaded_file = st.file_uploader(
                "Upload Yarn Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp"
                ],
                key="image_upload"
            )


            if uploaded_file is not None:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")


                st.markdown(
                    '<div class="preview-title">INPUT PREVIEW</div>',
                    unsafe_allow_html=True
                )


                st.image(
                    image,
                    use_container_width=True
                )


                analyze = st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                )


                if analyze:

                    with st.spinner(
                        "Analyzing image..."
                    ):

                        results = model.predict(
                            source=np.array(image),
                            conf=0.25,
                            verbose=False
                        )


                    result = results[0]

                    output_image = result.plot()

                    defects = get_defects(
                        result
                    )


                    # Store result
                    st.session_state.image_result = (
                        output_image,
                        defects
                    )


        # ====================================================
        # CAMERA
        # ====================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Capture Yarn Image"
            )


            if camera_image is not None:

                image = Image.open(
                    camera_image
                ).convert("RGB")


                st.markdown(
                    '<div class="preview-title">CAMERA PREVIEW</div>',
                    unsafe_allow_html=True
                )


                st.image(
                    image,
                    use_container_width=True
                )


                analyze = st.button(
                    "🔍 Analyze Camera Image",
                    use_container_width=True
                )


                if analyze:

                    with st.spinner(
                        "Analyzing camera image..."
                    ):

                        results = model.predict(
                            source=np.array(image),
                            conf=0.25,
                            verbose=False
                        )


                    result = results[0]

                    output_image = result.plot()

                    defects = get_defects(
                        result
                    )


                    st.session_state.image_result = (
                        output_image,
                        defects
                    )


        # ====================================================
        # VIDEO
        # ====================================================

        elif input_type == "🎥 Video":

            uploaded_video = st.file_uploader(
                "Upload Yarn Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv"
                ],
                key="video_upload"
            )


            if uploaded_video is not None:

                st.markdown(
                    '<div class="preview-title">INPUT VIDEO</div>',
                    unsafe_allow_html=True
                )


                st.video(
                    uploaded_video
                )


                analyze = st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                )


                if analyze:

                    with st.spinner(
                        "Analyzing video frames..."
                    ):

                        # --------------------------------
                        # INPUT TEMP FILE
                        # --------------------------------

                        input_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_file.write(
                            uploaded_video.getvalue()
                        )

                        input_file.close()


                        # --------------------------------
                        # VIDEO CAPTURE
                        # --------------------------------

                        cap = cv2.VideoCapture(
                            input_file.name
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


                        # --------------------------------
                        # RAW OUTPUT
                        # --------------------------------

                        raw_output = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        raw_output_path = (
                            raw_output.name
                        )

                        raw_output.close()


                        fourcc = (
                            cv2.VideoWriter_fourcc(
                                *"mp4v"
                            )
                        )


                        writer = cv2.VideoWriter(
                            raw_output_path,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        all_defects = {}


                        # --------------------------------
                        # PROCESS EVERY FRAME
                        # --------------------------------

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


                            # YOLO bounding boxes
                            processed_frame = (
                                result.plot()
                            )


                            writer.write(
                                processed_frame
                            )


                            # Collect defects
                            if (
                                result.boxes
                                is not None
                            ):

                                for box in result.boxes:

                                    class_id = int(
                                        box.cls[0].item()
                                    )

                                    confidence = float(
                                        box.conf[0].item()
                                    )

                                    defect_name = (
                                        model.names[
                                            class_id
                                        ]
                                    )


                                    if (
                                        defect_name
                                        not in all_defects
                                    ):

                                        all_defects[
                                            defect_name
                                        ] = confidence

                                    elif confidence > all_defects[
                                        defect_name
                                    ]:

                                        all_defects[
                                            defect_name
                                        ] = confidence


                        cap.release()

                        writer.release()


                        # --------------------------------
                        # CONVERT TO H264
                        # --------------------------------

                        browser_video = (
                            convert_video_to_browser_format(
                                raw_output_path
                            )
                        )


                        st.session_state.video_result = (
                            browser_video,
                            all_defects
                        )


                        # Remove input
                        try:
                            os.remove(
                                input_file.name
                            )
                        except:
                            pass


    # ========================================================
    # RIGHT - RESULT
    # ========================================================

    with right:

        st.markdown(
            """
            <div class="result-panel">

            <h2>🤖 INSPECTION RESULT</h2>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # IMAGE RESULT
        # ====================================================

        if "image_result" in st.session_state:

            output_image, defects = (
                st.session_state.image_result
            )


            st.image(
                output_image,
                caption="ANALYZED OUTPUT",
                width=650
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


                st.markdown(
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
        # VIDEO RESULT
        # ====================================================

        elif "video_result" in st.session_state:

            video_path, all_defects = (
                st.session_state.video_result
            )


            st.markdown(
                '<div class="preview-title">ANALYZED VIDEO</div>',
                unsafe_allow_html=True
            )


            # Browser compatible video
            with open(
                video_path,
                "rb"
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )


            st.video(
                video_bytes,
                format="video/mp4"
            )


            if len(all_defects) > 0:

                st.markdown(
                    """
                    <div class="bad-quality">
                        ❌ BAD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                    "### Detected Defects"
                )


                for name, confidence in (
                    all_defects.items()
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
        # NO RESULT
        # ====================================================

        else:

            st.info(
                "Upload an image/video and click Analyze to see the inspection result."
            )
