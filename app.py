import os
import tempfile
import subprocess
import shutil

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="YarnX",
    page_icon="🧶",
    layout="wide"
)


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = "best (6).pt"


# PyTorch 2.6+ compatibility for trusted local model
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


@st.cache_resource
def get_model():
    return YOLO(MODEL_PATH)


model = get_model()


# =========================================================
# SESSION
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "image_result" not in st.session_state:
    st.session_state.image_result = None

if "video_result" not in st.session_state:
    st.session_state.video_result = None


# =========================================================
# CSS
# =========================================================

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
    padding-top: 10px;
    padding-bottom: 10px;
}


/* =========================
   COMMON TITLE
   ========================= */

.title {
    border: 2px solid #222;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    padding: 12px;
    margin-bottom: 10px;
}


/* =========================
   FIRST PAGE
   ========================= */

.home-box {
    border: 2px solid #222;
    height: 520px;
    padding: 20px;
}

.description {
    font-size: 15px;
    line-height: 1.45;
    text-align: justify;
}

.team-title {
    font-size: 13px;
    font-weight: bold;
}

.team-text {
    font-size: 11px;
    line-height: 1.25;
}


/* =========================
   SECOND PAGE
   ========================= */

.panel-title {
    font-size: 21px;
    font-weight: bold;
    margin-bottom: 8px;
}

.preview-label {
    text-align: center;
    font-weight: bold;
    margin-bottom: 5px;
}

.preview {
    height: 250px;
    border: 2px solid #333;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
}


/* =========================
   QUALITY
   ========================= */

.good {
    border: 2px solid green;
    text-align: center;
    font-size: 23px;
    font-weight: bold;
    padding: 7px;
    margin-top: 8px;
}

.bad {
    border: 2px solid red;
    text-align: center;
    font-size: 23px;
    font-weight: bold;
    padding: 7px;
    margin-top: 8px;
}

.defect {
    border: 1px solid #777;
    padding: 6px;
    margin-top: 5px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNCTION: GET DEFECTS
# =========================================================

def get_defects(result):

    defects = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            name = model.names[class_id]

            defects.append({
                "name": name,
                "confidence": confidence
            })

    return defects


# =========================================================
# FUNCTION: CONVERT VIDEO
# =========================================================

def make_browser_video(input_video):

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        return input_video

    output_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    command = [
        ffmpeg,
        "-y",
        "-i",
        input_video,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        return output_video

    except Exception:

        return input_video


# =========================================================
# PAGE 1
# =========================================================

if st.session_state.page == 1:

    st.markdown(
        '<div class="title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # -----------------------------------------------------
    # LEFT
    # -----------------------------------------------------

    with left:

        st.markdown(
            """
            <div class="home-box">

                <h2 style="text-align:center;">
                    AI Career for Women (AICW)
                </h2>

                <h3 style="text-align:center;">
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

            st.session_state.page = 2

            st.rerun()


    # -----------------------------------------------------
    # RIGHT
    # -----------------------------------------------------

    with right:

        st.markdown(
            """
            <div class="home-box">

                <h2>Project Description</h2>

                <div class="description">

                YarnX is an AI-powered yarn quality
                inspection system designed to automatically
                detect and identify yarn defects using
                Computer Vision and Deep Learning.

                <br><br>

                The system accepts yarn images, camera
                input, and videos for inspection. A trained
                YOLO model analyzes the yarn and identifies
                defective regions by drawing bounding boxes
                around detected defects.

                <br><br>

                The system displays the detected defect,
                confidence score, and final quality result
                as GOOD or BAD. This helps reduce manual
                inspection effort and supports faster and
                more accurate yarn quality assessment.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        c1, c2, c3 = st.columns(
            [1.25, 1.1, 0.85],
            gap="small"
        )

        with c1:

            st.markdown(
                '<div class="team-title">TEAM MEMBERS</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="team-text">

                1. Gutti.Pavani<br>
                &nbsp;&nbsp;&nbsp;Devi Priya<br><br>

                2. Somasani.Sasi<br>
                &nbsp;&nbsp;&nbsp;Priya<br><br>

                3. Galidevara.Rama<br>
                &nbsp;&nbsp;&nbsp;Devi<br><br>

                4. Rambala.Harshitha<br>
                &nbsp;&nbsp;&nbsp;Sai Lakshmi

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                '<div class="team-title">GMAIL</div>',
                unsafe_allow_html=True
            )

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
                '<div class="team-title">GUIDE NAME</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="team-text">

                Md. Abdul Aziz

                <br><br>

                <b>Designation</b>

                <br><br>

                Co Lead &<br>
                Trainer AICW

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# PAGE 2
# =========================================================

elif st.session_state.page == 2:

    st.markdown(
        '<div class="title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )

    if st.button("⬅ Back"):

        st.session_state.page = 1

        st.session_state.image_result = None
        st.session_state.video_result = None

        st.rerun()


    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # =====================================================
    # INPUT
    # =====================================================

    with left:

        st.markdown(
            '<div class="panel-title">📥 INPUT</div>',
            unsafe_allow_html=True
        )

        st.markdown(
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


        # =================================================
        # IMAGE
        # =================================================

        if input_type == "🖼️ Image":

            file = st.file_uploader(
                "Upload Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp"
                ],
                key="image"
            )

            if file:

                image = Image.open(
                    file
                ).convert("RGB")

                st.markdown(
                    '<div class="preview-label">INPUT PREVIEW</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    width=400
                )

                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing..."
                    ):

                        results = model.predict(
                            np.array(image),
                            conf=0.25,
                            verbose=False
                        )

                    result = results[0]

                    output = result.plot()

                    defects = get_defects(
                        result
                    )

                    st.session_state.image_result = (
                        output,
                        defects
                    )

                    st.session_state.video_result = None


        # =================================================
        # CAMERA
        # =================================================

        elif input_type == "📷 Camera":

            camera = st.camera_input(
                "Capture Yarn"
            )

            if camera:

                image = Image.open(
                    camera
                ).convert("RGB")

                st.image(
                    image,
                    width=400
                )

                if st.button(
                    "🔍 Analyze Camera",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing..."
                    ):

                        results = model.predict(
                            np.array(image),
                            conf=0.25,
                            verbose=False
                        )

                    result = results[0]

                    output = result.plot()

                    defects = get_defects(
                        result
                    )

                    st.session_state.image_result = (
                        output,
                        defects
                    )

                    st.session_state.video_result = None


        # =================================================
        # VIDEO
        # =================================================

        elif input_type == "🎥 Video":

            video = st.file_uploader(
                "Upload Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv"
                ],
                key="video"
            )

            if video:

                st.markdown(
                    '<div class="preview-label">INPUT VIDEO</div>',
                    unsafe_allow_html=True
                )

                st.video(
                    video
                )

                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video..."
                    ):

                        input_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_file.write(
                            video.getvalue()
                        )

                        input_file.close()


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


                        raw_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        raw_file.close()

                        raw_output = raw_file.name


                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )

                        writer = cv2.VideoWriter(
                            raw_output,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        all_defects = {}


                        while True:

                            ret, frame = cap.read()

                            if not ret:
                                break


                            results = model.predict(
                                frame,
                                conf=0.25,
                                verbose=False
                            )

                            result = results[0]


                            # Draw bounding boxes
                            processed = result.plot()


                            writer.write(
                                processed
                            )


                            if result.boxes is not None:

                                for box in result.boxes:

                                    class_id = int(
                                        box.cls[0]
                                    )

                                    confidence = float(
                                        box.conf[0]
                                    )

                                    name = model.names[
                                        class_id
                                    ]


                                    if (
                                        name not in all_defects
                                        or
                                        confidence >
                                        all_defects[name]
                                    ):

                                        all_defects[
                                            name
                                        ] = confidence


                        cap.release()
                        writer.release()


                        # Convert to browser-playable MP4
                        final_video = (
                            make_browser_video(
                                raw_output
                            )
                        )


                        st.session_state.video_result = (
                            final_video,
                            all_defects
                        )

                        st.session_state.image_result = None


    # =====================================================
    # RESULT
    # =====================================================

    with right:

        st.markdown(
            '<div class="panel-title">🤖 INSPECTION RESULT</div>',
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # IMAGE RESULT
        # -------------------------------------------------

        if st.session_state.image_result:

            output, defects = (
                st.session_state.image_result
            )

            st.image(
                output,
                width=600
            )


            if defects:

                st.markdown(
                    '<div class="bad">❌ BAD QUALITY</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### Detected Defects"
                )

                for defect in defects:

                    st.markdown(
                        f"""
                        <div class="defect">

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
                    '<div class="good">✅ GOOD QUALITY</div>',
                    unsafe_allow_html=True
                )


        # -------------------------------------------------
        # VIDEO RESULT
        # -------------------------------------------------

        elif st.session_state.video_result:

            video_path, defects = (
                st.session_state.video_result
            )

            with open(
                video_path,
                "rb"
            ) as f:

                video_bytes = f.read()


            st.video(
                video_bytes,
                format="video/mp4"
            )


            if defects:

                st.markdown(
                    '<div class="bad">❌ BAD QUALITY</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### Detected Defects"
                )

                for name, confidence in defects.items():

                    st.markdown(
                        f"""
                        <div class="defect">

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
                    '<div class="good">✅ GOOD QUALITY</div>',
                    unsafe_allow_html=True
                )


        # -------------------------------------------------
        # EMPTY RESULT
        # -------------------------------------------------

        else:

            st.markdown(
                """
                <div class="preview"
                     style="height:350px;">

                    <b>
                    ANALYZED IMAGE / VIDEO
                    </b>

                </div>
                """,
                unsafe_allow_html=True
            )
