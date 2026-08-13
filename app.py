import os
import tempfile
import subprocess

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="YarnX – The Future of Yarn Inspection",
    page_icon="🧶",
    layout="wide"
)


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = "best (6).pt"


# =========================================================
# PYTORCH FIX
# =========================================================

_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):

    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.error(
            f"❌ {MODEL_PATH} not found."
        )

        st.stop()

    return YOLO(MODEL_PATH)


model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

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


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 10px;
    padding-bottom: 10px;
}


/* MAIN TITLE */

.main-title {
    border: 2px solid #222;
    padding: 10px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 10px;
}


/* HOME */

.home-box {
    border: 2px solid #222;
    height: 475px;
    padding: 20px;
}


/* QUALITY */

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


/* DEFECT */

.defect-box {
    border: 1px solid #777;
    padding: 7px;
    margin-top: 5px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# GET DEFECTS
# =========================================================

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

            defects.append({
                "name": defect_name,
                "confidence": confidence
            })

    return defects


# =========================================================
# RESIZE IMAGE FOR DISPLAY
# =========================================================

def resize_for_display(image, max_width=500, max_height=300):

    if isinstance(image, np.ndarray):

        image = Image.fromarray(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        )


    image = image.copy()

    image.thumbnail(
        (max_width, max_height),
        Image.Resampling.LANCZOS
    )

    return image


# =========================================================
# VIDEO BROWSER CONVERSION
# =========================================================

def convert_video(input_path):

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


# =========================================================
# =========================================================
# PAGE 1
# =========================================================
# =========================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )


    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # =====================================================
    # LEFT
    # =====================================================

    with left:

        with st.container(border=True):

            st.markdown(
                "<h2 style='text-align:center;'>AI Career for Women (AICW)</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<h3 style='text-align:center;'>Capstone Project</h3>",
                unsafe_allow_html=True
            )

            st.write("")
            st.write("")
            st.write("")
            st.write("")


        st.write("")

        if st.button(
            "PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "inspection"

            st.rerun()


    # =====================================================
    # RIGHT
    # =====================================================

    with right:

        with st.container(border=True):

            st.subheader(
                "Project Description"
            )

            st.write(
                """
                YarnX is an AI-powered yarn quality
                inspection system designed to automatically
                detect and identify yarn defects using
                Computer Vision and Deep Learning.
                """
            )

            st.write(
                """
                The system accepts yarn images, camera
                input, and videos for inspection. A trained
                YOLO model analyzes the yarn and identifies
                defective regions by drawing bounding boxes
                around detected defects.
                """
            )

            st.write(
                """
                The system displays the detected defect,
                confidence score, and final quality result
                as GOOD or BAD. This helps reduce manual
                inspection effort and supports faster and
                more accurate yarn quality assessment.
                """
            )


    # =====================================================
    # TEAM DETAILS
    # =====================================================

    st.write("")

    team_col, mail_col, guide_col = st.columns(
        [1.35, 1.25, 0.9],
        gap="small"
    )


    with team_col:

        st.markdown("**TEAM MEMBERS**")

        st.write("1. Gutti.Pavani Devi Priya")
        st.write("2. Somasani.Sasi Priya")
        st.write("3. Galidevara.Rama Devi")
        st.write("4. Rambala.Harshitha Sai Lakshmi")


    with mail_col:

        st.markdown("**GMAIL**")

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

        st.markdown("**GUIDE NAME**")

        st.write(
            "Md. Abdul Aziz"
        )

        st.markdown("**DESIGNATION**")

        st.write(
            "Co Lead & Trainer AICW"
        )


# =========================================================
# =========================================================
# PAGE 2
# =========================================================
# =========================================================

elif st.session_state.page == "inspection":

    st.markdown(
        '<div class="main-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # BACK
    # =====================================================

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


    # =====================================================
    # LEFT - INPUT
    # =====================================================

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


        # =================================================
        # IMAGE
        # =================================================

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


                # SMALL FIXED DISPLAY
                preview = resize_for_display(
                    image,
                    max_width=400,
                    max_height=230
                )


                st.image(
                    preview,
                    width=400
                )


                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        results = model.predict(
                            source=np.array(image),
                            conf=0.25,
                            verbose=False
                        )


                    result = results[0]


                    # =================================================
                    # CLEAR AND THICK YOLO BOXES
                    # =================================================

                    output = result.plot(
                        conf=True,
                        labels=True,
                        boxes=True,
                        line_width=4,
                        font_size=16
                    )


                    defects = get_defects(
                        result
                    )


                    st.session_state.image_output = output
                    st.session_state.image_defects = defects

                    st.session_state.video_output = None
                    st.session_state.video_defects = {}


        # =================================================
        # CAMERA
        # =================================================

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


                preview = resize_for_display(
                    image,
                    max_width=400,
                    max_height=230
                )


                st.image(
                    preview,
                    width=400
                )


                if st.button(
                    "🔍 Analyze Camera",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing yarn..."
                    ):

                        results = model.predict(
                            source=np.array(image),
                            conf=0.25,
                            verbose=False
                        )


                    result = results[0]


                    output = result.plot(
                        conf=True,
                        labels=True,
                        boxes=True,
                        line_width=4,
                        font_size=16
                    )


                    defects = get_defects(
                        result
                    )


                    st.session_state.image_output = output
                    st.session_state.image_defects = defects

                    st.session_state.video_output = None
                    st.session_state.video_defects = {}


        # =================================================
        # VIDEO
        # =================================================

        elif input_type == "🎥 Video":

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


                # SMALL VIDEO
                st.video(
                    uploaded_video,
                    format="video/mp4"
                )


                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video frames..."
                    ):

                        # ---------------------------------
                        # SAVE INPUT
                        # ---------------------------------

                        input_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_file.write(
                            uploaded_video.getvalue()
                        )

                        input_file.close()


                        # ---------------------------------
                        # OPEN VIDEO
                        # ---------------------------------

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


                        # ---------------------------------
                        # OUTPUT
                        # ---------------------------------

                        output_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        output_file.close()

                        raw_output = output_file.name


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


                        all_defects = {}


                        # ---------------------------------
                        # PROCESS EVERY FRAME
                        # ---------------------------------

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


                            # =====================================
                            # CLEAR THICK BOXES + LABELS
                            # =====================================

                            processed_frame = result.plot(
                                conf=True,
                                labels=True,
                                boxes=True,
                                line_width=4,
                                font_size=16
                            )


                            writer.write(
                                processed_frame
                            )


                            # ---------------------------------
                            # COLLECT DEFECTS
                            # ---------------------------------

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
                                    ):

                                        all_defects[
                                            name
                                        ] = confidence

                                    elif (
                                        confidence
                                        >
                                        all_defects[name]
                                    ):

                                        all_defects[
                                            name
                                        ] = confidence


                        cap.release()
                        writer.release()


                        # ---------------------------------
                        # CONVERT VIDEO
                        # ---------------------------------

                        final_video = convert_video(
                            raw_output
                        )


                        st.session_state.video_output = (
                            final_video
                        )

                        st.session_state.video_defects = (
                            all_defects
                        )

                        st.session_state.image_output = None
                        st.session_state.image_defects = []


    # =====================================================
    # RIGHT - RESULT
    # =====================================================

    with right:

        st.subheader(
            "🤖 INSPECTION RESULT"
        )


        # =================================================
        # IMAGE RESULT
        # =================================================

        if st.session_state.image_output is not None:

            st.write(
                "**ANALYZED IMAGE**"
            )


            # ---------------------------------------------
            # SMALL OUTPUT
            # ---------------------------------------------

            result_image = resize_for_display(
                st.session_state.image_output,
                max_width=550,
                max_height=320
            )


            st.image(
                result_image,
                width=550
            )


            defects = (
                st.session_state.image_defects
            )


            # ---------------------------------------------
            # BAD
            # ---------------------------------------------

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
                        <div class="defect-box">

                        🔴 <b>{defect["name"]}</b>

                        &nbsp;&nbsp;&nbsp;

                        Confidence:
                        {defect["confidence"] * 100:.2f}%

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # ---------------------------------------------
            # GOOD
            # ---------------------------------------------

            else:

                st.markdown(
                    """
                    <div class="good-quality">
                    ✅ GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # =================================================
        # VIDEO RESULT
        # =================================================

        elif st.session_state.video_output is not None:

            st.write(
                "**ANALYZED VIDEO**"
            )


            with open(
                st.session_state.video_output,
                "rb"
            ) as f:

                video_bytes = f.read()


            # ---------------------------------------------
            # VIDEO DISPLAY
            # ---------------------------------------------

            st.video(
                video_bytes,
                format="video/mp4"
            )


            defects = (
                st.session_state.video_defects
            )


            # ---------------------------------------------
            # BAD
            # ---------------------------------------------

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


                for name, confidence in defects.items():

                    st.markdown(
                        f"""
                        <div class="defect-box">

                        🔴 <b>{name}</b>

                        &nbsp;&nbsp;&nbsp;

                        Confidence:
                        {confidence * 100:.2f}%

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # ---------------------------------------------
            # GOOD
            # ---------------------------------------------

            else:

                st.markdown(
                    """
                    <div class="good-quality">
                    ✅ GOOD QUALITY
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # =================================================
        # BEFORE ANALYSIS
        # =================================================

        else:

            st.info(
                "Upload an image or video and click Analyze."
            )
