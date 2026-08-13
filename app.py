import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="YarnX – The Future of Yarn Inspection",
    page_icon="🧶",
    layout="wide"
)


# ==========================================================
# MODEL
# ==========================================================

MODEL_PATH = "best (6).pt"


# ==========================================================
# PYTORCH COMPATIBILITY
# ==========================================================

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
            f"Model file not found: {MODEL_PATH}"
        )

        st.stop()

    return YOLO(MODEL_PATH)


model = load_model()


# ==========================================================
# SESSION STATE
# ==========================================================

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


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 0.5rem;
}


/* Main title */

.main-title {
    border: 2px solid #222;
    padding: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 12px;
}


/* Home boxes */

.home-box {
    border: 2px solid #222;
    padding: 20px;
    height: 475px;
}


/* Description */

.description-text {
    font-size: 15px;
    line-height: 1.5;
    text-align: justify;
}


/* Second page */

.result-box {
    border: 2px solid #222;
    min-height: 430px;
    padding: 15px;
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

.defect-item {
    border: 1px solid #777;
    padding: 7px;
    margin-top: 5px;
}

</style>
""",
    unsafe_allow_html=True
)


# ==========================================================
# GET DEFECTS
# ==========================================================

def get_defects(result):

    defects = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())

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


# ==========================================================
# VIDEO CONVERSION
# ==========================================================

def convert_video_for_browser(input_path):

    try:

        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    except Exception:

        return input_path


    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name


    command = [
        ffmpeg_path,
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


    import subprocess

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


# ==========================================================
# PAGE 1
# ==========================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )


    left, right = st.columns(
        [35, 65],
        gap="small"
    )


    # ======================================================
    # LEFT SIDE
    # ======================================================

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


        st.write("")

        if st.button(
            "PREDICT",
            use_container_width=True
        ):

            st.session_state.page = "inspection"

            st.rerun()


    # ======================================================
    # RIGHT SIDE
    # ======================================================

    with right:

        with st.container(border=True):

            st.subheader("Project Description")

            st.write(
                """
                YarnX is an AI-powered yarn quality inspection
                system designed to automatically detect and
                identify yarn defects using Computer Vision
                and Deep Learning.
                """
            )

            st.write(
                """
                The system accepts yarn images, camera input,
                and videos for inspection. A trained YOLO model
                analyzes the yarn and identifies defective
                regions by drawing bounding boxes around
                detected defects.
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


    # ======================================================
    # TEAM DETAILS
    # ======================================================

    st.write("")

    team_col, mail_col, guide_col = st.columns(
        [1.35, 1.25, 0.9],
        gap="small"
    )


    with team_col:

        st.markdown("**TEAM MEMBERS**")

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


# ==========================================================
# PAGE 2
# ==========================================================

elif st.session_state.page == "inspection":

    st.markdown(
        '<div class="main-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )


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


    # ======================================================
    # INPUT
    # ======================================================

    with left:

        st.subheader("📥 INPUT")

        st.write("Select Input Type:")

        input_type = st.radio(
            "",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ],
            horizontal=True
        )


        # ==================================================
        # IMAGE INPUT
        # ==================================================

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


            if uploaded_image is not None:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")


                st.write("**INPUT PREVIEW**")

                st.image(
                    image,
                    width=350
                )


                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

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


                    st.session_state.image_output = (
                        output_image
                    )

                    st.session_state.image_defects = (
                        defects
                    )

                    st.session_state.video_output = None


        # ==================================================
        # CAMERA INPUT
        # ==================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Capture Yarn"
            )


            if camera_image is not None:

                image = Image.open(
                    camera_image
                ).convert("RGB")


                st.write("**CAMERA PREVIEW**")

                st.image(
                    image,
                    width=350
                )


                if st.button(
                    "🔍 Analyze Camera",
                    use_container_width=True
                ):

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


                    st.session_state.image_output = (
                        output_image
                    )

                    st.session_state.image_defects = (
                        defects
                    )

                    st.session_state.video_output = None


        # ==================================================
        # VIDEO INPUT
        # ==================================================

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


            if uploaded_video is not None:

                st.write("**INPUT VIDEO**")

                st.video(
                    uploaded_video
                )


                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video..."
                    ):

                        # Save uploaded video

                        input_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_temp.write(
                            uploaded_video.getvalue()
                        )

                        input_temp.close()


                        # Open video

                        cap = cv2.VideoCapture(
                            input_temp.name
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


                        # Temporary output

                        output_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
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


                        all_defects = {}


                        # Process frames

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


                            # Draw boxes
                            processed_frame = (
                                result.plot()
                            )


                            writer.write(
                                processed_frame
                            )


                            if result.boxes is not None:

                                for box in result.boxes:

                                    class_id = int(
                                        box.cls[0]
                                    )

                                    confidence = float(
                                        box.conf[0]
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


                        cap.release()

                        writer.release()


                        # Convert for browser playback

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


    # ======================================================
    # INSPECTION RESULT
    # ======================================================

    with right:

        st.subheader(
            "🤖 INSPECTION RESULT"
        )


        # ==================================================
        # IMAGE RESULT
        # ==================================================

        if st.session_state.image_output is not None:

            st.image(
                st.session_state.image_output,
                width=550
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

                    st.write(
                        f"🔴 **Defect:** "
                        f"{defect['name']}"
                    )

                    st.write(
                        f"📊 **Confidence:** "
                        f"{defect['confidence'] * 100:.2f}%"
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


        # ==================================================
        # VIDEO RESULT
        # ==================================================

        elif st.session_state.video_output is not None:

            st.write(
                "**ANALYZED VIDEO**"
            )


            with open(
                st.session_state.video_output,
                "rb"
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )


            st.video(
                video_bytes,
                format="video/mp4"
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


                for name, confidence in defects.items():

                    st.write(
                        f"🔴 **Defect:** {name}"
                    )

                    st.write(
                        f"📊 **Confidence:** "
                        f"{confidence * 100:.2f}%"
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


        # ==================================================
        # EMPTY RESULT
        # ==================================================

        else:

            with st.container(border=True):

                st.markdown(
                    """
                    <div style="
                    height:350px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:18px;
                    ">
                    ANALYZED IMAGE / VIDEO
                    </div>
                    """,
                    unsafe_allow_html=True
                )
