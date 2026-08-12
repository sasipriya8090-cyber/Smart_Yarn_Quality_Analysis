import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import tempfile
import os
import subprocess
import imageio_ffmpeg


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
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1250px;
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
}

/* ================= TITLE ================= */

.yarnx-title {
    text-align: center;
    color: #17365d;
    font-size: 38px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 5px;
    line-height: 1.2;
}

.yarnx-subtitle {
    text-align: center;
    color: #667085;
    font-size: 16px;
    margin-bottom: 30px;
}


/* ================= CARDS ================= */

.info-card {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
}

.card-title {
    color: #17365d;
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 16px;
}

.card-text {
    color: #475467;
    font-size: 15px;
    line-height: 1.75;
}


/* ================= SECTION TITLE ================= */

.section-title {
    color: #17365d;
    font-size: 25px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 18px;
}


/* ================= WAITING ================= */

.waiting-box {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 18px;
    padding: 55px 20px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
}

.waiting-icon {
    font-size: 42px;
    margin-bottom: 10px;
}

.waiting-title {
    color: #667085;
    font-size: 21px;
    font-weight: 800;
}


/* ================= PREVIEW ================= */

.preview-card {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 18px;
    padding: 15px;
    margin-top: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}


/* ================= RESULT ================= */

.result-card {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
}

.result-title {
    color: #17365d;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 15px;
}


/* ================= BUTTON ================= */

.stButton > button {
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    font-size: 16px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #667085;
    margin-top: 35px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE FUNCTION
# ============================================================

def show_title():

    st.markdown(
        '<div class="yarnx-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="yarnx-subtitle">AI-Powered Smart Yarn Quality Inspection</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FIND MODEL
# ============================================================

def find_model():

    # First try known filenames
    known_names = [
        "best (6).pt",
        "best.pt",
        "best (5).pt",
        "best (4).pt",
        "best (3).pt",
        "best (2).pt"
    ]

    for name in known_names:

        path = BASE_DIR / name

        if path.exists():

            try:

                if path.stat().st_size > 1000000:
                    return path

            except:
                pass


    # Search all .pt files automatically
    pt_files = list(BASE_DIR.rglob("*.pt"))

    valid_files = []

    for path in pt_files:

        try:

            size = path.stat().st_size

            if size > 1000000:
                valid_files.append((path, size))

        except:
            pass


    if valid_files:

        # Largest valid model
        valid_files.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return valid_files[0][0]


    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = find_model()

    if model_path is None:

        return None, None, "No .pt model file found."

    try:

        model = YOLO(str(model_path))

        return model, model_path, None

    except Exception as e:

        return None, model_path, str(e)


model, model_path, model_error = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "result_image" not in st.session_state:
    st.session_state.result_image = None

if "result_video" not in st.session_state:
    st.session_state.result_video = None

if "result_text" not in st.session_state:
    st.session_state.result_text = None


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    show_title()

    col1, col2 = st.columns(
        [1, 2],
        gap="large"
    )


    # ========================================================
    # AICW CARD
    # ========================================================

    with col1:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        AI Career for Women (AICW)
        </div>

        <div class="card-text">

        <b>Capstone Project</b><br><br>

        YarnX is an AI-powered yarn inspection
        application developed to automatically
        identify visible yarn quality defects.

        <br><br>

        The system combines Artificial Intelligence,
        Deep Learning and Computer Vision to support
        faster and more consistent yarn inspection.

        </div>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # DESCRIPTION
    # ========================================================

    with col2:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        Project Description
        </div>

        <div class="card-text">

        <b>YarnX – The Future of Yarn Inspection</b>
        is an AI-powered smart yarn quality inspection
        system designed to automatically analyze yarn
        samples and identify visible defects.

        <br><br>

        The application uses a trained
        <b>YOLO deep-learning object detection model</b>
        together with computer vision techniques to
        detect yarn defects from images, camera
        captures and video input.

        <br><br>

        The system can identify defects such as
        <b>loop fiber</b> and <b>protruding fiber</b>
        and displays the detected defect with
        bounding boxes and confidence scores.

        <br><br>

        By reducing manual inspection effort,
        YarnX can help improve inspection speed,
        consistency and quality monitoring in
        textile and yarn manufacturing environments.

        </div>

        </div>
        """, unsafe_allow_html=True)


    st.write("")


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            st.session_state.page = 2
            st.session_state.result_image = None
            st.session_state.result_video = None
            st.session_state.result_text = None

            st.rerun()


    st.write("")


    # ========================================================
    # TEAM
    # ========================================================

    t1, t2, t3 = st.columns(
        3,
        gap="large"
    )


    with t1:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        TEAM MEMBERS
        </div>

        <div class="card-text">

        1. Gutti.pavani devi Priya<br><br>

        2. Somasani.sasi priya<br><br>

        3. Galidevara.Rama Devi<br><br>

        4. Rambala.Harshitha sai Lakshmi

        </div>

        </div>
        """, unsafe_allow_html=True)


    with t2:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        GMAIL
        </div>

        <div class="card-text">

        gutthipavanidevipriya@gmail.com<br><br>

        Sasipriya8090@gmail.com<br><br>

        ramadevigalidevara0@gmail.com<br><br>

        harshitharambala3@gmail.com

        </div>

        </div>
        """, unsafe_allow_html=True)


    with t3:

        st.markdown("""
        <div class="info-card">

        <div class="card-title">
        GUIDE NAME
        </div>

        <div class="card-text">

        <b>Md. Abdul Aziz</b><br><br>

        <b>Designation</b><br><br>

        Co Lead & Trainer AICW

        </div>

        </div>
        """, unsafe_allow_html=True)


    st.markdown(
        '<div class="footer">YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2
# ============================================================

else:

    show_title()


    # ========================================================
    # BACK
    # ========================================================

    if st.button(
        "⬅️ Back to Home",
        use_container_width=False
    ):

        st.session_state.page = 1

        st.session_state.result_image = None
        st.session_state.result_video = None
        st.session_state.result_text = None

        st.rerun()


    st.write("")


    # ========================================================
    # TWO COLUMNS
    # ========================================================

    left, right = st.columns(
        [1, 1.25],
        gap="large"
    )


    # ========================================================
    # INPUT SIDE
    # ========================================================

    with left:

        st.markdown(
            '<div class="section-title">📥 INPUT</div>',
            unsafe_allow_html=True
        )


        input_type = st.radio(
            "Select Input Type",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ],
            horizontal=True
        )


        uploaded_file = None


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


            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")


                st.markdown(
                    '<div class="preview-card"><b>Image Preview</b></div>',
                    unsafe_allow_html=True
                )


                # SMALL IMAGE
                st.image(
                    image,
                    width=380
                )


        # ====================================================
        # CAMERA
        # ====================================================

        elif input_type == "📷 Camera":

            st.write(
                "Capture a yarn image using your camera:"
            )


            uploaded_file = st.camera_input(
                "📷 Take Yarn Photo"
            )


            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")


                st.markdown(
                    '<div class="preview-card"><b>Camera Preview</b></div>',
                    unsafe_allow_html=True
                )


                st.image(
                    image,
                    width=380
                )


        # ====================================================
        # VIDEO
        # ====================================================

        else:

            uploaded_file = st.file_uploader(
                "Upload Yarn Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                    "webm"
                ],
                key="video_upload"
            )


            if uploaded_file:

                st.markdown(
                    '<div class="preview-card"><b>Video Preview</b></div>',
                    unsafe_allow_html=True
                )


                # Small video width
                st.video(
                    uploaded_file.getvalue()
                )


        st.write("")


        analyze = st.button(
            "🔍 Analyze Image / Video",
            use_container_width=True
        )


    # ========================================================
    # RESULT SIDE
    # ========================================================

    with right:

        st.markdown(
            '<div class="section-title">🤖 INSPECTION RESULT</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # WAITING
        # ====================================================

        if (
            st.session_state.result_image is None
            and
            st.session_state.result_video is None
        ):

            st.markdown("""
            <div class="waiting-box">

                <div class="waiting-icon">
                    ⏳
                </div>

                <div class="waiting-title">
                    WAITING FOR ANALYSIS
                </div>

            </div>
            """, unsafe_allow_html=True)


        # ====================================================
        # IMAGE RESULT
        # ====================================================

        if st.session_state.result_image is not None:

            st.markdown("""
            <div class="result-card">

            <div class="result-title">
            🔎 Inspection Result
            </div>

            </div>
            """, unsafe_allow_html=True)


            st.image(
                st.session_state.result_image,
                width=420
            )


            if st.session_state.result_text:

                if "No yarn defect" in st.session_state.result_text:

                    st.success(
                        st.session_state.result_text
                    )

                else:

                    st.error(
                        st.session_state.result_text
                    )


        # ====================================================
        # VIDEO RESULT
        # ====================================================

        if st.session_state.result_video is not None:

            st.markdown("""
            <div class="result-card">

            <div class="result-title">
            🎥 Video Inspection Result
            </div>

            </div>
            """, unsafe_allow_html=True)


            st.video(
                st.session_state.result_video
            )


            if st.session_state.result_text:

                if "No yarn defect" in st.session_state.result_text:

                    st.success(
                        st.session_state.result_text
                    )

                else:

                    st.error(
                        st.session_state.result_text
                    )


# ============================================================
# ANALYSIS BUTTON
# ============================================================

if (
    st.session_state.page == 2
    and
    "analyze" in locals()
    and
    analyze
):


    # ========================================================
    # NO INPUT
    # ========================================================

    if uploaded_file is None:

        st.warning(
            "Please upload an image, capture an image, or upload a video first."
        )

        st.stop()


    # ========================================================
    # MODEL ERROR
    # ========================================================

    if model is None:

        st.error(
            "Model could not be loaded."
        )

        if model_path:

            st.warning(
                f"Model found at: {model_path}"
            )

        if model_error:

            st.code(
                model_error
            )

        st.stop()


    # ========================================================
    # IMAGE / CAMERA
    # ========================================================

    if input_type in [
        "🖼️ Image",
        "📷 Camera"
    ]:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            # YOLO
            results = model.predict(
                source=np.array(image),
                conf=0.20,
                iou=0.45,
                imgsz=640,
                verbose=False
            )


            result = results[0]


            # Draw bounding boxes
            plotted = result.plot()


            plotted = cv2.cvtColor(
                plotted,
                cv2.COLOR_BGR2RGB
            )


            st.session_state.result_image = plotted


            names = model.names

            detections = []


            if result.boxes is not None:

                for cls, conf in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist()
                ):

                    class_name = names[
                        int(cls)
                    ]

                    detections.append(
                        f"{class_name} – {conf * 100:.1f}%"
                    )


            if detections:

                st.session_state.result_text = (
                    "Defect detected: "
                    +
                    ", ".join(detections)
                )

            else:

                st.session_state.result_text = (
                    "No yarn defect detected. Good Quality."
                )


            st.rerun()


        except Exception as e:

            st.error(
                "Image analysis failed."
            )

            st.exception(e)


    # ========================================================
    # VIDEO
    # ========================================================

    elif input_type == "🎥 Video":

        input_path = None
        avi_path = None
        output_path = None

        try:

            # ------------------------------------------------
            # Save uploaded video
            # ------------------------------------------------

            suffix = Path(
                uploaded_file.name
            ).suffix


            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )


            temp_input.write(
                uploaded_file.getbuffer()
            )


            temp_input.close()


            input_path = temp_input.name


            # ------------------------------------------------
            # Open video
            # ------------------------------------------------

            cap = cv2.VideoCapture(
                input_path
            )


            if not cap.isOpened():

                st.error(
                    "Video could not be opened."
                )

                st.stop()


            fps = cap.get(
                cv2.CAP_PROP_FPS
            )


            if not fps or fps <= 0:

                fps = 20


            original_width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )


            original_height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )


            # ------------------------------------------------
            # Keep result video small
            # ------------------------------------------------

            MAX_WIDTH = 640


            if original_width > MAX_WIDTH:

                scale = (
                    MAX_WIDTH /
                    original_width
                )

                width = MAX_WIDTH

                height = int(
                    original_height *
                    scale
                )

            else:

                width = original_width
                height = original_height


            width -= width % 2
            height -= height % 2


            # ------------------------------------------------
            # Temporary AVI
            # ------------------------------------------------

            temp_avi = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".avi"
            )

            temp_avi.close()

            avi_path = temp_avi.name


            fourcc = cv2.VideoWriter_fourcc(
                *"MJPG"
            )


            writer = cv2.VideoWriter(
                avi_path,
                fourcc,
                fps,
                (width, height)
            )


            detected_classes = set()


            # ------------------------------------------------
            # Process video
            # ------------------------------------------------

            while True:

                ret, frame = cap.read()


                if not ret:
                    break


                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )


                results = model.predict(
                    source=frame,
                    conf=0.20,
                    iou=0.45,
                    imgsz=640,
                    verbose=False
                )


                result = results[0]


                if result.boxes is not None:

                    for cls in result.boxes.cls.tolist():

                        detected_classes.add(
                            model.names[int(cls)]
                        )


                annotated = result.plot()


                writer.write(
                    annotated
                )


            cap.release()
            writer.release()


            # ------------------------------------------------
            # Convert to MP4
            # ------------------------------------------------

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


            temp_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_output.close()

            output_path = temp_output.name


            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    avi_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "30",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    output_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )


            with open(
                output_path,
                "rb"
            ) as f:

                video_data = f.read()


            st.session_state.result_video = video_data


            if detected_classes:

                st.session_state.result_text = (
                    "Defects detected: "
                    +
                    ", ".join(
                        sorted(detected_classes)
                    )
                )

            else:

                st.session_state.result_text = (
                    "No yarn defect detected. Good Quality."
                )


            # Cleanup

            for file_path in [
                input_path,
                avi_path,
                output_path
            ]:

                try:

                    if file_path:
                        os.remove(file_path)

                except:

                    pass


            st.rerun()


        except Exception as e:

            st.error(
                "Video analysis failed."
            )

            st.exception(e)

            try:
                cap.release()
            except:
                pass
