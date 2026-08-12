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
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1250px;
    padding-top: 0.6rem !important;
    padding-bottom: 0.4rem !important;
}

/* ================= TITLE ================= */

.yarnx-title {
    text-align: center;
    color: #17365d;
    font-size: 34px;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 2px 0;
}

.yarnx-subtitle {
    text-align: center;
    color: #667085;
    font-size: 14px;
    margin: 0 0 12px 0;
}


/* ================= CARDS ================= */

.info-card {
    background: #ffffff;
    border: 1px solid #e4e7ec;
    border-radius: 15px;
    padding: 17px 20px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.card-title {
    color: #17365d;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 9px;
}

.card-text {
    color: #475467;
    font-size: 14px;
    line-height: 1.5;
}


/* ================= SECTION ================= */

.section-title {
    color: #17365d;
    font-size: 23px;
    font-weight: 800;
    margin: 3px 0 8px 0;
}


/* ================= RESULT ================= */

.result-box {
    background: #ffffff;
    border: 1px solid #e4e7ec;
    border-radius: 15px;
    padding: 14px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.result-heading {
    color: #17365d;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 8px;
}


/* ================= STATUS ================= */

.bad-quality {
    background: #fff1f1;
    border: 1px solid #f3b4b4;
    border-radius: 10px;
    padding: 11px 14px;
    color: #b42318;
    font-weight: 700;
    margin-top: 8px;
}

.good-quality {
    background: #ecfdf3;
    border: 1px solid #a6e3c1;
    border-radius: 10px;
    padding: 11px 14px;
    color: #067647;
    font-weight: 700;
    margin-top: 8px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #667085;
    font-size: 12px;
    margin-top: 8px;
}


/* ================= REDUCE GAPS ================= */

div[data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}

div[data-testid="stHorizontalBlock"] {
    gap: 0.7rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# COMMON TITLE
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

    st.markdown(
        """
        <div class="yarnx-subtitle">
        AI-Powered Smart Yarn Quality Inspection System
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FIND TRAINED MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_model():

    # First check the model names that were used in the project

    preferred_names = [
        "best (6).pt",
        "best.pt",
        "best (5).pt",
        "best (4).pt",
        "best (3).pt",
        "best (2).pt"
    ]

    for name in preferred_names:

        path = BASE_DIR / name

        if path.exists():

            try:

                if path.stat().st_size > 1_000_000:
                    return path

            except:
                pass


    # Search repository for any valid .pt model

    models = []

    for path in BASE_DIR.rglob("*.pt"):

        try:

            size = path.stat().st_size

            if size > 1_000_000:

                models.append(
                    (path, size)
                )

        except:
            pass


    if models:

        models.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return models[0][0]


    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = find_model()

    if model_path is None:

        return (
            None,
            None,
            "Trained model (.pt) was not found."
        )


    try:

        model = YOLO(
            str(model_path)
        )

        return (
            model,
            model_path,
            None
        )


    except Exception as e:

        return (
            None,
            model_path,
            str(e)
        )


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


if "quality_status" not in st.session_state:
    st.session_state.quality_status = None


if "last_input_type" not in st.session_state:
    st.session_state.last_input_type = None


# ============================================================
# CLEAR PREVIOUS RESULT
# ============================================================

def clear_results():

    st.session_state.result_image = None

    st.session_state.result_video = None

    st.session_state.result_text = None

    st.session_state.quality_status = None


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    show_title()


    # ========================================================
    # DESCRIPTION AREA
    # ========================================================

    col1, col2 = st.columns(
        [0.85, 1.7],
        gap="small"
    )


    # --------------------------------------------------------
    # AICW
    # --------------------------------------------------------

    with col1:

        st.markdown(
            """
            <div class="info-card">

            <div class="card-title">
            AI Career for Women (AICW)
            </div>

            <div class="card-text">

            <b>Capstone Project</b>

            <br><br>

            YarnX is an AI-powered yarn quality
            inspection application developed to
            automatically identify visible yarn defects.

            <br><br>

            The system combines Artificial Intelligence,
            Deep Learning and Computer Vision to make
            yarn inspection faster, easier and more
            consistent.

            <br><br>

            It is designed to support textile,
            weaving and yarn manufacturing applications.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PROJECT DESCRIPTION
    # --------------------------------------------------------

    with col2:

        st.markdown(
            """
            <div class="info-card">

            <div class="card-title">
            Project Description
            </div>

            <div class="card-text">

            <b>YarnX – The Future of Yarn Inspection</b>
            is an AI-powered smart yarn quality
            inspection system designed to automatically
            analyze yarn samples and identify visible
            quality defects.

            <br><br>

            The application uses a trained
            <b>YOLO deep-learning object detection model</b>
            along with Computer Vision techniques.
            It supports yarn inspection through
            <b>images, camera captures and videos</b>.

            <br><br>

            The trained model can identify yarn defects
            such as <b>loop fiber</b> and
            <b>protruding fiber</b>. Detected defects are
            displayed using bounding boxes and confidence
            information.

            <br><br>

            When defects are detected, the system reports
            the yarn as <b>Bad Quality / Defective</b>
            and clearly displays the detected defect type.
            If no defect is detected, the system reports
            <b>Good Quality</b>.

            <br><br>

            YarnX helps reduce manual inspection effort,
            improve inspection consistency and support
            faster quality monitoring in textile and yarn
            manufacturing environments.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.write("")

    p1, p2, p3 = st.columns(
        [1, 1, 1]
    )

    with p2:

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            clear_results()

            st.session_state.page = 2

            st.rerun()


    st.write("")


    # ========================================================
    # TEAM
    # ========================================================

    team1, team2, team3 = st.columns(
        3,
        gap="small"
    )


    # --------------------------------------------------------
    # TEAM MEMBERS
    # --------------------------------------------------------

    with team1:

        st.markdown(
            """
            <div class="info-card">

            <div class="card-title">
            TEAM MEMBERS
            </div>

            <div class="card-text">

            1. Gutti.pavani devi Priya<br>
            2. Somasani.sasi priya<br>
            3. Galidevara.Rama Devi<br>
            4. Rambala.Harshitha sai Lakshmi

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # GMAIL
    # --------------------------------------------------------

    with team2:

        st.markdown(
            """
            <div class="info-card">

            <div class="card-title">
            GMAIL
            </div>

            <div class="card-text">

            gutthipavanidevipriya@gmail.com<br>
            Sasipriya8090@gmail.com<br>
            ramadevigalidevara0@gmail.com<br>
            harshitharambala3@gmail.com

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # GUIDE
    # --------------------------------------------------------

    with team3:

        st.markdown(
            """
            <div class="info-card">

            <div class="card-title">
            GUIDE NAME
            </div>

            <div class="card-text">

            <b>Md. Abdul Aziz</b>

            <br><br>

            <b>Designation:</b><br>

            Co Lead & Trainer AICW

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="footer">YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2
# ============================================================

else:

    # IMPORTANT:
    # SAME TITLE AGAIN

    show_title()


    # ========================================================
    # BACK BUTTON
    # ========================================================

    if st.button(
        "⬅️ Back to Home"
    ):

        clear_results()

        st.session_state.page = 1

        st.rerun()


    st.write("")


    # ========================================================
    # INPUT + RESULT
    # ========================================================

    input_col, result_col = st.columns(
        [0.9, 1.15],
        gap="small"
    )


    # ========================================================
    # INPUT
    # ========================================================

    with input_col:

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
            horizontal=True,
            key="input_type_radio"
        )


        # ====================================================
        # VERY IMPORTANT
        # CLEAR OLD RESULT WHEN INPUT CHANGES
        # ====================================================

        if (
            st.session_state.last_input_type is not None
            and
            st.session_state.last_input_type != input_type
        ):

            clear_results()


        st.session_state.last_input_type = input_type


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


                st.caption(
                    "Image Preview"
                )


                st.image(
                    image,
                    width=320
                )


        # ====================================================
        # CAMERA
        # ====================================================

        elif input_type == "📷 Camera":

            st.caption(
                "Capture yarn sample using camera"
            )


            uploaded_file = st.camera_input(
                "📷 Take Yarn Photo",
                key="camera_input"
            )


            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")


                st.image(
                    image,
                    width=320
                )


        # ====================================================
        # VIDEO
        # ====================================================

        elif input_type == "🎥 Video":

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

                st.caption(
                    "Video Preview"
                )


                # Small preview
                v1, v2, v3 = st.columns(
                    [0.05, 0.9, 0.05]
                )


                with v2:

                    st.video(
                        uploaded_file.getvalue()
                    )


        st.write("")


        analyze = st.button(
            "🔍 Analyze Image / Video",
            use_container_width=True
        )


    # ========================================================
    # RESULT
    # ========================================================

    with result_col:

        st.markdown(
            '<div class="section-title">🤖 INSPECTION RESULT</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # NOTHING ANALYZED
        # ====================================================

        if (
            st.session_state.result_image is None
            and
            st.session_state.result_video is None
        ):

            st.markdown(
                """
                <div class="result-box"
                     style="text-align:center;
                            padding:55px 10px;">

                <div style="font-size:36px;">
                ⏳
                </div>

                <div style="
                color:#667085;
                font-size:18px;
                font-weight:800;
                margin-top:7px;
                ">

                WAITING FOR ANALYSIS

                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # IMAGE RESULT ONLY
        # ====================================================

        if st.session_state.result_image is not None:

            st.markdown(
                """
                <div class="result-box">

                <div class="result-heading">
                🖼️ Image Inspection Result
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.image(
                st.session_state.result_image,
                width=390
            )


            if st.session_state.quality_status == "bad":

                st.markdown(
                    f"""
                    <div class="bad-quality">
                    ❌ BAD QUALITY / DEFECTIVE<br>
                    {st.session_state.result_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif st.session_state.quality_status == "good":

                st.markdown(
                    """
                    <div class="good-quality">
                    ✅ GOOD QUALITY – NO DEFECT DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # VIDEO RESULT ONLY
        # ====================================================

        elif st.session_state.result_video is not None:

            st.markdown(
                """
                <div class="result-box">

                <div class="result-heading">
                🎥 Video Inspection Result
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            rv1, rv2, rv3 = st.columns(
                [0.03, 0.94, 0.03]
            )


            with rv2:

                st.video(
                    st.session_state.result_video
                )


            if st.session_state.quality_status == "bad":

                st.markdown(
                    f"""
                    <div class="bad-quality">
                    ❌ BAD QUALITY / DEFECTIVE<br>
                    {st.session_state.result_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif st.session_state.quality_status == "good":

                st.markdown(
                    """
                    <div class="good-quality">
                    ✅ GOOD QUALITY – NO DEFECT DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# ANALYZE
# ============================================================

if (
    st.session_state.page == 2
    and
    "analyze" in locals()
    and
    analyze
):

    # ========================================================
    # CLEAR OLD RESULTS FIRST
    # ========================================================

    clear_results()


    # ========================================================
    # INPUT CHECK
    # ========================================================

    if uploaded_file is None:

        st.warning(
            "Please upload an image, capture an image, or upload a video first."
        )

        st.stop()


    # ========================================================
    # MODEL CHECK
    # ========================================================

    if model is None:

        st.error(
            "Model could not be loaded."
        )


        if model_path:

            st.info(
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


            # ------------------------------------------------
            # YOLO PREDICTION
            # ------------------------------------------------

            results = model.predict(
                source=np.array(image),
                conf=0.20,
                iou=0.45,
                imgsz=640,
                verbose=False
            )


            result = results[0]


            # ------------------------------------------------
            # DRAW BOXES
            # ------------------------------------------------

            plotted = result.plot()


            plotted = cv2.cvtColor(
                plotted,
                cv2.COLOR_BGR2RGB
            )


            st.session_state.result_image = plotted


            # ------------------------------------------------
            # DETECT DEFECTS
            # ------------------------------------------------

            detections = []


            if result.boxes is not None:

                for cls, conf in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist()
                ):

                    class_name = model.names[
                        int(cls)
                    ]


                    detections.append(
                        f"{class_name} ({conf * 100:.1f}%)"
                    )


            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            if detections:

                st.session_state.quality_status = "bad"


                st.session_state.result_text = (
                    "Detected defects: "
                    +
                    ", ".join(detections)
                )


            else:

                st.session_state.quality_status = "good"


                st.session_state.result_text = (
                    "No yarn defect detected."
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
            # SAVE UPLOADED VIDEO
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
            # OPEN VIDEO
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


            if fps <= 0:

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
            # REDUCE OUTPUT VIDEO SIZE
            # ------------------------------------------------

            MAX_WIDTH = 640


            if original_width > MAX_WIDTH:

                scale = (
                    MAX_WIDTH /
                    original_width
                )


                width = MAX_WIDTH


                height = int(
                    original_height * scale
                )

            else:

                width = original_width
                height = original_height


            width -= width % 2
            height -= height % 2


            # ------------------------------------------------
            # TEMP AVI
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


            # =================================================
            # PROCESS VIDEO
            # =================================================

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


                # ---------------------------------------------
                # SAVE DEFECT NAMES
                # ---------------------------------------------

                if result.boxes is not None:

                    for cls in result.boxes.cls.tolist():

                        detected_classes.add(
                            model.names[
                                int(cls)
                            ]
                        )


                # ---------------------------------------------
                # DRAW BOXES
                # ---------------------------------------------

                annotated = result.plot()


                writer.write(
                    annotated
                )


            cap.release()
            writer.release()


            # =================================================
            # CONVERT AVI → MP4
            # =================================================

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


            # ------------------------------------------------
            # READ OUTPUT VIDEO
            # ------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as video_file:

                video_data = video_file.read()


            # IMPORTANT:
            # ONLY video result is stored

            st.session_state.result_image = None

            st.session_state.result_video = video_data


            # ------------------------------------------------
            # QUALITY RESULT
            # ------------------------------------------------

            if detected_classes:

                st.session_state.quality_status = "bad"


                st.session_state.result_text = (
                    "Detected defects: "
                    +
                    ", ".join(
                        sorted(
                            detected_classes
                        )
                    )
                )


            else:

                st.session_state.quality_status = "good"


                st.session_state.result_text = (
                    "No yarn defect detected."
                )


            # ------------------------------------------------
            # CLEANUP
            # ------------------------------------------------

            for p in [
                input_path,
                avi_path,
                output_path
            ]:

                try:

                    if p and os.path.exists(p):

                        os.remove(p)

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
