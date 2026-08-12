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
import io


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="YarnX – The Future of Yarn Inspection",
    page_icon="🧶",
    layout="wide"
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
    padding-top: 15px !important;
    padding-bottom: 20px !important;
}

/* TITLE */

.yarn-title {
    text-align: center;
    color: #17365d;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 3px;
}

.yarn-subtitle {
    text-align: center;
    color: #667085;
    font-size: 15px;
    margin-bottom: 18px;
}

/* CARDS */

.card {
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 15px;
    padding: 22px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

/* WAITING */

.waiting {
    height: 300px;
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 15px;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
}

.waiting-text {
    font-size: 19px;
    font-weight: 700;
    color: #475467;
}

/* RESULT */

.result-card {
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 15px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

def show_title():

    st.markdown(
        """
        <div class="yarn-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        <div class="yarn-subtitle">
            AI-Powered Smart Yarn Quality Inspection System
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_model():

    possible_models = [
        "best (6).pt",
        "best.pt",
        "best (5).pt",
        "best (4).pt",
        "best (3).pt",
        "best (2).pt"
    ]

    for name in possible_models:

        path = BASE_DIR / name

        if path.exists():

            try:

                if path.stat().st_size > 1_000_000:
                    return path

            except:
                pass

    # Search all folders
    models = []

    for path in BASE_DIR.rglob("*.pt"):

        try:

            if path.stat().st_size > 1_000_000:
                models.append(path)

        except:
            pass

    if models:

        models.sort(
            key=lambda x: x.stat().st_size,
            reverse=True
        )

        return models[0]

    return None


@st.cache_resource
def load_model():

    model_path = find_model()

    if model_path is None:

        return None, "best.pt model not found."

    try:

        model = YOLO(str(model_path))

        return model, None

    except Exception as e:

        return None, str(e)


model, model_error = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "input_type" not in st.session_state:
    st.session_state.input_type = "Image"

if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "result_type" not in st.session_state:
    st.session_state.result_type = None

if "result_data" not in st.session_state:
    st.session_state.result_data = None

if "defects" not in st.session_state:
    st.session_state.defects = []

if "bad_quality" not in st.session_state:
    st.session_state.bad_quality = False


# ============================================================
# CLEAR RESULT
# ============================================================

def clear_result():

    st.session_state.result_type = None
    st.session_state.result_data = None
    st.session_state.defects = []
    st.session_state.bad_quality = False


def clear_everything():

    clear_result()

    st.session_state.file_bytes = None
    st.session_state.file_name = None


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    show_title()

    left, right = st.columns(
        [0.75, 1.55],
        gap="small"
    )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "AI Career for Women (AICW)"
        )

        st.write(
            "**Capstone Project**"
        )

        st.write(
            "YarnX is an AI-powered yarn quality "
            "inspection application developed to "
            "automatically identify visible yarn defects."
        )

        st.write(
            "The system combines Artificial Intelligence, "
            "Deep Learning and Computer Vision to support "
            "faster and more consistent yarn inspection."
        )

        st.write(
            "It is designed to support quality monitoring "
            "in textile, weaving and yarn-manufacturing "
            "environments."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            clear_everything()

            st.session_state.page = 2

            st.rerun()

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Project Description"
        )

        st.write(
            "**YarnX – The Future of Yarn Inspection**"
        )

        st.write(
            "YarnX is an AI-powered smart yarn quality "
            "inspection system designed to automatically "
            "analyze yarn samples and identify visible "
            "quality defects."
        )

        st.write(
            "The application uses a trained YOLO "
            "deep-learning object-detection model together "
            "with Computer Vision techniques. It supports "
            "yarn inspection using images, camera input "
            "and videos."
        )

        st.write(
            "The trained model can identify defects such "
            "as loop fiber and protruding fiber. When a "
            "defect is detected, the affected region is "
            "highlighted using detection boxes."
        )

        st.write(
            "The system helps reduce manual inspection "
            "effort, improve inspection consistency and "
            "support faster yarn-quality monitoring in "
            "textile and manufacturing environments."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # BOTTOM
    # --------------------------------------------------------

    team, gmail, guide = st.columns(3)

    with team:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("TEAM MEMBERS")

        st.write("1. Gutti.pavani devi Priya")
        st.write("2. Somasani.sasi priya")
        st.write("3. Galidevara.Rama Devi")
        st.write("4. Rambala.Harshitha sai Lakshmi")

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with gmail:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("GMAIL")

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

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with guide:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("GUIDE NAME")

        st.write(
            "**Md. Abdul Aziz**"
        )

        st.write(
            "**Designation**"
        )

        st.write(
            "Co Lead & Trainer AICW"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# PAGE 2
# ============================================================

else:

    show_title()

    if st.button("⬅️ Back to Home"):

        clear_everything()

        st.session_state.page = 1

        st.rerun()

    st.write("")

    input_col, result_col = st.columns(
        [0.85, 1.15],
        gap="small"
    )


    # ========================================================
    # INPUT SIDE
    # ========================================================

    with input_col:

        st.subheader("📥 INPUT")

        selected_type = st.radio(
            "Select Input Type",
            [
                "Image",
                "Camera",
                "Video"
            ],
            horizontal=True,
            index=[
                "Image",
                "Camera",
                "Video"
            ].index(
                st.session_state.input_type
            )
        )

        # ----------------------------------------------------
        # INPUT TYPE CHANGE
        # ----------------------------------------------------

        if selected_type != st.session_state.input_type:

            clear_everything()

            st.session_state.input_type = selected_type

            st.rerun()


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if selected_type == "Image":

            st.write("**Upload Yarn Image**")

            uploaded = st.file_uploader(
                "Upload Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp"
                ],
                key="image_file"
            )


        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        elif selected_type == "Camera":

            st.write("**Camera Capture**")

            uploaded = st.camera_input(
                "Capture Yarn Image",
                key="camera_file"
            )


        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        else:

            st.write("**Upload Yarn Video**")

            uploaded = st.file_uploader(
                "Upload Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                    "webm"
                ],
                key="video_file"
            )


        # ----------------------------------------------------
        # SAVE UPLOADED FILE
        #
        # THIS FIXES:
        # Upload -> Analyze -> Upload Again problem
        # ----------------------------------------------------

        if uploaded is not None:

            st.session_state.file_bytes = (
                uploaded.getvalue()
            )

            st.session_state.file_name = (
                uploaded.name
            )


        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------

        if (
            st.session_state.file_bytes
            is not None
        ):

            if selected_type in [
                "Image",
                "Camera"
            ]:

                try:

                    preview = Image.open(
                        io.BytesIO(
                            st.session_state.file_bytes
                        )
                    ).convert("RGB")

                    st.image(
                        preview,
                        caption="Selected Yarn Image",
                        width=300
                    )

                except:
                    pass

            elif selected_type == "Video":

                st.video(
                    st.session_state.file_bytes
                )


        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        analyze = st.button(
            "🔍 Analyze Image / Video",
            use_container_width=True
        )


    # ========================================================
    # RESULT SIDE
    # ========================================================

    with result_col:

        st.subheader(
            "🤖 INSPECTION RESULT"
        )

        # ----------------------------------------------------
        # WAITING
        # ----------------------------------------------------

        if st.session_state.result_type is None:

            st.markdown(
                """
                <div class="waiting">
                    <div class="waiting-text">
                        ⏳<br><br>
                        WAITING FOR ANALYSIS
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # IMAGE RESULT
        # ====================================================

        elif (
            st.session_state.result_type
            == "image"
        ):

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🖼️ Image Inspection Result"
            )

            st.image(
                st.session_state.result_data,
                width=430
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            # BAD QUALITY ONLY
            if st.session_state.bad_quality:

                st.error(
                    "❌ BAD QUALITY / DEFECTIVE"
                )

                st.warning(
                    "Detected defect: "
                    + ", ".join(
                        st.session_state.defects
                    )
                )

            else:

                st.success(
                    "✅ GOOD QUALITY"
                )


        # ====================================================
        # VIDEO RESULT
        # ====================================================

        elif (
            st.session_state.result_type
            == "video"
        ):

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🎥 Video Inspection Result"
            )

            # PLAY PROCESSED VIDEO
            st.video(
                st.session_state.result_data
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            # BAD QUALITY ONLY
            if st.session_state.bad_quality:

                st.error(
                    "❌ BAD QUALITY / DEFECTIVE"
                )

                st.warning(
                    "Detected defect: "
                    + ", ".join(
                        st.session_state.defects
                    )
                )

            else:

                st.success(
                    "✅ GOOD QUALITY"
                )


# ============================================================
# ANALYZE IMAGE / CAMERA
# ============================================================

if (
    st.session_state.page == 2
    and "analyze" in locals()
    and analyze
):

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if (
        st.session_state.file_bytes
        is None
    ):

        st.warning(
            "Please upload, capture or select an input first."
        )

        st.stop()


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        st.error(
            "Model could not be loaded."
        )

        if model_error:

            st.code(
                model_error
            )

        st.stop()


    # ========================================================
    # IMAGE / CAMERA
    # ========================================================

    if selected_type in [
        "Image",
        "Camera"
    ]:

        try:

            image = Image.open(
                io.BytesIO(
                    st.session_state.file_bytes
                )
            ).convert("RGB")


            results = model.predict(
                source=np.array(image),
                conf=0.20,
                iou=0.45,
                imgsz=640,
                verbose=False
            )


            result = results[0]


            # YOLO annotated image
            annotated = result.plot()

            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )


            # Get defect names
            defects = []

            if result.boxes is not None:

                for cls in result.boxes.cls.tolist():

                    class_id = int(cls)

                    name = str(
                        model.names[class_id]
                    )

                    if name not in defects:

                        defects.append(name)


            # SAVE RESULT BEFORE RERUN
            st.session_state.result_type = "image"

            st.session_state.result_data = annotated

            st.session_state.defects = defects

            if defects:

                st.session_state.bad_quality = True

            else:

                st.session_state.bad_quality = False


            # Because file is already stored in
            # session_state, rerun will NOT ask
            # for upload again.
            st.rerun()


        except Exception as e:

            st.error(
                "Image analysis failed."
            )

            st.exception(e)


    # ========================================================
    # VIDEO
    # ========================================================

    elif selected_type == "Video":

        input_path = None
        avi_path = None
        output_path = None

        cap = None
        writer = None

        try:

            # ------------------------------------------------
            # CREATE TEMP INPUT
            # ------------------------------------------------

            suffix = Path(
                st.session_state.file_name
                or "video.mp4"
            ).suffix

            if not suffix:

                suffix = ".mp4"


            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )

            temp_input.write(
                st.session_state.file_bytes
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

                raise RuntimeError(
                    "Video could not be opened."
                )


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


            # Keep result reasonably sized
            max_width = 640


            if original_width > max_width:

                scale = (
                    max_width
                    / original_width
                )

                width = max_width

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


            writer = cv2.VideoWriter(
                avi_path,
                cv2.VideoWriter_fourcc(
                    *"MJPG"
                ),
                fps,
                (width, height)
            )


            detected = set()


            # ------------------------------------------------
            # PROCESS EVERY FRAME
            # ------------------------------------------------

            while True:

                ret, frame = cap.read()

                if not ret:

                    break


                frame = cv2.resize(
                    frame,
                    (width, height)
                )


                results = model.predict(
                    source=frame,
                    conf=0.20,
                    iou=0.45,
                    imgsz=640,
                    verbose=False
                )


                result = results[0]


                # Defect names
                if result.boxes is not None:

                    for cls in result.boxes.cls.tolist():

                        class_id = int(cls)

                        name = str(
                            model.names[
                                class_id
                            ]
                        )

                        detected.add(
                            name
                        )


                # Draw boxes
                annotated = result.plot()


                if (
                    annotated.shape[1]
                    != width
                    or
                    annotated.shape[0]
                    != height
                ):

                    annotated = cv2.resize(
                        annotated,
                        (width, height)
                    )


                writer.write(
                    annotated
                )


            cap.release()
            cap = None


            writer.release()
            writer = None


            # ------------------------------------------------
            # CONVERT AVI -> MP4
            # ------------------------------------------------

            ffmpeg = (
                imageio_ffmpeg
                .get_ffmpeg_exe()
            )


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
            # READ PROCESSED VIDEO
            # ------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as f:

                video_data = f.read()


            # SAVE RESULT
            st.session_state.result_type = "video"

            st.session_state.result_data = video_data

            st.session_state.defects = sorted(
                detected
            )


            if detected:

                st.session_state.bad_quality = True

            else:

                st.session_state.bad_quality = False


            # Rerun after saving result.
            # Uploaded file is still in session_state.
            st.rerun()


        except Exception as e:

            st.error(
                "Video analysis failed."
            )

            st.exception(e)


        finally:

            if cap is not None:

                try:
                    cap.release()
                except:
                    pass


            if writer is not None:

                try:
                    writer.release()
                except:
                    pass


            for path in [
                input_path,
                avi_path,
                output_path
            ]:

                try:

                    if (
                        path
                        and os.path.exists(path)
                    ):

                        os.remove(path)

                except:
                    pass
