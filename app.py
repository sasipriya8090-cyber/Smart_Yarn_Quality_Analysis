import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import io
import tempfile
import os
import subprocess
import imageio_ffmpeg


# ============================================================
# PAGE CONFIGURATION
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
    padding-top: 12px !important;
    padding-bottom: 15px !important;
}

.yarnx-title {
    text-align: center;
    color: #17365d;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 3px;
}

.yarnx-subtitle {
    text-align: center;
    color: #667085;
    font-size: 14px;
    margin-bottom: 12px;
}

.card {
    background: white;
    border: 1px solid #dfe4ea;
    border-radius: 14px;
    padding: 18px;
}

.waiting-box {
    height: 300px;
    background: white;
    border: 1px solid #dfe4ea;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.waiting-text {
    color: #475467;
    font-size: 18px;
    font-weight: 700;
}

.result-title {
    color: #17365d;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# COMMON TITLE
# BOTH PAGES HAVE THE SAME TITLE
# ============================================================

def show_title():

    st.markdown(
        """
        <div class="yarnx-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>

        <div class="yarnx-subtitle">
            AI-Powered Smart Yarn Quality Inspection System
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FIND MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_model():

    model_names = [
        "best (6).pt",
        "best.pt",
        "best (5).pt",
        "best (4).pt",
        "best (3).pt",
        "best (2).pt",
        "best (1).pt"
    ]

    for name in model_names:

        path = BASE_DIR / name

        if path.exists():

            try:

                if path.stat().st_size > 1_000_000:
                    return path

            except:
                pass


    # Search remaining .pt files
    candidates = []

    for path in BASE_DIR.rglob("*.pt"):

        try:

            if path.stat().st_size > 1_000_000:
                candidates.append(path)

        except:
            pass


    if candidates:

        candidates.sort(
            key=lambda x: x.stat().st_size,
            reverse=True
        )

        return candidates[0]


    return None


@st.cache_resource
def load_model():

    path = find_model()

    if path is None:

        return None, None

    try:

        return YOLO(str(path)), None

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
# CLEAR FUNCTIONS
# ============================================================

def clear_result():

    st.session_state.result_type = None
    st.session_state.result_data = None
    st.session_state.defects = []
    st.session_state.bad_quality = False


def clear_all():

    clear_result()

    st.session_state.file_bytes = None
    st.session_state.file_name = None


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    show_title()


    # --------------------------------------------------------
    # TOP SECTION
    # --------------------------------------------------------

    left, right = st.columns(
        [0.75, 1.55],
        gap="small"
    )


    # ========================================================
    # LEFT SIDE
    # EXACTLY AS USER STRUCTURE
    # NO EXTRA MATTER UNDER CAPSTONE
    # ========================================================

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

        st.write("")

        if st.button(
            "PREDICT",
            use_container_width=True
        ):

            clear_all()

            st.session_state.page = 2

            st.rerun()

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # RIGHT SIDE
    # PROJECT DESCRIPTION
    # ========================================================

    with right:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Project Description"
        )

        st.write(
            "Yarn quality inspection system using "
            "Artificial Intelligence and Computer Vision "
            "to identify yarn defects from images, "
            "camera input and videos."
        )

        st.write(
            "The system uses a trained YOLO deep-learning "
            "model to detect visible yarn defects and "
            "highlight the defective regions."
        )

        st.write(
            "YarnX provides an easy way to inspect yarn "
            "samples and identify defects automatically."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    st.write("")


    # ========================================================
    # BOTTOM SECTION
    # ========================================================

    team_col, gmail_col, guide_col = st.columns(
        3,
        gap="small"
    )


    # TEAM MEMBERS

    with team_col:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "TEAM MEMBERS"
        )

        st.write(
            "1. Gutti.pavani devi Priya"
        )

        st.write(
            "2. Somasani.sasi priya"
        )

        st.write(
            "3. Galidevara.Rama Devi"
        )

        st.write(
            "4. Rambala.Harshitha sai Lakshmi"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # GMAIL

    with gmail_col:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "GMAIL"
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

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # GUIDE

    with guide_col:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "GUIDE NAME"
        )

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


    # --------------------------------------------------------
    # BACK
    # --------------------------------------------------------

    if st.button(
        "⬅️ Back to Home"
    ):

        clear_all()

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

        st.subheader(
            "📥 INPUT"
        )


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
        # CHANGE INPUT TYPE
        # ----------------------------------------------------

        if selected_type != st.session_state.input_type:

            clear_all()

            st.session_state.input_type = selected_type

            st.rerun()


        uploaded = None


        # ====================================================
        # IMAGE
        # ====================================================

        if selected_type == "Image":

            st.write(
                "**Upload Yarn Image**"
            )

            uploaded = st.file_uploader(
                "Choose image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp"
                ],
                key="image_uploader"
            )


        # ====================================================
        # CAMERA
        # ====================================================

        elif selected_type == "Camera":

            st.write(
                "**Camera Capture**"
            )

            uploaded = st.camera_input(
                "Capture Yarn Image",
                key="camera_input"
            )


        # ====================================================
        # VIDEO
        # ====================================================

        else:

            st.write(
                "**Upload Yarn Video**"
            )

            uploaded = st.file_uploader(
                "Choose video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                    "webm"
                ],
                key="video_uploader"
            )


        # ====================================================
        # SAVE FILE
        # THIS PREVENTS RE-UPLOAD AFTER ANALYZE
        # ====================================================

        if uploaded is not None:

            st.session_state.file_bytes = (
                uploaded.getvalue()
            )

            st.session_state.file_name = (
                uploaded.name
            )


        # ====================================================
        # PREVIEW
        # ====================================================

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
                        caption="Preview",
                        width=300
                    )

                except:
                    pass


            elif selected_type == "Video":

                st.video(
                    st.session_state.file_bytes
                )


        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

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

                        ⏳

                        <br><br>

                        WAITING FOR ANALYSIS

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # IMAGE RESULT
        # ====================================================

        elif st.session_state.result_type == "image":

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-title">'
                '🖼️ Image Inspection Result'
                '</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # YOLO IMAGE WITH BOXES + LABELS + CONFIDENCE
            # ------------------------------------------------

            st.image(
                st.session_state.result_data,
                width=430
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # BAD QUALITY
            # ------------------------------------------------

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


            # ------------------------------------------------
            # GOOD QUALITY
            # ------------------------------------------------

            else:

                st.success(
                    "✅ GOOD QUALITY"
                )


        # ====================================================
        # VIDEO RESULT
        # ====================================================

        elif st.session_state.result_type == "video":

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-title">'
                '🎥 Video Inspection Result'
                '</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # PROCESSED VIDEO
            # YOLO BOXES + LABELS ARE DRAWN FRAME BY FRAME
            # ------------------------------------------------

            st.video(
                st.session_state.result_data
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # BAD QUALITY
            # ------------------------------------------------

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


            # ------------------------------------------------
            # GOOD QUALITY
            # ------------------------------------------------

            else:

                st.success(
                    "✅ GOOD QUALITY"
                )


# ============================================================
# ANALYZE
# ============================================================

if (
    st.session_state.page == 2
    and "analyze" in locals()
    and analyze
):


    # ========================================================
    # CHECK FILE
    # ========================================================

    if st.session_state.file_bytes is None:

        st.warning(
            "Please upload, capture or select an input first."
        )

        st.stop()


    # ========================================================
    # CHECK MODEL
    # ========================================================

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
    # IMAGE / CAMERA ANALYSIS
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
            # DRAW BOXES + CLASS NAMES + CONFIDENCE
            #
            # result.plot() automatically creates:
            # bounding box
            # defect name
            # confidence
            # ------------------------------------------------

            annotated = result.plot()


            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # GET DEFECT NAMES
            # ------------------------------------------------

            defects = []


            if result.boxes is not None:

                for cls in result.boxes.cls.tolist():

                    class_id = int(cls)

                    name = str(
                        model.names[class_id]
                    )

                    if name not in defects:

                        defects.append(
                            name
                        )


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            st.session_state.result_type = "image"

            st.session_state.result_data = annotated

            st.session_state.defects = defects

            st.session_state.bad_quality = (
                len(defects) > 0
            )


            # IMPORTANT:
            # Uploaded file is already saved in
            # session_state.
            # Therefore rerun will NOT ask for upload again.

            st.rerun()


        except Exception as e:

            st.error(
                "Image analysis failed."
            )

            st.exception(e)


    # ========================================================
    # VIDEO ANALYSIS
    # ========================================================

    elif selected_type == "Video":

        input_path = None
        avi_path = None
        output_path = None

        cap = None
        writer = None


        try:

            # ------------------------------------------------
            # CREATE TEMP VIDEO FILE
            # ------------------------------------------------

            suffix = Path(
                st.session_state.file_name
                or "input.mp4"
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


            # ------------------------------------------------
            # CONTROL OUTPUT SIZE
            # ------------------------------------------------

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


            if not writer.isOpened():

                raise RuntimeError(
                    "Could not create processed video."
                )


            detected = set()


            # =================================================
            # PROCESS EACH VIDEO FRAME
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


                # ------------------------------------------------
                # YOLO
                # ------------------------------------------------

                results = model.predict(
                    source=frame,
                    conf=0.20,
                    iou=0.45,
                    imgsz=640,
                    verbose=False
                )


                result = results[0]


                # ------------------------------------------------
                # DEFECT NAMES
                # ------------------------------------------------

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


                # ------------------------------------------------
                # DRAW BOXES + LABELS + CONFIDENCE
                # ON EVERY FRAME
                # ------------------------------------------------

                annotated = result.plot()


                if (
                    annotated.shape[1] != width
                    or
                    annotated.shape[0] != height
                ):

                    annotated = cv2.resize(
                        annotated,
                        (width, height)
                    )


                writer.write(
                    annotated
                )


            # ------------------------------------------------
            # RELEASE
            # ------------------------------------------------

            cap.release()

            cap = None


            writer.release()

            writer = None


            # =================================================
            # CONVERT AVI TO MP4
            # =================================================

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
            # READ FINAL VIDEO
            # ------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as video_file:

                video_data = video_file.read()


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            st.session_state.result_type = "video"

            st.session_state.result_data = (
                video_data
            )

            st.session_state.defects = sorted(
                detected
            )

            st.session_state.bad_quality = (
                len(detected) > 0
            )


            # Uploaded file is safely stored in
            # session_state, so rerun will not
            # ask for upload again.

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
