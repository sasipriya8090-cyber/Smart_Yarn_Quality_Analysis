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
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="YarnX – The Future of Yarn Inspection",
    page_icon="🧶",
    layout="wide",
)


# ============================================================
# 2. SIMPLE CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 0.4rem !important;
        padding-bottom: 0.5rem !important;
    }

    .yarn-title {
        text-align: center;
        color: #17365d;
        font-size: 34px;
        font-weight: 800;
        margin: 0 0 2px 0;
    }

    .yarn-subtitle {
        text-align: center;
        color: #667085;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .waiting-box {
        min-height: 260px;
        background: white;
        border: 1px solid #dfe4ea;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .waiting-title {
        color: #475467;
        font-size: 18px;
        font-weight: 700;
    }

    .defect-box {
        background: #fff1f1;
        border: 1px solid #f1b5b5;
        border-radius: 10px;
        padding: 12px;
        margin-top: 8px;
    }

    .good-box {
        background: #ecfdf3;
        border: 1px solid #a7e3bd;
        border-radius: 10px;
        padding: 12px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. COMMON TITLE
#    SAME TITLE ON BOTH PAGES
# ============================================================

def show_title():
    st.markdown(
        '<div class="yarn-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="yarn-subtitle">AI-Powered Smart Yarn Quality Inspection System</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 4. FIND AND LOAD BEST.PT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_model():
    # Your GitHub may contain best (6).pt.
    preferred_names = [
        "best (6).pt",
        "best.pt",
        "best (5).pt",
        "best (4).pt",
        "best (3).pt",
        "best (2).pt",
        "best (1).pt",
    ]

    for name in preferred_names:
        path = BASE_DIR / name
        if path.exists():
            try:
                if path.stat().st_size > 1_000_000:
                    return path
            except Exception:
                pass

    # If the exact name is different, find the largest valid .pt file.
    candidates = []

    for path in BASE_DIR.rglob("*.pt"):
        try:
            size = path.stat().st_size
            if size > 1_000_000:
                candidates.append((path, size))
        except Exception:
            pass

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


@st.cache_resource
def load_model():
    model_path = find_model()

    if model_path is None:
        return None, None, "No trained .pt model was found in the GitHub repository."

    try:
        model = YOLO(str(model_path))
        return model, model_path, None
    except Exception as error:
        return None, model_path, str(error)


model, model_path, model_error = load_model()


# ============================================================
# 5. SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "result_kind" not in st.session_state:
    st.session_state.result_kind = None

if "result_data" not in st.session_state:
    st.session_state.result_data = None

if "quality" not in st.session_state:
    st.session_state.quality = None

if "defects" not in st.session_state:
    st.session_state.defects = []

if "selected_type" not in st.session_state:
    st.session_state.selected_type = None

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None


def clear_result():
    st.session_state.result_kind = None
    st.session_state.result_data = None
    st.session_state.quality = None
    st.session_state.defects = []


def clear_when_input_changes(input_type, uploaded_file):
    """
    Clears the previous result when:
    Image -> Camera
    Camera -> Video
    Video -> Image
    OR a different uploaded file is selected.
    """

    if st.session_state.selected_type != input_type:
        clear_result()
        st.session_state.selected_type = input_type
        st.session_state.file_signature = None

    if uploaded_file is not None:
        signature = (
            input_type,
            uploaded_file.name,
            getattr(uploaded_file, "size", 0),
        )

        if (
            st.session_state.file_signature is not None
            and st.session_state.file_signature != signature
        ):
            clear_result()

        st.session_state.file_signature = signature


# ============================================================
# 6. PAGE 1 - HOME
# ============================================================

if st.session_state.page == 1:

    show_title()

    left, right = st.columns([0.75, 1.55], gap="small")

    # ---------------- LEFT ----------------

    with left:

        st.subheader("AI Career for Women (AICW)")
        st.write("**Capstone Project**")

        st.write(
            "YarnX is an AI-powered yarn quality inspection "
            "application developed to automatically identify "
            "visible yarn defects."
        )

        st.write(
            "The system combines Artificial Intelligence, "
            "Deep Learning and Computer Vision to support "
            "faster, easier and more consistent yarn inspection."
        )

        st.write(
            "It can be used for quality monitoring in textile, "
            "weaving and yarn-manufacturing environments."
        )

        st.write("")

        if st.button("🔍 PREDICT", use_container_width=True):

            clear_result()
            st.session_state.page = 2
            st.session_state.selected_type = None
            st.session_state.file_signature = None

            st.rerun()

    # ---------------- RIGHT ----------------

    with right:

        st.subheader("Project Description")

        st.markdown(
            "**YarnX – The Future of Yarn Inspection**"
        )

        st.write(
            "YarnX is an AI-powered smart yarn quality inspection "
            "system designed to automatically analyze yarn samples "
            "and identify visible quality defects."
        )

        st.write(
            "The application uses a trained YOLO deep-learning "
            "object-detection model together with Computer Vision "
            "techniques. It supports yarn inspection using images, "
            "camera input and videos."
        )

        st.write(
            "The trained model identifies defects such as "
            "**loop fiber** and **protruding fiber**. Detected "
            "regions are displayed with bounding boxes and "
            "confidence information."
        )

        st.write(
            "When a defect is detected, the application reports "
            "**BAD QUALITY / DEFECTIVE** and clearly shows the "
            "detected defect name. When no defect is detected, "
            "the result is reported as **GOOD QUALITY**."
        )

        st.write(
            "The system is intended to reduce manual inspection "
            "effort, improve inspection consistency and support "
            "faster yarn-quality monitoring in textile and "
            "manufacturing environments."
        )

    st.write("")

    # ---------------- BOTTOM THREE BOXES ----------------

    team_col, gmail_col, guide_col = st.columns(3, gap="small")

    with team_col:

        st.subheader("TEAM MEMBERS")

        st.write("1. Gutti.pavani devi Priya")
        st.write("2. Somasani.sasi priya")
        st.write("3. Galidevara.Rama Devi")
        st.write("4. Rambala.Harshitha sai Lakshmi")

    with gmail_col:

        st.subheader("GMAIL")

        st.write("gutthipavanidevipriya@gmail.com")
        st.write("Sasipriya8090@gmail.com")
        st.write("ramadevigalidevara0@gmail.com")
        st.write("harshitharambala3@gmail.com")

    with guide_col:

        st.subheader("GUIDE NAME")

        st.write("**Md. Abdul Aziz**")
        st.write("**Designation**")
        st.write("Co Lead & Trainer AICW")


# ============================================================
# 7. PAGE 2 - INPUT + RESULT
# ============================================================

else:

    # SAME TITLE AGAIN
    show_title()

    if st.button("⬅️ Back to Home"):
        clear_result()
        st.session_state.page = 1
        st.session_state.selected_type = None
        st.session_state.file_signature = None
        st.rerun()

    st.write("")

    input_col, result_col = st.columns(
        [0.85, 1.15],
        gap="small",
    )

    # ========================================================
    # LEFT SIDE - INPUT
    # ========================================================

    with input_col:

        st.subheader("📥 INPUT")

        input_type = st.radio(
            "Select Input Type",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video",
            ],
            horizontal=True,
        )

        uploaded_file = None

        # ---------------- IMAGE ----------------

        if input_type == "🖼️ Image":

            st.write("**Upload Yarn Image**")

            uploaded_file = st.file_uploader(
                "Select image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp",
                ],
                key="image_uploader",
            )

            if uploaded_file is not None:

                preview = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    preview,
                    caption="Selected Yarn Image",
                    width=300,
                )

        # ---------------- CAMERA ----------------

        elif input_type == "📷 Camera":

            st.write("**Camera Input**")

            uploaded_file = st.camera_input(
                "Capture Yarn Image",
                key="camera_input",
            )

        # ---------------- VIDEO ----------------

        else:

            st.write("**Upload Yarn Video**")

            uploaded_file = st.file_uploader(
                "Select video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                    "webm",
                ],
                key="video_uploader",
            )

            if uploaded_file is not None:

                st.video(
                    uploaded_file.getvalue()
                )

        # Clear old result when input changes.
        clear_when_input_changes(
            input_type,
            uploaded_file,
        )

        analyze = st.button(
            "🔍 Analyze Image / Video",
            use_container_width=True,
        )

    # ========================================================
    # RIGHT SIDE - RESULT
    # ========================================================

    with result_col:

        st.subheader("🤖 INSPECTION RESULT")

        if st.session_state.result_kind is None:

            st.markdown(
                """
                <div class="waiting-box">
                    <div class="waiting-title">
                        ⏳<br>
                        WAITING FOR ANALYSIS
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif st.session_state.result_kind == "image":

            st.markdown("### 🖼️ Image Inspection Result")

            # The actual analyzed image is displayed.
            st.image(
                st.session_state.result_data,
                width=430,
            )

            if st.session_state.quality == "bad":

                st.error(
                    "❌ BAD QUALITY / DEFECTIVE"
                )

                st.write(
                    "**Detected defect:** "
                    + ", ".join(
                        st.session_state.defects
                    )
                )

            else:

                st.success(
                    "✅ GOOD QUALITY – NO DEFECT DETECTED"
                )

        elif st.session_state.result_kind == "video":

            st.markdown("### 🎥 Video Inspection Result")

            # The actual processed video is played.
            st.video(
                st.session_state.result_data
            )

            if st.session_state.quality == "bad":

                st.error(
                    "❌ BAD QUALITY / DEFECTIVE"
                )

                st.write(
                    "**Detected defect:** "
                    + ", ".join(
                        st.session_state.defects
                    )
                )

            else:

                st.success(
                    "✅ GOOD QUALITY – NO DEFECT DETECTED"
                )


# ============================================================
# 8. ANALYZE BUTTON
# ============================================================

if (
    st.session_state.page == 2
    and "analyze" in locals()
    and analyze
):

    if uploaded_file is None:

        st.warning(
            "Please select an image, capture an image, "
            "or upload a video first."
        )

        st.stop()

    if model is None:

        st.error(
            "Model could not be loaded."
        )

        if model_path:
            st.write(
                f"Model path: {model_path}"
            )

        if model_error:
            st.code(model_error)

        st.stop()

    # ========================================================
    # IMAGE / CAMERA ANALYSIS
    # ========================================================

    if input_type in [
        "🖼️ Image",
        "📷 Camera",
    ]:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            results = model.predict(
                source=np.array(image),
                conf=0.20,
                iou=0.45,
                imgsz=640,
                verbose=False,
            )

            result = results[0]

            # YOLO annotated image.
            annotated = result.plot()

            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB,
            )

            defects = []

            if result.boxes is not None:

                for cls in result.boxes.cls.tolist():

                    class_id = int(cls)
                    name = str(
                        model.names[class_id]
                    )

                    if name not in defects:
                        defects.append(name)

            if defects:

                st.session_state.result_kind = "image"
                st.session_state.result_data = annotated
                st.session_state.quality = "bad"
                st.session_state.defects = defects

            else:

                st.session_state.result_kind = "image"
                st.session_state.result_data = annotated
                st.session_state.quality = "good"
                st.session_state.defects = []

            # IMPORTANT:
            # NO st.rerun() here.
            # This keeps the uploaded file and result in the
            # SAME execution, so the app does not ask again.

        except Exception as error:

            st.error(
                "Image analysis failed."
            )

            st.exception(error)

    # ========================================================
    # VIDEO ANALYSIS
    # ========================================================

    elif input_type == "🎥 Video":

        input_path = None
        avi_path = None
        output_path = None
        cap = None
        writer = None

        try:

            suffix = (
                Path(uploaded_file.name).suffix
                or ".mp4"
            )

            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            )

            temp_input.write(
                uploaded_file.getbuffer()
            )

            temp_input.close()

            input_path = temp_input.name

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

            # Keep processed video reasonably sized.
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

            if width <= 0 or height <= 0:

                raise RuntimeError(
                    "Invalid video dimensions."
                )

            temp_avi = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".avi",
            )

            temp_avi.close()

            avi_path = temp_avi.name

            writer = cv2.VideoWriter(
                avi_path,
                cv2.VideoWriter_fourcc(
                    *"MJPG"
                ),
                fps,
                (width, height),
            )

            if not writer.isOpened():

                raise RuntimeError(
                    "Could not create processed video."
                )

            detected_classes = set()

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )

                results = model.predict(
                    source=frame,
                    conf=0.20,
                    iou=0.45,
                    imgsz=640,
                    verbose=False,
                )

                result = results[0]

                if result.boxes is not None:

                    for cls in result.boxes.cls.tolist():

                        class_id = int(cls)

                        name = str(
                            model.names[
                                class_id
                            ]
                        )

                        detected_classes.add(
                            name
                        )

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
                        (width, height),
                    )

                writer.write(
                    annotated
                )

            cap.release()
            cap = None

            writer.release()
            writer = None

            # Convert to browser-friendly MP4.
            ffmpeg = (
                imageio_ffmpeg
                .get_ffmpeg_exe()
            )

            temp_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            )

            temp_output.close()

            output_path = (
                temp_output.name
            )

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
                    output_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            with open(
                output_path,
                "rb",
            ) as video_file:

                video_data = (
                    video_file.read()
                )

            st.session_state.result_kind = "video"
            st.session_state.result_data = video_data

            if detected_classes:

                st.session_state.quality = "bad"

                st.session_state.defects = sorted(
                    detected_classes
                )

            else:

                st.session_state.quality = "good"
                st.session_state.defects = []

            # IMPORTANT:
            # NO st.rerun() here.
            # Therefore the selected video remains available
            # during the same execution and the result is shown.

        except Exception as error:

            st.error(
                "Video analysis failed."
            )

            st.exception(error)

        finally:

            if cap is not None:

                try:
                    cap.release()
                except Exception:
                    pass

            if writer is not None:

                try:
                    writer.release()
                except Exception:
                    pass

            for path in [
                input_path,
                avi_path,
                output_path,
            ]:

                try:

                    if (
                        path
                        and os.path.exists(path)
                    ):
                        os.remove(path)

                except Exception:
                    pass
