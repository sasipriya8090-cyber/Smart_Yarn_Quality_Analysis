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
# PAGE
# ============================================================

st.set_page_config(
    page_title="YarnX – The Future of Yarn Inspection",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Compact CSS only. No HTML UI is used, so tags cannot appear
# as text inside the app.
st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.block-container {
    max-width: 1250px;
    padding-top: 0.35rem !important;
    padding-bottom: 0.25rem !important;
}
div[data-testid="stVerticalBlock"] { gap: 0.35rem; }
div[data-testid="stHorizontalBlock"] { gap: 0.7rem; }
h1, h2, h3 { color: #17365d; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE - SAME ON BOTH PAGES
# ============================================================

def show_title():
    st.title("🧶 YarnX – The Future of Yarn Inspection")
    st.caption("AI-Powered Smart Yarn Quality Inspection System")

# ============================================================
# MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

def find_model():
    preferred = [
        "best (6).pt",
        "best.pt",
        "best (5).pt",
        "best (4).pt",
        "best (3).pt",
        "best (2).pt",
    ]

    for name in preferred:
        p = BASE_DIR / name
        if p.exists():
            try:
                if p.stat().st_size > 1_000_000:
                    return p
            except Exception:
                pass

    candidates = []
    for p in BASE_DIR.rglob("*.pt"):
        try:
            if p.stat().st_size > 1_000_000:
                candidates.append((p, p.stat().st_size))
        except Exception:
            pass

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


@st.cache_resource
def load_model():
    path = find_model()

    if path is None:
        return None, None, "No trained .pt model was found."

    try:
        return YOLO(str(path)), path, None
    except Exception as e:
        return None, path, str(e)


model, model_path, model_error = load_model()

# ============================================================
# SESSION STATE
# ONE RESULT AT A TIME
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

if "last_input_type" not in st.session_state:
    st.session_state.last_input_type = None


def clear_result():
    st.session_state.result_kind = None
    st.session_state.result_data = None
    st.session_state.quality = None
    st.session_state.defects = []


def save_result(kind, data, quality, defects):
    # Completely replaces the previous result.
    st.session_state.result_kind = kind
    st.session_state.result_data = data
    st.session_state.quality = quality
    st.session_state.defects = defects


# ============================================================
# PAGE 1 - HOME
# ============================================================

if st.session_state.page == 1:

    show_title()

    left, right = st.columns([0.85, 1.75], gap="small")

    with left:
        with st.container(border=True):
            st.subheader("AI Career for Women (AICW)")
            st.write("**Capstone Project**")
            st.write(
                "YarnX is an AI-powered yarn quality inspection "
                "application developed to automatically identify "
                "visible yarn defects."
            )
            st.write(
                "The system combines Artificial Intelligence, "
                "Deep Learning and Computer Vision to make yarn "
                "inspection faster, easier and more consistent."
            )
            st.write(
                "It is designed to support textile, weaving and "
                "yarn manufacturing quality-monitoring applications."
            )

    with right:
        with st.container(border=True):
            st.subheader("Project Description")
            st.write(
                "**YarnX – The Future of Yarn Inspection** is an "
                "AI-powered smart yarn quality inspection system "
                "designed to automatically analyze yarn samples "
                "and identify visible quality defects."
            )
            st.write(
                "The application uses a trained **YOLO deep-learning "
                "object detection model** together with Computer "
                "Vision techniques. It supports inspection through "
                "**images, camera captures and videos**."
            )
            st.write(
                "The trained model identifies defects such as "
                "**loop fiber** and **protruding fiber** and displays "
                "detected regions using bounding boxes and confidence "
                "scores."
            )
            st.write(
                "When a defect is detected, the application reports "
                "**BAD QUALITY / DEFECTIVE** and shows the detected "
                "defect type. When no defect is detected, it reports "
                "**GOOD QUALITY**."
            )
            st.write(
                "YarnX helps reduce manual inspection effort, improve "
                "inspection consistency and support faster yarn-quality "
                "monitoring in textile and manufacturing environments."
            )

    st.write("")

    _, mid, _ = st.columns([1, 0.65, 1])
    with mid:
        if st.button("🔍 PREDICT", use_container_width=True):
            clear_result()
            st.session_state.page = 2
            st.session_state.last_input_type = None
            st.rerun()

    st.write("")

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        with st.container(border=True):
            st.subheader("TEAM MEMBERS")
            st.write(
                "1. Gutti.pavani devi Priya\n\n"
                "2. Somasani.sasi priya\n\n"
                "3. Galidevara.Rama Devi\n\n"
                "4. Rambala.Harshitha sai Lakshmi"
            )

    with c2:
        with st.container(border=True):
            st.subheader("GMAIL")
            st.write(
                "gutthipavanidevipriya@gmail.com\n\n"
                "Sasipriya8090@gmail.com\n\n"
                "ramadevigalidevara0@gmail.com\n\n"
                "harshitharambala3@gmail.com"
            )

    with c3:
        with st.container(border=True):
            st.subheader("GUIDE NAME")
            st.write("**Md. Abdul Aziz**")
            st.write("**Designation:**")
            st.write("Co Lead & Trainer AICW")

    st.caption("YarnX – The Future of Yarn Inspection")


# ============================================================
# PAGE 2 - INSPECTION
# ============================================================

else:

    show_title()

    if st.button("⬅️ Back to Home"):
        clear_result()
        st.session_state.page = 1
        st.session_state.last_input_type = None
        st.rerun()

    st.write("")

    input_col, result_col = st.columns([0.9, 1.1], gap="small")

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    with input_col:

        st.subheader("📥 INPUT")

        input_type = st.radio(
            "Select Input Type",
            ["🖼️ Image", "📷 Camera", "🎥 Video"],
            horizontal=True,
            key="input_type",
        )

        # Clear the old result as soon as the user changes
        # Image / Camera / Video.
        if (
            st.session_state.last_input_type is not None
            and st.session_state.last_input_type != input_type
        ):
            clear_result()

        st.session_state.last_input_type = input_type

        uploaded_file = None

        if input_type == "🖼️ Image":

            uploaded_file = st.file_uploader(
                "Upload Yarn Image",
                type=["jpg", "jpeg", "png", "webp"],
                key="image_upload",
            )

            if uploaded_file:
                preview = Image.open(uploaded_file).convert("RGB")
                st.caption("Image Preview")
                st.image(preview, width=300)

        elif input_type == "📷 Camera":

            st.caption("Capture the yarn sample using your camera.")

            uploaded_file = st.camera_input(
                "📷 Take Yarn Photo",
                key="camera_upload",
            )

            if uploaded_file:
                preview = Image.open(uploaded_file).convert("RGB")
                st.image(preview, width=300)

        else:

            uploaded_file = st.file_uploader(
                "Upload Yarn Video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="video_upload",
            )

            if uploaded_file:
                st.caption("Video Preview")
                st.video(uploaded_file.getvalue())

        analyze = st.button(
            "🔍 Analyze Image / Video",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with result_col:

        st.subheader("🤖 INSPECTION RESULT")

        if st.session_state.result_kind is None:

            st.info("⏳ WAITING FOR ANALYSIS")

        elif st.session_state.result_kind == "image":

            st.markdown("### 🖼️ Image Inspection Result")

            st.image(
                st.session_state.result_data,
                width=430,
            )

            if st.session_state.quality == "bad":
                st.error(
                    "❌ BAD QUALITY / DEFECTIVE\n\n"
                    "Detected defects: "
                    + ", ".join(st.session_state.defects)
                )
            else:
                st.success(
                    "✅ GOOD QUALITY – NO DEFECT DETECTED"
                )

        elif st.session_state.result_kind == "video":

            st.markdown("### 🎥 Video Inspection Result")

            st.video(
                st.session_state.result_data
            )

            if st.session_state.quality == "bad":
                st.error(
                    "❌ BAD QUALITY / DEFECTIVE\n\n"
                    "Detected defects: "
                    + ", ".join(st.session_state.defects)
                )
            else:
                st.success(
                    "✅ GOOD QUALITY – NO DEFECT DETECTED"
                )


# ============================================================
# ANALYSIS
# ============================================================

if (
    st.session_state.page == 2
    and "analyze" in locals()
    and analyze
):

    # Remove any previous result BEFORE starting a new analysis.
    clear_result()

    if uploaded_file is None:
        st.warning(
            "Please upload an image, capture an image, or upload a video first."
        )
        st.stop()

    if model is None:
        st.error("Model could not be loaded.")

        if model_path:
            st.info(f"Model path: {model_path}")

        if model_error:
            st.code(model_error)

        st.stop()

    # ========================================================
    # IMAGE / CAMERA
    # ========================================================

    if input_type in ["🖼️ Image", "📷 Camera"]:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            results = model.predict(
                source=np.array(image),
                conf=0.20,
                iou=0.45,
                imgsz=640,
                verbose=False,
            )

            result = results[0]

            annotated = result.plot()
            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB,
            )

            defects = []

            if result.boxes is not None:
                for cls, conf in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist(),
                ):
                    name = model.names[int(cls)]

                    if name not in defects:
                        defects.append(name)

            if defects:
                save_result(
                    "image",
                    annotated,
                    "bad",
                    defects,
                )
            else:
                save_result(
                    "image",
                    annotated,
                    "good",
                    [],
                )

            st.rerun()

        except Exception as e:
            st.error("Image analysis failed.")
            st.exception(e)

    # ========================================================
    # VIDEO
    # ========================================================

    elif input_type == "🎥 Video":

        input_path = None
        avi_path = None
        output_path = None
        cap = None
        writer = None

        try:

            suffix = Path(uploaded_file.name).suffix or ".mp4"

            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            )

            temp_input.write(
                uploaded_file.getbuffer()
            )
            temp_input.close()

            input_path = temp_input.name

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                raise RuntimeError("Video could not be opened.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 20

            original_width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            original_height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            # Keep the processed video reasonably small.
            max_width = 640

            if original_width > max_width:
                scale = max_width / original_width
                width = max_width
                height = int(original_height * scale)
            else:
                width = original_width
                height = original_height

            width -= width % 2
            height -= height % 2

            if width <= 0 or height <= 0:
                raise RuntimeError("Invalid video dimensions.")

            temp_avi = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".avi",
            )
            temp_avi.close()
            avi_path = temp_avi.name

            writer = cv2.VideoWriter(
                avi_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
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
                        detected_classes.add(
                            model.names[int(cls)]
                        )

                annotated = result.plot()

                if (
                    annotated.shape[1] != width
                    or annotated.shape[0] != height
                ):
                    annotated = cv2.resize(
                        annotated,
                        (width, height),
                    )

                writer.write(annotated)

            cap.release()
            cap = None

            writer.release()
            writer = None

            # Browser-friendly MP4
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

            temp_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
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
                    output_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            with open(output_path, "rb") as f:
                video_data = f.read()

            if detected_classes:
                save_result(
                    "video",
                    video_data,
                    "bad",
                    sorted(detected_classes),
                )
            else:
                save_result(
                    "video",
                    video_data,
                    "good",
                    [],
                )

            for path in [input_path, avi_path, output_path]:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

            st.rerun()

        except Exception as e:

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

            for path in [input_path, avi_path, output_path]:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

            st.error("Video analysis failed.")
            st.exception(e)
