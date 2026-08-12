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
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1250px;
    padding-top: 0.35rem !important;
    padding-bottom: 0.25rem !important;
}

.yarnx-title {
    text-align: center;
    color: #17365d;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
}

.yarnx-subtitle {
    text-align: center;
    color: #667085;
    font-size: 14px;
    margin: 2px 0 10px 0;
}

.info-card,
.result-card {
    background: #ffffff;
    border: 1px solid #e4e7ec;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.card-title,
.section-title {
    color: #17365d;
    font-weight: 800;
}

.card-title {
    font-size: 20px;
    margin-bottom: 8px;
}

.section-title {
    font-size: 22px;
    margin-bottom: 7px;
}

.card-text {
    color: #475467;
    font-size: 14px;
    line-height: 1.48;
}

.result-title {
    color: #17365d;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 7px;
}

.waiting {
    background: #ffffff;
    border: 1px solid #e4e7ec;
    border-radius: 14px;
    min-height: 245px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #667085;
}

.waiting-icon {
    font-size: 34px;
}

.waiting-title {
    font-size: 17px;
    font-weight: 800;
    margin-top: 5px;
}

.good-quality {
    background: #ecfdf3;
    border: 1px solid #a6e3c1;
    border-radius: 10px;
    padding: 10px 13px;
    color: #067647;
    font-weight: 700;
    margin-top: 8px;
}

.bad-quality {
    background: #fff1f1;
    border: 1px solid #f3b4b4;
    border-radius: 10px;
    padding: 10px 13px;
    color: #b42318;
    font-weight: 700;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #667085;
    font-size: 12px;
    margin-top: 8px;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.28rem;
}

div[data-testid="stHorizontalBlock"] {
    gap: 0.65rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TITLE - USED ON BOTH PAGES
# ============================================================

def show_title():
    st.markdown(
        '<div class="yarnx-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="yarnx-subtitle">AI-Powered Smart Yarn Quality Inspection System</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL SEARCH
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
    model_path = find_model()

    if model_path is None:
        return None, None, "Trained .pt model was not found in the repository."

    try:
        model = YOLO(str(model_path))
        return model, model_path, None
    except Exception as e:
        return None, model_path, str(e)


model, model_path, model_error = load_model()


# ============================================================
# ONE RESULT STATE ONLY
# ============================================================

defaults = {
    "page": 1,
    "result_kind": None,       # None / "image" / "video"
    "result_data": None,       # image array or video bytes
    "quality": None,           # None / "good" / "bad"
    "result_text": "",
    "last_input_type": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_result():
    st.session_state.result_kind = None
    st.session_state.result_data = None
    st.session_state.quality = None
    st.session_state.result_text = ""


def set_result(kind, data, quality, text):
    # Replace the previous result completely.
    st.session_state.result_kind = kind
    st.session_state.result_data = data
    st.session_state.quality = quality
    st.session_state.result_text = text


# ============================================================
# PAGE 1 - HOME
# ============================================================

if st.session_state.page == 1:

    show_title()

    left, right = st.columns([0.85, 1.75], gap="small")

    with left:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">AI Career for Women (AICW)</div>
                <div class="card-text">
                    <b>Capstone Project</b><br><br>

                    <b>YarnX</b> is an AI-powered yarn quality
                    inspection application developed to identify
                    visible yarn defects automatically.<br><br>

                    The project combines <b>Artificial Intelligence,
                    Deep Learning and Computer Vision</b> to support
                    faster and more consistent yarn inspection.<br><br>

                    It is designed for yarn, textile, weaving and
                    manufacturing quality-monitoring applications.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">Project Description</div>
                <div class="card-text">
                    <b>YarnX – The Future of Yarn Inspection</b> is an
                    AI-powered smart yarn quality inspection system that
                    automatically analyzes yarn samples and identifies
                    visible defects.<br><br>

                    The application uses a trained <b>YOLO object-detection
                    model</b> with Computer Vision techniques. It supports
                    <b>image, camera and video</b> input for inspection.<br><br>

                    The trained model identifies defects such as
                    <b>loop fiber</b> and <b>protruding fiber</b> and
                    displays detected regions with bounding boxes and
                    confidence scores.<br><br>

                    If defects are detected, the application reports
                    <b>BAD QUALITY / DEFECTIVE</b> and shows the detected
                    defect names. If no defect is detected, it reports
                    <b>GOOD QUALITY</b>.<br><br>

                    YarnX helps reduce manual inspection effort, improve
                    inspection consistency and support faster yarn-quality
                    monitoring in textile and manufacturing environments.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    _, button_col, _ = st.columns([1, 0.65, 1])

    with button_col:
        if st.button("🔍 PREDICT", use_container_width=True):
            clear_result()
            st.session_state.page = 2
            st.session_state.last_input_type = None
            st.rerun()

    st.write("")

    team1, team2, team3 = st.columns(3, gap="small")

    with team1:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">TEAM MEMBERS</div>
                <div class="card-text">
                    1. Gutti.pavani devi Priya<br>
                    2. Somasani.sasi priya<br>
                    3. Galidevara.Rama Devi<br>
                    4. Rambala.Harshitha sai Lakshmi
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with team2:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">GMAIL</div>
                <div class="card-text">
                    gutthipavanidevipriya@gmail.com<br>
                    Sasipriya8090@gmail.com<br>
                    ramadevigalidevara0@gmail.com<br>
                    harshitharambala3@gmail.com
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with team3:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">GUIDE NAME</div>
                <div class="card-text">
                    <b>Md. Abdul Aziz</b><br><br>
                    <b>Designation:</b><br>
                    Co Lead & Trainer AICW
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer">YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2 - INSPECTION
# ============================================================

else:

    # SAME TITLE ON PAGE 2
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

        st.markdown(
            '<div class="section-title">📥 INPUT</div>',
            unsafe_allow_html=True,
        )

        input_type = st.radio(
            "Select Input Type",
            ["🖼️ Image", "📷 Camera", "🎥 Video"],
            horizontal=True,
            key="input_type",
        )

        # IMPORTANT:
        # Every time user changes Image/Camera/Video,
        # the previous result is removed immediately.
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
                key="image_file",
            )

            if uploaded_file:
                preview = Image.open(uploaded_file).convert("RGB")
                st.caption("Image Preview")
                st.image(preview, width=300)

        elif input_type == "📷 Camera":

            st.caption("Capture the yarn sample using your camera.")

            uploaded_file = st.camera_input(
                "📷 Take Yarn Photo",
                key="camera_file",
            )

            if uploaded_file:
                preview = Image.open(uploaded_file).convert("RGB")
                st.image(preview, width=300)

        else:

            uploaded_file = st.file_uploader(
                "Upload Yarn Video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="video_file",
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

        st.markdown(
            '<div class="section-title">🤖 INSPECTION RESULT</div>',
            unsafe_allow_html=True,
        )

        # WAITING STATE
        if st.session_state.result_kind is None:

            st.markdown(
                """
                <div class="waiting">
                    <div class="waiting-icon">⏳</div>
                    <div class="waiting-title">WAITING FOR ANALYSIS</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # IMAGE RESULT
        elif st.session_state.result_kind == "image":

            st.markdown(
                '<div class="result-card"><div class="result-title">🖼️ Image Inspection Result</div></div>',
                unsafe_allow_html=True,
            )

            st.image(
                st.session_state.result_data,
                width=430,
            )

            if st.session_state.quality == "bad":
                st.markdown(
                    f'<div class="bad-quality">❌ BAD QUALITY / DEFECTIVE<br>{st.session_state.result_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="good-quality">✅ GOOD QUALITY – NO DEFECT DETECTED</div>',
                    unsafe_allow_html=True,
                )

        # VIDEO RESULT
        elif st.session_state.result_kind == "video":

            st.markdown(
                '<div class="result-card"><div class="result-title">🎥 Video Inspection Result</div></div>',
                unsafe_allow_html=True,
            )

            st.video(
                st.session_state.result_data,
            )

            if st.session_state.quality == "bad":
                st.markdown(
                    f'<div class="bad-quality">❌ BAD QUALITY / DEFECTIVE<br>{st.session_state.result_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="good-quality">✅ GOOD QUALITY – NO DEFECT DETECTED</div>',
                    unsafe_allow_html=True,
                )


    # ========================================================
    # ANALYSIS
    # ========================================================

    if analyze:

        # Always remove whatever was previously displayed.
        clear_result()

        if uploaded_file is None:
            st.warning(
                "Please upload an image, capture an image, or upload a video first."
            )
            st.stop()

        if model is None:
            st.error("Model could not be loaded.")

            if model_path:
                st.info(f"Model found at: {model_path}")

            if model_error:
                st.code(model_error)

            st.stop()

        # ====================================================
        # IMAGE / CAMERA ANALYSIS
        # ====================================================

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

                # YOLO annotated image
                annotated = result.plot()

                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB,
                )

                detections = []

                if result.boxes is not None:
                    for cls, conf in zip(
                        result.boxes.cls.tolist(),
                        result.boxes.conf.tolist(),
                    ):
                        name = model.names[int(cls)]
                        detections.append(
                            f"{name} ({conf * 100:.1f}%)"
                        )

                if detections:
                    set_result(
                        "image",
                        annotated,
                        "bad",
                        "Detected defects: " + ", ".join(detections),
                    )
                else:
                    set_result(
                        "image",
                        annotated,
                        "good",
                        "No yarn defect detected.",
                    )

                st.rerun()

            except Exception as e:
                st.error("Image analysis failed.")
                st.exception(e)

        # ====================================================
        # VIDEO ANALYSIS
        # ====================================================

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
                temp_input.write(uploaded_file.getbuffer())
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

                # Keep processed video reasonably sized.
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

                fourcc = cv2.VideoWriter_fourcc(*"MJPG")

                writer = cv2.VideoWriter(
                    avi_path,
                    fourcc,
                    fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError("Could not create processed video.")

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

                    # Ensure output frame exactly matches writer size.
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

                # Convert to browser-friendly MP4.
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

                    set_result(
                        "video",
                        video_data,
                        "bad",
                        "Detected defects: "
                        + ", ".join(sorted(detected_classes)),
                    )

                else:

                    set_result(
                        "video",
                        video_data,
                        "good",
                        "No yarn defect detected.",
                    )

                # Clean temporary files only.
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
