import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np
import tempfile
import os
import subprocess


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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f7f9fc 0%, #eef3fa 100%);
    }

    /* Remove unnecessary top space */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        color: #173b70;
        margin-top: 5px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #61708a;
        margin-bottom: 28px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 6px 22px rgba(30, 60, 100, 0.10);
        border: 1px solid #e4eaf2;
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 22px;
        font-weight: 750;
        color: #173b70;
        margin-bottom: 18px;
    }

    /* First page */
    .aicw-title {
        font-size: 30px;
        font-weight: 800;
        color: #173b70;
        margin-top: 25px;
    }

    .capstone {
        font-size: 22px;
        font-weight: 700;
        color: #263b59;
        margin-top: 35px;
    }

    .description-title {
        font-size: 26px;
        font-weight: 800;
        color: #173b70;
    }

    .description {
        font-size: 17px;
        line-height: 1.7;
        color: #4b5c73;
    }

    /* Team cards */
    .team-box {
        background: white;
        border-radius: 18px;
        padding: 22px;
        min-height: 270px;
        box-shadow: 0 6px 22px rgba(30, 60, 100, 0.09);
        border: 1px solid #e4eaf2;
    }

    .team-heading {
        font-size: 18px;
        font-weight: 800;
        color: #173b70;
        margin-bottom: 18px;
    }

    .team-text {
        font-size: 15px;
        line-height: 2;
        color: #344861;
    }

    /* Result boxes */
    .good-result {
        background: #e9f8ef;
        border: 1px solid #b9e8c9;
        border-radius: 16px;
        padding: 22px;
        margin-top: 10px;
    }

    .bad-result {
        background: #fff0f0;
        border: 1px solid #f2bcbc;
        border-radius: 16px;
        padding: 22px;
        margin-top: 10px;
    }

    .good-title {
        color: #168542;
        font-size: 28px;
        font-weight: 800;
    }

    .bad-title {
        color: #d62828;
        font-size: 28px;
        font-weight: 800;
    }

    .result-text {
        color: #35465d;
        font-size: 17px;
        margin-top: 8px;
    }

    .waiting {
        background: #f3f6fa;
        border: 1px solid #dfe6ef;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        color: #61708a;
        font-size: 18px;
    }

    /* Preview */
    .preview-title {
        font-size: 16px;
        font-weight: 700;
        color: #52657f;
        margin-bottom: 8px;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #cbd6e5;
        min-height: 45px;
        font-weight: 700;
        font-size: 16px;
    }

    div.stButton > button:hover {
        border-color: #315f9e;
        color: #173b70;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 14px;
        margin-top: 25px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Streamlit Cloud repository lo best.pt ekkada unna search chestundi
MODEL_CANDIDATES = [
    BASE_DIR / "best.pt",
    BASE_DIR / "weights" / "best.pt",
    BASE_DIR / "yarn_model_100ep" / "weights" / "best.pt",
    BASE_DIR / "yarn_model-3" / "weights" / "best.pt",
]

MODEL_PATH = None

for candidate in MODEL_CANDIDATES:
    if candidate.exists():
        MODEL_PATH = candidate
        break


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "input_type" not in st.session_state:
    st.session_state.input_type = "Image"


# ============================================================
# FIRST PAGE
# ============================================================

def show_home_page():

    st.markdown(
        '<div class="main-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-Powered Smart Yarn Quality Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    left, right = st.columns([1, 2], gap="large")

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="card">'
            '<div class="aicw-title">AI Career for Women (AICW)</div>'
            '<div class="capstone">Capstone Project</div>'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button("🔍  PREDICT", key="predict_button"):
            st.session_state.page = "inspection"
            st.rerun()

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="card">'
            '<div class="description-title">Project Description</div>'
            '<hr>'
            '<div class="description">'
            'Yarn quality inspection system using AI and computer vision '
            'to identify yarn defects from images, camera input, and videos.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # TEAM
    # --------------------------------------------------------

    team_col, gmail_col, guide_col = st.columns([1.4, 1.3, 0.9], gap="medium")

    with team_col:

        st.markdown("""
        <div class="team-box">
            <div class="team-heading">TEAM MEMBERS</div>
            <div class="team-text">
                1. Gutti.pavani devi Priya<br><br>
                2. Somasani.sasi priya<br><br>
                3. Galidevara.Rama Devi<br><br>
                4. Rambala.Harshitha sai Lakshmi
            </div>
        </div>
        """, unsafe_allow_html=True)

    with gmail_col:

        st.markdown("""
        <div class="team-box">
            <div class="team-heading">GMAIL</div>
            <div class="team-text">
                gutthipavanidevipriya@gmail.com<br><br>
                Sasipriya8090@gmail.com<br><br>
                ramadevigalidevara0gmail.com<br><br>
                harshitharambala3@gmail.com
            </div>
        </div>
        """, unsafe_allow_html=True)

    with guide_col:

        st.markdown("""
        <div class="team-box">
            <div class="team-heading">GUIDE NAME</div>
            <div class="team-text">
                <b>Md.Abdul Aziz</b><br><br>
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
# HELPER: DETECTION INFORMATION
# ============================================================

def get_detection_info(result):

    detections = []

    if result.boxes is None:
        return detections

    if len(result.boxes) == 0:
        return detections

    names = result.names

    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    for cls_id, conf in zip(classes, confidences):

        if isinstance(names, dict):
            label = names.get(int(cls_id), str(cls_id))
        else:
            label = names[int(cls_id)]

        detections.append({
            "label": str(label).replace("_", " ").title(),
            "confidence": float(conf)
        })

    return detections


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(model, image):

    image_array = np.array(image)

    results = model.predict(
        source=image_array,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    detections = get_detection_info(result)

    annotated_bgr = result.plot(
        conf=True,
        labels=True,
        boxes=True,
        line_width=2
    )

    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB
    )

    if detections:

        max_conf = max(
            item["confidence"] for item in detections
        )

        defect_names = []

        for item in detections:
            if item["label"] not in defect_names:
                defect_names.append(item["label"])

        return {
            "quality": "BAD",
            "defects": defect_names,
            "confidence": max_conf * 100,
            "annotated": annotated_rgb
        }

    else:

        return {
            "quality": "GOOD",
            "defects": [],
            "confidence": 0,
            "annotated": annotated_rgb
        }


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(model, input_video_path):

    cap = cv2.VideoCapture(str(input_video_path))

    if not cap.isOpened():
        raise RuntimeError("Video could not be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    # Keep video manageable
    max_width = 960

    if width > max_width:
        new_width = max_width
        new_height = int(height * new_width / width)
    else:
        new_width = width
        new_height = height

    temp_dir = Path(tempfile.mkdtemp())

    raw_video = temp_dir / "annotated_raw.avi"
    final_video = temp_dir / "annotated_video.mp4"

    # MJPG AVI is more reliable for OpenCV writing
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        str(raw_video),
        fourcc,
        fps,
        (new_width, new_height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    all_defects = {}
    max_confidence = 0.0
    total_frames = 0
    detected_frames = 0

    progress = st.progress(0)
    status = st.empty()

    # Try to estimate frame count
    total_input_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        # Resize for faster processing
        if frame.shape[1] != new_width or frame.shape[0] != new_height:
            frame = cv2.resize(
                frame,
                (new_width, new_height)
            )

        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        detections = get_detection_info(result)

        if detections:
            detected_frames += 1

            for item in detections:

                label = item["label"]
                conf = item["confidence"]

                all_defects[label] = all_defects.get(
                    label, 0
                ) + 1

                if conf > max_confidence:
                    max_confidence = conf

        annotated = result.plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2
        )

        writer.write(annotated)

        if total_input_frames > 0:

            percent = min(
                int((total_frames / total_input_frames) * 100),
                100
            )

            progress.progress(percent)
            status.text(
                f"Processing video... {percent}%"
            )

    cap.release()
    writer.release()

    progress.empty()
    status.empty()

    # --------------------------------------------------------
    # Convert AVI → MP4 using imageio-ffmpeg
    # --------------------------------------------------------

    try:

        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(raw_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(final_video)
        ]

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:
            raise RuntimeError(
                "Video conversion failed:\n"
                + process.stderr[-1000:]
            )

    except Exception as e:

        # If conversion fails, keep AVI as fallback
        st.warning(
            "MP4 conversion failed. Using fallback video format."
        )

        final_video = raw_video

    if all_defects:

        return {
            "quality": "BAD",
            "defects": list(all_defects.keys()),
            "confidence": max_confidence * 100,
            "video_path": str(final_video),
            "frames": total_frames,
            "detected_frames": detected_frames
        }

    else:

        return {
            "quality": "GOOD",
            "defects": [],
            "confidence": 0,
            "video_path": str(final_video),
            "frames": total_frames,
            "detected_frames": 0
        }


# ============================================================
# RESULT DISPLAY
# ============================================================

def show_result(result, input_type):

    if result is None:

        st.markdown("""
        <div class="waiting">
            ⏳ <b>WAITING FOR ANALYSIS</b><br><br>
            Analyze chesaka result ikkada display avvali.
        </div>
        """, unsafe_allow_html=True)

        return

    if result["quality"] == "GOOD":

        st.markdown("""
        <div class="good-result">
            <div class="good-title">🟢 GOOD QUALITY</div>
            <div class="result-text">
                No yarn defect detected.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        defects_text = ", ".join(result["defects"])

        st.markdown(
            f"""
            <div class="bad-result">
                <div class="bad-title">🔴 BAD QUALITY</div>
                <div class="result-text">
                    <b>Defect Detected</b><br><br>
                    <b>Defect:</b> {defects_text}<br>
                    <b>Confidence:</b> {result["confidence"]:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Annotated IMAGE
    # --------------------------------------------------------

    if input_type in ["Image", "Camera"]:

        st.markdown(
            '<div class="preview-title">🔎 INSPECTED IMAGE</div>',
            unsafe_allow_html=True
        )

        st.image(
            result["annotated"],
            width=480
        )

    # --------------------------------------------------------
    # Annotated VIDEO
    # --------------------------------------------------------

    elif input_type == "Video":

        st.markdown(
            '<div class="preview-title">🎥 PROCESSED VIDEO</div>',
            unsafe_allow_html=True
        )

        video_path = result["video_path"]

        if os.path.exists(video_path):

            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(
                video_bytes,
                format="video/mp4",
                width=480
            )

            if result["quality"] == "BAD":

                st.caption(
                    f"Defect detected in "
                    f"{result['detected_frames']} video frames."
                )


# ============================================================
# SECOND PAGE — INSPECTION
# ============================================================

def show_inspection_page():

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '🧶 YarnX – The Future of Yarn Inspection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Smart Yarn Quality Inspection'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Model check
    # --------------------------------------------------------

    if MODEL_PATH is None:

        st.error("❌ Model could not be loaded.")

        st.warning(
            "Please place best.pt in the GitHub repository "
            "next to app.py."
        )

        st.code("""
smart_yarn_quality_analysis/
│
├── app.py
├── best.pt
└── requirements.txt
        """)

        if st.button("⬅️ Back"):
            st.session_state.page = "home"
            st.rerun()

        return

    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:

        st.error("❌ Model could not be loaded.")
        st.code(str(e))

        st.info(
            "Check that best.pt is a valid YOLO trained model."
        )

        if st.button("⬅️ Back"):
            st.session_state.page = "home"
            st.rerun()

        return

    # --------------------------------------------------------
    # Two columns
    # --------------------------------------------------------

    input_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )

    # ========================================================
    # INPUT
    # ========================================================

    with input_col:

        st.markdown(
            '<div class="card-title">📥 INPUT</div>',
            unsafe_allow_html=True
        )

        input_type = st.radio(
            "Select Input Type:",
            ["Image", "Camera", "Video"],
            horizontal=True,
            key="selected_input"
        )

        st.session_state.input_type = input_type

        uploaded_image = None
        uploaded_video = None
        camera_image = None

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if input_type == "Image":

            uploaded_image = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                key="image_uploader"
            )

            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")

                st.markdown(
                    '<div class="preview-title">'
                    'ORIGINAL IMAGE'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    width=430
                )

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        elif input_type == "Camera":

            camera_image = st.camera_input(
                "Capture Yarn Image"
            )

            if camera_image:

                image = Image.open(
                    camera_image
                ).convert("RGB")

                st.markdown(
                    '<div class="preview-title">'
                    'CAMERA CAPTURE'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    image,
                    width=430
                )

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        elif input_type == "Video":

            uploaded_video = st.file_uploader(
                "Upload Video",
                type=["mp4", "mov", "avi", "mkv"],
                key="video_uploader"
            )

            if uploaded_video:

                st.markdown(
                    '<div class="preview-title">'
                    'ORIGINAL VIDEO'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Small preview
                st.video(
                    uploaded_video,
                    width=430
                )

        st.markdown("<br>", unsafe_allow_html=True)

        analyze_button = st.button(
            "🔍  Analyze Image/Video",
            key="analyze_button"
        )

        # ====================================================
        # ANALYZE
        # ====================================================

        if analyze_button:

            # -----------------------------------------------
            # IMAGE
            # -----------------------------------------------

            if input_type == "Image":

                if uploaded_image is None:

                    st.warning(
                        "Please upload an image first."
                    )

                else:

                    with st.spinner(
                        "AI is inspecting the yarn image..."
                    ):

                        image = Image.open(
                            uploaded_image
                        ).convert("RGB")

                        try:

                            result = analyze_image(
                                model,
                                image
                            )

                            st.session_state.analysis_result = result

                        except Exception as e:

                            st.error(
                                "Image analysis failed."
                            )

                            st.exception(e)

            # -----------------------------------------------
            # CAMERA
            # -----------------------------------------------

            elif input_type == "Camera":

                if camera_image is None:

                    st.warning(
                        "Please capture an image first."
                    )

                else:

                    with st.spinner(
                        "AI is inspecting the camera image..."
                    ):

                        image = Image.open(
                            camera_image
                        ).convert("RGB")

                        try:

                            result = analyze_image(
                                model,
                                image
                            )

                            st.session_state.analysis_result = result

                        except Exception as e:

                            st.error(
                                "Camera analysis failed."
                            )

                            st.exception(e)

            # -----------------------------------------------
            # VIDEO
            # -----------------------------------------------

            elif input_type == "Video":

                if uploaded_video is None:

                    st.warning(
                        "Please upload a video first."
                    )

                else:

                    with st.spinner(
                        "AI is analyzing the video frame by frame..."
                    ):

                        temp_input = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        temp_input.write(
                            uploaded_video.getbuffer()
                        )

                        temp_input.close()

                        try:

                            result = analyze_video(
                                model,
                                temp_input.name
                            )

                            st.session_state.analysis_result = result

                        except Exception as e:

                            st.error(
                                "Video analysis failed."
                            )

                            st.exception(e)

                        finally:

                            try:
                                os.unlink(temp_input.name)
                            except:
                                pass

    # ========================================================
    # RESULT
    # ========================================================

    with result_col:

        st.markdown(
            '<div class="card-title">'
            '🤖 INSPECTION RESULT'
            '</div>',
            unsafe_allow_html=True
        )

        show_result(
            st.session_state.analysis_result,
            st.session_state.input_type
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # BACK
    # --------------------------------------------------------

    if st.button("⬅️ Back to Home", key="back_button"):

        st.session_state.analysis_result = None
        st.session_state.page = "home"
        st.rerun()


# ============================================================
# APP ROUTING
# ============================================================

if st.session_state.page == "home":

    show_home_page()

else:

    show_inspection_page()
