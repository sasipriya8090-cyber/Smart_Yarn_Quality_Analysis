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
    padding-top: 0.25rem !important;
    padding-bottom: 0.35rem !important;
}

.yarnx-title {
    text-align: center;
    color: #17365d;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
    line-height: 1.15;
}

.yarnx-subtitle {
    text-align: center;
    color: #667085;
    font-size: 14px;
    margin: 2px 0 12px 0;
}

.info-box {
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 14px;
    padding: 18px;
    min-height: 255px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

.bottom-box {
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 14px;
    padding: 16px;
    min-height: 180px;
}

.waiting-box {
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 14px;
    min-height: 300px;
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

.result-card {
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 14px;
    padding: 14px;
}

.result-heading {
    color: #17365d;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 8px;
}

.defect-text {
    font-size: 16px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #667085;
    font-size: 12px;
    margin-top: 8px;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE - SAME ON BOTH PAGES
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
# MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_model():
    names = [
        "best (6).pt",
        "best.pt",
        "best (5).pt",
        "best (4).pt",
        "best (3).pt",
        "best (2).pt",
        "best (1).pt",
    ]

    for name in names:
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
    p = find_model()

    if p is None:
        return None, None, "No trained .pt model was found."

    try:
        return YOLO(str(p)), p, None
    except Exception as e:
        return None, p, str(e)


model, model_path, model_error = load_model()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": 1,
    "input_type": "🖼️ Image",
    "uploaded_bytes": None,
    "uploaded_name": None,
    "uploaded_signature": None,
    "result_kind": None,
    "result_data": None,
    "quality": None,
    "defects": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_result():
    st.session_state.result_kind = None
    st.session_state.result_data = None
    st.session_state.quality = None
    st.session_state.defects = []


def clear_input():
    st.session_state.uploaded_bytes = None
    st.session_state.uploaded_name = None
    st.session_state.uploaded_signature = None


def reset_all():
    clear_result()
    clear_input()


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    show_title()

    left, right = st.columns(
        [0.75, 1.55],
        gap="small",
    )

    with left:

        st.markdown('<div class="info-box">', unsafe_allow_html=True)

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
            "It is designed to support quality monitoring "
            "in textile, weaving and yarn-manufacturing environments."
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        if st.button("🔍 PREDICT", use_container_width=True):

            reset_all()
            st.session_state.page = 2
            st.rerun()

    with right:

        st.markdown('<div class="info-box">', unsafe_allow_html=True)

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
            "techniques. It supports yarn inspection through "
            "images, camera input and videos."
        )

        st.write(
            "The trained model identifies defects such as "
            "**loop fiber** and **protruding fiber**. The detected "
            "regions are displayed using bounding boxes and "
            "confidence information."
        )

        st.write(
            "When a defect is detected, the application reports "
            "**BAD QUALITY / DEFECTIVE** and clearly displays the "
            "detected defect name. When no defect is detected, "
            "the application reports **GOOD QUALITY**."
        )

        st.write(
            "The system helps reduce manual inspection effort, "
            "improve inspection consistency and support faster "
            "yarn-quality monitoring in textile and manufacturing "
            "environments."
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    team, gmail, guide = st.columns(
        3,
        gap="small",
    )

    with team:

        st.markdown('<div class="bottom-box">', unsafe_allow_html=True)
        st.subheader("TEAM MEMBERS")
        st.write("1. Gutti.pavani devi Priya")
        st.write("2. Somasani.sasi priya")
        st.write("3. Galidevara.Rama Devi")
        st.write("4. Rambala.Harshitha sai Lakshmi")
        st.markdown("</div>", unsafe_allow_html=True)

    with gmail:

        st.markdown('<div class="bottom-box">', unsafe_allow_html=True)
        st.subheader("GMAIL")
        st.write("gutthipavanidevipriya@gmail.com")
        st.write("Sasipriya8090@gmail.com")
        st.write("ramadevigalidevara0@gmail.com")
        st.write("harshitharambala3@gmail.com")
        st.markdown("</div>", unsafe_allow_html=True)

    with guide:

        st.markdown('<div class="bottom-box">', unsafe_allow_html=True)
        st.subheader("GUIDE NAME")
        st.write("**Md. Abdul Aziz**")
        st.write("**Designation**")
        st.write("Co Lead & Trainer AICW")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2
# ============================================================

else:

    show_title()

    if st.button("⬅️ Back to Home"):
        reset_all()
        st.session_state.page = 1
        st.rerun()

    st.write("")

    input_col, result_col = st.columns(
        [0.85, 1.15],
        gap="small",
    )

    # ========================================================
    # INPUT
    # ========================================================

    with input_col:

        st.subheader("📥 INPUT")

        selected_type = st.radio(
            "Select Input Type",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video",
            ],
            horizontal=True,
            index=[
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video",
            ].index(st.session_state.input_type),
        )

        # If input type changed:
        # clear old result AND old uploaded data.
        if selected_type != st.session_state.input_type:

            clear_result()
            clear_input()

            st.session_state.input_type = selected_type

            # Important: rerun only for switching input mode.
            st.rerun()

        uploaded_file = None

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if selected_type == "🖼️ Image":

            st.write("**Upload Yarn Image**")

            uploaded_file = st.file_uploader(
                "Choose an image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp",
                ],
                key="image_upload",
            )

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        elif selected_type == "📷 Camera":

            st.write("**Camera Input**")

            uploaded_file = st.camera_input(
                "Capture Yarn Image",
                key="camera_upload",
            )

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        else:

            st.write("**Upload Yarn Video**")

            uploaded_file = st.file_uploader(
                "Choose a video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                    "webm",
                ],
                key="video_upload",
            )

        # ----------------------------------------------------
        # STORE UPLOAD IMMEDIATELY
        #
        # THIS IS THE IMPORTANT FIX.
        #
        # Streamlit reruns the script after every button click.
        # We store the uploaded bytes in session_state so the
        # Analyze button never loses the selected file.
        # ----------------------------------------------------

        if uploaded_file is not None:

            current_signature = (
                selected_type,
                uploaded_file.name,
                getattr(
                    uploaded_file,
                    "size",
                    len(uploaded_file.getvalue()),
                ),
            )

            # A different file was selected.
            if (
                st.session_state.uploaded_signature is not None
                and st.session_state.uploaded_signature
                != current_signature
            ):
                clear_result()

            st.session_state.uploaded_signature = (
                current_signature
            )

            st.session_state.uploaded_bytes = (
                uploaded_file.getvalue()
            )

            st.session_state.uploaded_name = (
                uploaded_file.name
            )

        # Preview
        if st.session_state.uploaded_bytes is not None:

            if selected_type in [
                "🖼️ Image",
                "📷 Camera",
            ]:

                try:

                    preview = Image.open(
                        __import__("io").BytesIO(
                            st.session_state.uploaded_bytes
                        )
                    ).convert("RGB")

                    st.image(
                        preview,
                        caption="Selected Yarn Image",
                        width=300,
                    )

                except Exception:
                    pass

            elif selected_type == "🎥 Video":

                st.video(
                    st.session_state.uploaded_bytes
                )

        analyze = st.button(
            "🔍 Analyze Image / Video",
            use_container_width=True,
        )

    # ========================================================
    # RESULT
    # ========================================================

    with result_col:

        st.subheader("🤖 INSPECTION RESULT")

        # Waiting state
        if st.session_state.result_kind is None:

            st.markdown(
                """
                <div class="waiting-box">
                    <div class="waiting-title">
                        ⏳<br><br>
                        WAITING FOR ANALYSIS
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # IMAGE RESULT
        # ----------------------------------------------------

        elif st.session_state.result_kind == "image":

            st.markdown(
                '<div class="result-card">'
                '<div class="result-heading">'
                '🖼️ Image Inspection Result'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.image(
                st.session_state.result_data,
                width=430,
            )

            if st.session_state.quality == "bad":

                st.error(
                    "❌ BAD QUALITY / DEFECTIVE"
                )

                st.markdown(
                    '<div class="defect-text">'
                    "Detected defect: "
                    + ", ".join(
                        st.session_state.defects
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

            else:

                st.success(
                    "✅ GOOD QUALITY – NO DEFECT DETECTED"
                )

        # ----------------------------------------------------
        # VIDEO RESULT
        # ----------------------------------------------------

        elif st.session_state.result_kind == "video":

            st.markdown(
                '<div class="result-card">'
                '<div class="result-heading">'
                '🎥 Video Inspection Result'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Processed video is played here.
            st.video(
                st.session_state.result_data
            )

            if st.session_state.quality == "bad":

                st.error(
                    "❌ BAD QUALITY / DEFECTIVE"
                )

                st.markdown(
                    '<div class="defect-text">'
                    "Detected defect: "
                    + ", ".join(
                        st.session_state.defects
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

            else:

                st.success(
                    "✅ GOOD QUALITY – NO DEFECT DETECTED"
                )


# ============================================================
# 8. ANALYSIS
# ============================================================

if (
    st.session_state.page == 2
    and "analyze" in locals()
    and analyze
):

    # --------------------------------------------------------
    # CHECK STORED FILE
    # --------------------------------------------------------

    if st.session_state.uploaded_bytes is None:

        st.warning(
            "Please select an image, capture an image, "
            "or upload a video first."
        )

        st.stop()

    if model is None:

        st.error("Model could not be loaded.")

        if model_path:
            st.write(f"Model path: {model_path}")

        if model_error:
            st.code(model_error)

        st.stop()

    # ========================================================
    # IMAGE / CAMERA
    # ========================================================

    if selected_type in [
        "🖼️ Image",
        "📷 Camera",
    ]:

        try:

            import io

            image = Image.open(
                io.BytesIO(
                    st.session_state.uploaded_bytes
                )
            ).convert("RGB")

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

                for cls in result.boxes.cls.tolist():

                    class_id = int(cls)

                    name = str(
                        model.names[class_id]
                    )

                    if name not in defects:
                        defects.append(name)

            # Store exactly one result.
            st.session_state.result_kind = "image"
            st.session_state.result_data = annotated

            if defects:

                st.session_state.quality = "bad"
                st.session_state.defects = defects

            else:

                st.session_state.quality = "good"
                st.session_state.defects = []

            # Rerun AFTER storing the result.
            # Uploaded bytes are already saved in session_state,
            # so the file will NOT be lost and upload will NOT
            # be requested again.
            st.rerun()

        except Exception as e:

            st.error("Image analysis failed.")
            st.exception(e)

    # ========================================================
    # VIDEO
    # ========================================================

    elif selected_type == "🎥 Video":

        input_path = None
        avi_path = None
        output_path = None
        cap = None
        writer = None

        try:

            suffix = (
                Path(
                    st.session_state.uploaded_name
                    or "input.mp4"
                ).suffix
                or ".mp4"
            )

            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            )

            temp_input.write(
                st.session_state.uploaded_bytes
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

            # Keep result video reasonably sized.
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

                        class_id = int(cls)

                        name = str(
                            model.names[class_id]
                        )

                        detected_classes.add(
                            name
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

                writer.write(
                    annotated
                )

            cap.release()
            cap = None

            writer.release()
            writer = None

            ffmpeg = (
                imageio_ffmpeg
                .get_ffmpeg_exe()
            )

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

            with open(
                output_path,
                "rb",
            ) as video_file:

                video_data = (
                    video_file.read()
                )

            # Store exactly ONE video result.
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

            # Same fix as image:
            # result is saved first, then rerun.
            # Uploaded video bytes remain in session_state.
            st.rerun()

        except Exception as e:

            st.error("Video analysis failed.")
            st.exception(e)

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
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
