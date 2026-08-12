import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import subprocess
import os

# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.set_page_config(
    page_title="YarnX – The Future of Yarn Inspection",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TITLE = "YarnX – The Future of Yarn Inspection"
CONF = 0.25

# IMPORTANT:
# Put best.pt in the SAME GitHub folder as app.py.
MODEL_LOCATIONS = [
    Path("best (6).pt"),
    Path("weights/best.pt"),
    Path("models/best.pt"),
    Path("yarn_model_100ep/weights/best.pt"),
]

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown("""
<style>
.stApp{
    background:linear-gradient(135deg,#f6f9ff,#edf4ff,#fbfcff);
}
.block-container{
    max-width:1250px;
    padding-top:1.2rem;
    padding-bottom:2rem;
}
.hero{
    text-align:center;
    padding:8px 10px 18px;
}
.hero h1{
    color:#173b73;
    font-size:2.1rem;
    font-weight:800;
    margin:0;
}
.hero p{
    color:#637894;
    margin:6px 0 0;
}
.card{
    background:#fff;
    border:1px solid #dce5f2;
    border-radius:18px;
    padding:22px;
    box-shadow:0 8px 25px rgba(30,65,110,.07);
    height:100%;
}
.heading{
    color:#173b73;
    font-weight:800;
    font-size:1.15rem;
    margin-bottom:14px;
}
.aicw{
    color:#173b73;
    font-size:1.55rem;
    font-weight:800;
    margin-bottom:22px;
}
.capstone{
    color:#334b6b;
    font-size:1.25rem;
    font-weight:700;
    margin-bottom:30px;
}
.desc{
    color:#526782;
    line-height:1.65;
    font-size:1rem;
}
.team{
    color:#405673;
    line-height:2;
}
.guide{
    color:#263f62;
    font-weight:700;
    line-height:1.8;
}
.good{
    background:#edfbf1;
    border:1px solid #a8dfbb;
    border-radius:15px;
    padding:20px;
}
.bad{
    background:#fff0f0;
    border:1px solid #efb0b0;
    border-radius:15px;
    padding:20px;
}
.good-title{
    color:#168044;
    font-size:1.6rem;
    font-weight:900;
}
.bad-title{
    color:#c62828;
    font-size:1.6rem;
    font-weight:900;
}
.wait{
    background:#f7faff;
    border:1px dashed #c9d6e8;
    border-radius:15px;
    padding:40px 15px;
    text-align:center;
    color:#657994;
    min-height:180px;
}
.note{
    color:#71829a;
    font-size:.88rem;
}
div.stButton>button{
    border-radius:12px;
    min-height:44px;
    font-weight:800;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def find_model():
    for p in MODEL_LOCATIONS:
        if p.exists() and p.is_file():
            return p
    for p in Path(".").rglob("best.pt"):
        if ".git" not in p.parts and "venv" not in p.parts and ".venv" not in p.parts:
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_model(path):
    # Do not import cv2 directly.
    from ultralytics import YOLO
    return YOLO(path)


def defect_name(model, cls_id):
    names = model.names
    raw = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
    return {
        "loop_fiber": "Loop Fiber",
        "protruding_fiber": "Protruding Fiber",
    }.get(str(raw), str(raw).replace("_", " ").title())


def image_predict(model, image):
    results = model.predict(
        source=np.array(image),
        conf=CONF,
        verbose=False,
        save=False,
    )
    r = results[0]

    boxes = []
    if r.boxes is not None and len(r.boxes):
        for cls, score in zip(
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        ):
            boxes.append({
                "name": defect_name(model, int(cls)),
                "confidence": float(score),
            })

    # YOLO plot returns RGB when source is RGB.
    annotated = Image.fromarray(r.plot())
    return annotated, boxes


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def process_video(model, uploaded):
    # OpenCV is imported only when a video is actually analyzed.
    import cv2

    suffix = Path(uploaded.name).suffix.lower() or ".mp4"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        source = td / ("input" + suffix)
        raw_out = td / "raw.mp4"
        final_out = td / "processed.mp4"
        source.write_bytes(uploaded.getbuffer())

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError("The uploaded video could not be opened.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1:
            fps = 20.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Could not read video dimensions.")

        # Keep processed video small enough for one-page display.
        max_width = 720
        if width > max_width:
            scale = max_width / width
            out_w = int(width * scale)
            out_h = int(height * scale)
        else:
            out_w, out_h = width, height

        writer = cv2.VideoWriter(
            str(raw_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_w, out_h),
        )

        detections = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Smaller inference image = faster Streamlit Cloud processing.
            if rgb.shape[1] > 960:
                scale = 960 / rgb.shape[1]
                rgb = cv2.resize(
                    rgb,
                    (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            result = model.predict(
                source=rgb,
                conf=CONF,
                verbose=False,
                save=False,
            )[0]

            if result.boxes is not None and len(result.boxes):
                for cls, score in zip(
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                ):
                    detections.append({
                        "name": defect_name(model, int(cls)),
                        "confidence": float(score),
                    })

            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_RGB2BGR)
            annotated = cv2.resize(
                annotated,
                (out_w, out_h),
                interpolation=cv2.INTER_AREA,
            )
            writer.write(annotated)

        cap.release()
        writer.release()

        # Browser-friendly H.264 MP4.
        cmd = [
            ffmpeg_exe(), "-y",
            "-i", str(raw_out),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "27",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(final_out),
        ]
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])

        return final_out.read_bytes(), detections


def show_result(data):
    detections = data["detections"]

    if not detections:
        st.markdown("""
        <div class="good">
            <div class="good-title">🟢 GOOD QUALITY</div>
            <div>No yarn defect detected.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        detections = sorted(
            detections,
            key=lambda x: x["confidence"],
            reverse=True,
        )
        main = detections[0]
        st.markdown(f"""
        <div class="bad">
            <div class="bad-title">🔴 BAD QUALITY</div>
            <div><b>Defect: {main["name"]}</b></div>
            <div>Confidence: {main["confidence"]*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        types = list(dict.fromkeys(d["name"] for d in detections))
        if len(types) > 1:
            st.write("**Detected defects:** " + ", ".join(types))

        if data.get("image") is not None:
            st.write("**Image with defect boxes**")
            st.image(data["image"], width=430)

        if data.get("video") is not None:
            st.write("**Processed video with defect boxes**")
            st.video(data["video"])


# ------------------------------------------------------------
# Page state
# ------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = 1
if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# PAGE 1
# ============================================================
if st.session_state.page == 1:

    st.markdown(f"""
    <div class="hero">
        <h1>🧶 {TITLE}</h1>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("""
        <div class="card">
            <div class="aicw">AI Career for Women (AICW)</div>
            <div class="capstone">Capstone Project</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍  PREDICT", use_container_width=True):
            st.session_state.page = 2
            st.session_state.result = None
            st.rerun()

    with right:
        st.markdown("""
        <div class="card">
            <div class="heading">Project Description</div>
            <div class="desc">
                Yarn quality inspection system using AI and computer vision
                to identify yarn defects from images, camera input, and videos.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.15, 1.15, .85], gap="large")

    with c1:
        st.markdown("""
        <div class="card">
            <div class="heading">TEAM MEMBERS</div>
            <div class="team">
                1. Gutti.pavani devi Priya<br><br>
                2. Somasani.sasi priya<br><br>
                3. Galidevara.Rama Devi<br><br>
                4. Rambala.Harshitha sai Lakshmi
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <div class="heading">GMAIL</div>
            <div class="team">
                gutthipavanidevipriya@gmail.com<br><br>
                Sasipriya8090@gmail.com<br><br>
                ramadevigalidevara0gmail.com<br><br>
                harshitharambala3@gmail.com
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <div class="heading">GUIDE NAME</div>
            <div class="guide">
                Md.Abdul Aziz<br><br>
                <b>Designation</b><br><br>
                Co Lead &amp; Trainer AICW
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 2
# ============================================================
else:

    st.markdown(f"""
    <div class="hero">
        <h1>🧶 {TITLE}</h1>
    </div>
    """, unsafe_allow_html=True)

    top1, top2 = st.columns([3, 1])
    with top1:
        mode = st.radio(
            "Select Input Type",
            ["🖼️ Image", "📷 Camera", "🎥 Video"],
            horizontal=True,
        )
    with top2:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.page = 1
            st.session_state.result = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    input_col, result_col = st.columns(2, gap="large")

    with input_col:
        st.markdown('<div class="heading">📥 INPUT</div>', unsafe_allow_html=True)

        source = None

        if mode == "🖼️ Image":
            source = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png", "webp"],
            )
        elif mode == "📷 Camera":
            source = st.camera_input("Capture yarn image")
        else:
            source = st.file_uploader(
                "Upload video",
                type=["mp4", "mov", "avi", "mkv", "webm"],
            )

        if source is not None:
            if mode in ["🖼️ Image", "📷 Camera"]:
                img = Image.open(source).convert("RGB")
                # Small display: fits on one page.
                st.image(
                    img,
                    caption="Original Image / Camera Capture",
                    width=420,
                )
            else:
                st.video(source)

        analyze = st.button(
            "🔍 Analyze Image" if mode != "🎥 Video" else "🔍 Analyze Video",
            type="primary",
            use_container_width=True,
        )

    with result_col:
        st.markdown(
            '<div class="heading">🤖 INSPECTION RESULT</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.result is None:
            st.markdown("""
            <div class="wait">
                <div style="font-size:2rem">⏳</div>
                <b>WAITING FOR ANALYSIS</b>
                <div class="note">
                    Analyze chesaka result ikkada display avvali.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            show_result(st.session_state.result)

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------
    if analyze:
        if source is None:
            st.warning("First upload an image/video or capture an image.")
            st.stop()

        model_path = find_model()

        if model_path is None:
            st.error("best.pt not found.")
            st.info(
                "GitHub repository lo app.py unna same folder lo "
                "trained best.pt upload cheyyi."
            )
            st.stop()

        try:
            with st.spinner("AI is analyzing..."):
                model = load_model(str(model_path))

                if mode in ["🖼️ Image", "📷 Camera"]:
                    img = Image.open(source).convert("RGB")
                    annotated, detections = image_predict(model, img)

                    st.session_state.result = {
                        "detections": detections,
                        "image": annotated,
                        "video": None,
                    }
                else:
                    video_bytes, detections = process_video(model, source)

                    st.session_state.result = {
                        "detections": detections,
                        "image": None,
                        "video": video_bytes,
                    }

            st.rerun()

        except Exception as e:
            st.error("Analysis failed.")
            st.code(str(e))
