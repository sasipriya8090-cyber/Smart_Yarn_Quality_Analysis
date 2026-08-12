import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
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

.stApp {
    background: #f5f7fb;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1.5rem;
    max-width: 1400px;
}

/* PAGE TITLE */

.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 18px;
}

/* SECTION TITLES */

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 12px;
}

/* PAGE 1 */

.aicw-title {
    font-size: 30px;
    font-weight: 800;
    color: #172554;
}

.capstone {
    font-size: 24px;
    font-weight: 700;
    color: #334155;
    margin-top: 25px;
    margin-bottom: 25px;
}

.description-title {
    font-size: 24px;
    font-weight: 800;
    color: #172554;
}

.description {
    font-size: 16px;
    line-height: 1.6;
    color: #475569;
}

/* TEAM */

.team-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 15px;
    min-height: 220px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

/* WAITING */

.waiting {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 15px;
    padding: 45px 15px;
    text-align: center;
    margin-top: 10px;
}

.waiting h3 {
    color: #64748b;
}

/* GOOD */

.good-result {
    background: #ecfdf5;
    border: 2px solid #86efac;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    margin-top: 10px;
}

.good-result h2 {
    color: #15803d;
    font-size: 28px;
}

/* BAD */

.bad-result {
    background: #fef2f2;
    border: 2px solid #fca5a5;
    border-radius: 15px;
    padding: 22px;
    text-align: center;
    margin-top: 10px;
}

.bad-result h2 {
    color: #dc2626;
    font-size: 28px;
}

.defect-box {
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 15px;
    border-radius: 10px;
    margin-top: 12px;
    margin-bottom: 12px;
}

.confidence {
    font-size: 18px;
    font-weight: 700;
    color: #334155;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "result_ready" not in st.session_state:
    st.session_state.result_ready = False

if "result_type" not in st.session_state:
    st.session_state.result_type = None

if "result_data" not in st.session_state:
    st.session_state.result_data = None


# ============================================================
# MODEL
# ============================================================

# IMPORTANT:
# Change this path only if your best.pt is somewhere else.

MODEL_PATH = "/content/drive/MyDrive/yarn_training/yarn_model_100ep/weights/best.pt"

CONF_THRESHOLD = 0.50


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:

    model = load_model()

except Exception as e:

    st.error("❌ Trained model could not be loaded.")

    st.write(
        "Please check the best.pt path."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    st.markdown(
        """
        <div class="main-title">
        🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1, 2],
        gap="large"
    )


    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

    with left:

        st.markdown(
            """
            <div class="aicw-title">
            AI Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="capstone">
            Capstone Project
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            st.session_state.page = 2

            st.session_state.result_ready = False
            st.session_state.result_type = None
            st.session_state.result_data = None

            st.rerun()


    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    with right:

        st.markdown(
            """
            <div class="description-title">
            Project Description
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="description">
            Yarn quality inspection system using AI and computer
            vision to identify yarn defects from images, camera
            input, and videos.
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # --------------------------------------------------------
    # TEAM INFORMATION
    # --------------------------------------------------------

    team_col, gmail_col, guide_col = st.columns(
        [1.5, 1.5, 1],
        gap="medium"
    )


    with team_col:

        st.markdown(
            """
            <div class="team-box">
            <b>TEAM MEMBERS</b>
            <br><br>
            1. Gutti.pavani devi Priya
            <br><br>
            2. Somasani.sasi priya
            <br><br>
            3. Galidevara.Rama Devi
            <br><br>
            4. Rambala.Harshitha sai Lakshmi
            </div>
            """,
            unsafe_allow_html=True
        )


    with gmail_col:

        st.markdown(
            """
            <div class="team-box">
            <b>GMAIL</b>
            <br><br>
            gutthipavanidevipriya@gmail.com
            <br><br>
            Sasipriya8090@gmail.com
            <br><br>
            ramadevigalidevara0gmail.com
            <br><br>
            harshitharambala3@gmail.com
            </div>
            """,
            unsafe_allow_html=True
        )


    with guide_col:

        st.markdown(
            """
            <div class="team-box">
            <b>GUIDE NAME</b>
            <br><br>
            Md.Abdul Aziz
            <br><br>
            <b>Designation</b>
            <br><br>
            Co Lead & Trainer AICW
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <div class="footer">
        YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2
# ============================================================

else:

    st.markdown(
        """
        <div class="main-title">
        🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BACK
    # --------------------------------------------------------

    if st.button("← Back to Project"):

        st.session_state.page = 1

        st.session_state.result_ready = False
        st.session_state.result_type = None
        st.session_state.result_data = None

        st.rerun()


    # --------------------------------------------------------
    # TWO COLUMNS
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
            """
            <div class="section-title">
            📥 INPUT
            </div>
            """,
            unsafe_allow_html=True
        )


        input_type = st.radio(
            "Select Input Type:",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ],
            label_visibility="visible"
        )


        # ====================================================
        # IMAGE
        # ====================================================

        if input_type == "🖼️ Image":

            uploaded_image = st.file_uploader(
                "Upload Image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="image_upload"
            )


            if uploaded_image is not None:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")


                # SMALL IMAGE PREVIEW
                st.image(
                    image,
                    caption="Original Image",
                    width=450
                )


                analyze_image = st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                )


                if analyze_image:

                    with st.spinner(
                        "Analyzing image..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=CONF_THRESHOLD,
                            verbose=False
                        )[0]


                    # GOOD
                    if len(result.boxes) == 0:

                        st.session_state.result_ready = True
                        st.session_state.result_type = "good"
                        st.session_state.result_data = None


                    # BAD
                    else:

                        detections = []


                        for box in result.boxes:

                            class_id = int(
                                box.cls[0]
                            )

                            confidence = float(
                                box.conf[0]
                            )

                            class_name = result.names[
                                class_id
                            ]

                            detections.append(
                                (
                                    class_name,
                                    confidence
                                )
                            )


                        best_defect = max(
                            detections,
                            key=lambda x: x[1]
                        )


                        annotated = result.plot()


                        annotated = cv2.cvtColor(
                            annotated,
                            cv2.COLOR_BGR2RGB
                        )


                        st.session_state.result_ready = True

                        st.session_state.result_type = "bad"

                        st.session_state.result_data = {
                            "defect": best_defect[0],
                            "confidence": best_defect[1],
                            "image": annotated
                        }


                    st.rerun()


        # ====================================================
        # CAMERA
        # ====================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Take a picture of the yarn"
            )


            if camera_image is not None:

                image = Image.open(
                    camera_image
                ).convert("RGB")


                # SMALL CAMERA PREVIEW
                st.image(
                    image,
                    caption="Captured Image",
                    width=450
                )


                analyze_camera = st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                )


                if analyze_camera:

                    with st.spinner(
                        "Analyzing captured image..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=CONF_THRESHOLD,
                            verbose=False
                        )[0]


                    if len(result.boxes) == 0:

                        st.session_state.result_ready = True
                        st.session_state.result_type = "good"
                        st.session_state.result_data = None


                    else:

                        detections = []


                        for box in result.boxes:

                            class_id = int(
                                box.cls[0]
                            )

                            confidence = float(
                                box.conf[0]
                            )

                            class_name = result.names[
                                class_id
                            ]

                            detections.append(
                                (
                                    class_name,
                                    confidence
                                )
                            )


                        best_defect = max(
                            detections,
                            key=lambda x: x[1]
                        )


                        annotated = result.plot()


                        annotated = cv2.cvtColor(
                            annotated,
                            cv2.COLOR_BGR2RGB
                        )


                        st.session_state.result_ready = True

                        st.session_state.result_type = "bad"

                        st.session_state.result_data = {
                            "defect": best_defect[0],
                            "confidence": best_defect[1],
                            "image": annotated
                        }


                    st.rerun()


        # ====================================================
        # VIDEO
        # ====================================================

        elif input_type == "🎥 Video":

            uploaded_video = st.file_uploader(
                "Upload Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv"
                ],
                key="video_upload"
            )


            if uploaded_video is not None:

                # SMALL VIDEO PREVIEW
                st.video(
                    uploaded_video,
                    format="video/mp4"
                )


                analyze_video = st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                )


                if analyze_video:

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        # ------------------------------------
                        # SAVE INPUT
                        # ------------------------------------

                        input_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_file.write(
                            uploaded_video.getbuffer()
                        )

                        input_file.close()


                        # ------------------------------------
                        # OPEN VIDEO
                        # ------------------------------------

                        cap = cv2.VideoCapture(
                            input_file.name
                        )


                        fps = cap.get(
                            cv2.CAP_PROP_FPS
                        )

                        if fps <= 0:
                            fps = 20


                        width = int(
                            cap.get(
                                cv2.CAP_PROP_FRAME_WIDTH
                            )
                        )

                        height = int(
                            cap.get(
                                cv2.CAP_PROP_FRAME_HEIGHT
                            )
                        )


                        total_frames = int(
                            cap.get(
                                cv2.CAP_PROP_FRAME_COUNT
                            )
                        )


                        # ------------------------------------
                        # RAW OUTPUT AVI
                        # ------------------------------------

                        raw_output = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".avi"
                        )

                        raw_output.close()


                        fourcc = cv2.VideoWriter_fourcc(
                            *"MJPG"
                        )


                        writer = cv2.VideoWriter(
                            raw_output.name,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        any_defect = False

                        detected_defects = {}


                        progress = st.progress(0)


                        frame_count = 0


                        # ====================================
                        # PROCESS EVERY FRAME
                        # ====================================

                        while True:

                            ret, frame = cap.read()


                            if not ret:
                                break


                            # Convert BGR -> RGB
                            frame_rgb = cv2.cvtColor(
                                frame,
                                cv2.COLOR_BGR2RGB
                            )


                            result = model.predict(
                                source=frame_rgb,
                                conf=CONF_THRESHOLD,
                                verbose=False
                            )[0]


                            # --------------------------------
                            # DETECT DEFECTS
                            # --------------------------------

                            for box in result.boxes:

                                class_id = int(
                                    box.cls[0]
                                )

                                confidence = float(
                                    box.conf[0]
                                )

                                class_name = result.names[
                                    class_id
                                ]


                                any_defect = True


                                if class_name not in detected_defects:

                                    detected_defects[
                                        class_name
                                    ] = confidence

                                else:

                                    detected_defects[
                                        class_name
                                    ] = max(
                                        detected_defects[
                                            class_name
                                        ],
                                        confidence
                                    )


                            # --------------------------------
                            # DRAW BOUNDING BOXES
                            # --------------------------------

                            annotated = result.plot()


                            # result.plot() gives BGR image
                            writer.write(
                                annotated
                            )


                            frame_count += 1


                            if total_frames > 0:

                                progress.progress(
                                    min(
                                        frame_count /
                                        total_frames,
                                        1.0
                                    )
                                )


                        cap.release()
                        writer.release()

                        progress.empty()


                        # ====================================
                        # GOOD VIDEO
                        # ====================================

                        if not any_defect:

                            st.session_state.result_ready = True

                            st.session_state.result_type = "good"

                            st.session_state.result_data = None


                        # ====================================
                        # BAD VIDEO
                        # ====================================

                        else:

                            best_defect = max(
                                detected_defects,
                                key=detected_defects.get
                            )


                            # --------------------------------
                            # CONVERT AVI -> H264 MP4
                            # --------------------------------

                            final_mp4 = tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".mp4"
                            )

                            final_mp4.close()


                            ffmpeg_command = [
                                "ffmpeg",
                                "-y",
                                "-i",
                                raw_output.name,
                                "-c:v",
                                "libx264",
                                "-pix_fmt",
                                "yuv420p",
                                "-movflags",
                                "+faststart",
                                final_mp4.name
                            ]


                            conversion = subprocess.run(
                                ffmpeg_command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )


                            # --------------------------------
                            # IF H264 CONVERSION SUCCESS
                            # --------------------------------

                            if (
                                conversion.returncode == 0
                                and os.path.exists(
                                    final_mp4.name
                                )
                            ):

                                final_video_path = (
                                    final_mp4.name
                                )

                            else:

                                final_video_path = (
                                    raw_output.name
                                )


                            # --------------------------------
                            # READ VIDEO AS BYTES
                            # --------------------------------

                            with open(
                                final_video_path,
                                "rb"
                            ) as video_file:

                                video_bytes = (
                                    video_file.read()
                                )


                            # --------------------------------
                            # STORE BYTES IN SESSION
                            # --------------------------------

                            st.session_state.result_ready = True

                            st.session_state.result_type = "bad_video"

                            st.session_state.result_data = {

                                "defect":
                                    best_defect,

                                "confidence":
                                    detected_defects[
                                        best_defect
                                    ],

                                "video":
                                    video_bytes
                            }


                            # --------------------------------
                            # CLEAN TEMP FILES
                            # --------------------------------

                            try:

                                os.remove(
                                    input_file.name
                                )

                                os.remove(
                                    raw_output.name
                                )

                                if os.path.exists(
                                    final_mp4.name
                                ):

                                    os.remove(
                                        final_mp4.name
                                    )

                            except Exception:

                                pass


                        st.rerun()


    # ========================================================
    # RESULT COLUMN
    # ========================================================

    with result_col:

        st.markdown(
            """
            <div class="section-title">
            🤖 INSPECTION RESULT
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # WAITING
        # ----------------------------------------------------

        if not st.session_state.result_ready:

            st.markdown(
                """
                <div class="waiting">
                    <h3>⏳ WAITING FOR ANALYSIS</h3>
                    <p>
                    Analyze chesaka result ikkada display avvali.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # GOOD
        # ----------------------------------------------------

        elif st.session_state.result_type == "good":

            st.markdown(
                """
                <div class="good-result">
                    <h2>🟢 GOOD QUALITY</h2>
                    <p>No defect detected.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # BAD IMAGE / CAMERA
        # ----------------------------------------------------

        elif st.session_state.result_type == "bad":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="bad-result">
                    <h2>🔴 BAD QUALITY</h2>
                    <p>Defect Detected</p>
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="defect-box">
                    <b>Defect:</b> {data["defect"]}
                    <br><br>
                    <span class="confidence">
                    Confidence:
                    {data["confidence"] * 100:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )


            # SMALL ANNOTATED IMAGE

            st.image(
                data["image"],
                caption="Detected Defect",
                width=450
            )


        # ----------------------------------------------------
        # BAD VIDEO
        # ----------------------------------------------------

        elif st.session_state.result_type == "bad_video":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="bad-result">
                    <h2>🔴 BAD QUALITY</h2>
                    <p>Defect Detected</p>
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="defect-box">
                    <b>Defect:</b> {data["defect"]}
                    <br><br>
                    <span class="confidence">
                    Confidence:
                    {data["confidence"] * 100:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )


            # ----------------------------------------------
            # PROCESSED VIDEO
            # ----------------------------------------------

            st.video(
                data["video"],
                format="video/mp4"
            )


            st.caption(
                "Processed video with detected defect boxes"
            )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">
        YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )
