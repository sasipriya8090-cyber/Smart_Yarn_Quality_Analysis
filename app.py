import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os


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
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 25px;
}

/* Page 1 */
.aicw-title {
    font-size: 32px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 12px;
}

.capstone {
    font-size: 25px;
    font-weight: 700;
    color: #334155;
    margin-top: 35px;
    margin-bottom: 30px;
}

.description-title {
    font-size: 25px;
    font-weight: 700;
    color: #172554;
}

.description {
    font-size: 17px;
    line-height: 1.7;
    color: #475569;
}

/* Cards */
.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 25px;
    min-height: 420px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
}

/* Team table */
.team-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
    padding: 15px;
    margin-top: 25px;
}

/* Input / Result headings */
.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 15px;
}

/* Waiting result */
.waiting {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 15px;
    padding: 45px 20px;
    text-align: center;
    margin-top: 20px;
}

.waiting h3 {
    color: #64748b;
}

/* Good result */
.good-result {
    background: #ecfdf5;
    border: 2px solid #86efac;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-top: 15px;
}

.good-result h2 {
    color: #15803d;
    font-size: 30px;
}

/* Bad result */
.bad-result {
    background: #fef2f2;
    border: 2px solid #fca5a5;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-top: 15px;
}

.bad-result h2 {
    color: #dc2626;
    font-size: 30px;
}

.defect-box {
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 18px;
    border-radius: 10px;
    margin-top: 15px;
}

.confidence {
    font-size: 19px;
    font-weight: 700;
    color: #334155;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 30px;
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

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best.pt"
)

CONF_THRESHOLD = 0.50


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error("❌ Trained model could not be loaded.")
    st.write("Make sure `best.pt` is present beside `app.py`.")
    st.code(str(e))
    st.stop()


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    # Browser-style title
    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:36px;
            font-weight:800;
            color:#172554;
            margin-bottom:25px;
        ">
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
    # LEFT
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="aicw-title">AI Career for Women (AICW)</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="capstone">Capstone Project</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "🔍 PREDICT",
            use_container_width=True
        ):

            st.session_state.page = 2
            st.session_state.result_ready = False
            st.rerun()


    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="description-title">Project Description</div>',
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


    # --------------------------------------------------------
    # TEAM DETAILS
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    team_col, gmail_col, guide_col = st.columns(
        [1.5, 1.5, 1],
        gap="medium"
    )

    with team_col:

        st.markdown(
            '<div class="team-box"><b>TEAM MEMBERS</b><br><br>'
            '1. Gutti.pavani devi Priya<br><br>'
            '2. Somasani.sasi priya<br><br>'
            '3. Galidevara.Rama Devi<br><br>'
            '4. Rambala.Harshitha sai Lakshmi'
            '</div>',
            unsafe_allow_html=True
        )

    with gmail_col:

        st.markdown(
            '<div class="team-box"><b>GMAIL</b><br><br>'
            'gutthipavanidevipriya@gmail.com<br><br>'
            'Sasipriya8090@gmail.com<br><br>'
            'ramadevigalidevara0gmail.com<br><br>'
            'harshitharambala3@gmail.com'
            '</div>',
            unsafe_allow_html=True
        )

    with guide_col:

        st.markdown(
            '<div class="team-box">'
            '<b>GUIDE NAME</b><br><br>'
            'Md.Abdul Aziz<br><br>'
            '<b>Designation</b><br><br>'
            'Co Lead & Trainer AICW'
            '</div>',
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

    # --------------------------------------------------------
    # TOP TITLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '🧶 YarnX – The Future of Yarn Inspection'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BACK BUTTON
    # --------------------------------------------------------

    if st.button("← Back to Project"):

        st.session_state.page = 1
        st.session_state.result_ready = False
        st.rerun()


    # --------------------------------------------------------
    # TWO COLUMN LAYOUT
    # --------------------------------------------------------

    input_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # INPUT COLUMN
    # ========================================================

    with input_col:

        st.markdown(
            '<div class="section-title">📥 INPUT</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # INPUT TYPE
        # ----------------------------------------------------

        input_type = st.radio(
            "Select Input Type:",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ],
            horizontal=False
        )


        st.write("")


        # ====================================================
        # IMAGE
        # ====================================================

        if input_type == "🖼️ Image":

            uploaded_image = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                key="image_upload"
            )


            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")


                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )


                analyze_image = st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                )


                # ------------------------------------------------
                # ANALYZE
                # ------------------------------------------------

                if analyze_image:

                    with st.spinner(
                        "Analyzing image..."
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
        # CAMERA
        # ====================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Take a picture of the yarn"
            )


            if camera_image:

                image = Image.open(
                    camera_image
                ).convert("RGB")


                st.image(
                    image,
                    caption="Captured Image",
                    use_container_width=True
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
                type=["mp4", "avi", "mov", "mkv"],
                key="video_upload"
            )


            if uploaded_video:

                st.video(
                    uploaded_video
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
                        # TEMP INPUT VIDEO
                        # ------------------------------------

                        input_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        input_temp.write(
                            uploaded_video.getbuffer()
                        )

                        input_temp.close()


                        cap = cv2.VideoCapture(
                            input_temp.name
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
                        # TEMP OUTPUT VIDEO
                        # ------------------------------------

                        output_temp = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        output_temp.close()


                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )


                        writer = cv2.VideoWriter(
                            output_temp.name,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        any_defect = False
                        detected_defects = {}


                        progress = st.progress(0)


                        frame_count = 0


                        # ------------------------------------
                        # PROCESS FRAMES
                        # ------------------------------------

                        while True:

                            ret, frame = cap.read()

                            if not ret:
                                break


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
                            # DETECTIONS
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


                                if (
                                    class_name
                                    not in detected_defects
                                ):

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
                            # DRAW BOXES
                            # --------------------------------

                            annotated = result.plot()

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


                        # ------------------------------------
                        # RESULT
                        # ------------------------------------

                        if not any_defect:

                            st.session_state.result_ready = True
                            st.session_state.result_type = "good"
                            st.session_state.result_data = None

                        else:

                            best_defect = max(
                                detected_defects,
                                key=detected_defects.get
                            )


                            st.session_state.result_ready = True
                            st.session_state.result_type = "bad_video"
                            st.session_state.result_data = {
                                "defect": best_defect,
                                "confidence":
                                    detected_defects[
                                        best_defect
                                    ],
                                "video":
                                    output_temp.name
                            }


                        # Input temp file can be removed
                        try:
                            os.remove(
                                input_temp.name
                            )
                        except:
                            pass


                        st.rerun()


    # ========================================================
    # RESULT COLUMN
    # ========================================================

    with result_col:

        st.markdown(
            '<div class="section-title">'
            '🤖 INSPECTION RESULT'
            '</div>',
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
        # GOOD QUALITY
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
                    <b>Defect:</b> {data["defect"]}<br><br>
                    <span class="confidence">
                    Confidence:
                    {data["confidence"] * 100:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )


            st.image(
                data["image"],
                caption="Detected Defect",
                use_container_width=True
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
                    <b>Defect:</b> {data["defect"]}<br><br>
                    <span class="confidence">
                    Confidence:
                    {data["confidence"] * 100:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )


            # ----------------------------------------------
            # SHOW PROCESSED VIDEO
            # ----------------------------------------------

            video_path = data["video"]


            if os.path.exists(video_path):

                with open(
                    video_path,
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()


                st.video(
                    video_bytes
                )


                st.caption(
                    "Processed video with detected defect boxes"
                )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        '<div class="footer">'
        'YarnX – The Future of Yarn Inspection'
        '</div>',
        unsafe_allow_html=True
    )
