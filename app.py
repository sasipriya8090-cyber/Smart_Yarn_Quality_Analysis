import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import io


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

/* ---------------- GLOBAL ---------------- */

.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc 0%,
        #eef4ff 50%,
        #f8fafc 100%
    );
}

.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}


/* ---------------- MAIN TITLE ---------------- */

.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    color: #172554;
    margin-top: 10px;
    margin-bottom: 28px;
    letter-spacing: 0.3px;
}


/* ---------------- PAGE 1 ---------------- */

.intro-card {
    background: rgba(255,255,255,0.95);
    border: 1px solid #dbe4f0;
    border-radius: 20px;
    padding: 30px;
    min-height: 290px;
    box-shadow: 0 8px 28px rgba(15,23,42,0.07);
}

.aicw-title {
    font-size: 29px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 20px;
}

.capstone {
    font-size: 23px;
    font-weight: 700;
    color: #334155;
    margin-top: 22px;
    margin-bottom: 28px;
}

.description-title {
    font-size: 25px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 14px;
}

.description {
    font-size: 17px;
    line-height: 1.8;
    color: #475569;
}


/* ---------------- TEAM CARDS ---------------- */

.team-card {
    background: rgba(255,255,255,0.97);
    border: 1px solid #dbe4f0;
    border-radius: 17px;
    padding: 20px;
    min-height: 245px;
    box-shadow: 0 7px 22px rgba(15,23,42,0.06);
}

.team-heading {
    font-size: 17px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 16px;
}


/* ---------------- PAGE 2 ---------------- */

.inspection-card {
    background: rgba(255,255,255,0.97);
    border: 1px solid #dbe4f0;
    border-radius: 20px;
    padding: 22px;
    min-height: 540px;
    box-shadow: 0 8px 28px rgba(15,23,42,0.07);
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 15px;
}


/* ---------------- WAITING ---------------- */

.waiting {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 55px 20px;
    text-align: center;
    margin-top: 20px;
}

.waiting-icon {
    font-size: 38px;
}

.waiting-title {
    font-size: 21px;
    font-weight: 800;
    color: #475569;
    margin-top: 10px;
}

.waiting-text {
    color: #64748b;
    font-size: 15px;
}


/* ---------------- GOOD RESULT ---------------- */

.good-result {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #f0fdf4
    );
    border: 2px solid #86efac;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 15px;
}

.good-result h2 {
    color: #15803d;
    font-size: 29px;
    margin-bottom: 8px;
}


/* ---------------- BAD RESULT ---------------- */

.bad-result {
    background: linear-gradient(
        135deg,
        #fff1f2,
        #fef2f2
    );
    border: 2px solid #fca5a5;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    margin-top: 15px;
}

.bad-result h2 {
    color: #dc2626;
    font-size: 29px;
    margin-bottom: 8px;
}

.defect-box {
    background: #fff7ed;
    border-left: 5px solid #f97316;
    border-radius: 10px;
    padding: 16px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.confidence {
    font-size: 18px;
    font-weight: 800;
    color: #334155;
}


/* ---------------- MEDIA LABEL ---------------- */

.media-label {
    font-size: 14px;
    font-weight: 700;
    color: #475569;
    margin-top: 8px;
    margin-bottom: 5px;
}


/* ---------------- FOOTER ---------------- */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-top: 30px;
}


/* ---------------- BUTTON ---------------- */

.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
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
# MODEL PATH
# ============================================================

MODEL_PATH = "best.pt"

CONF_THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO(MODEL_PATH)


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.info(
        "Please check the path of best.pt."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_best_detection(result):

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        detections.append(
            (
                class_name,
                confidence
            )
        )

    if not detections:
        return None

    return max(
        detections,
        key=lambda x: x[1]
    )


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INTRO SECTION
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 2],
        gap="large"
    )


    with left:

        st.markdown(
            """
            <div class="intro-card">

                <div class="aicw-title">
                    AI Career for Women (AICW)
                </div>

                <div class="capstone">
                    Capstone Project
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        if st.button(
            "🔍  PREDICT",
            use_container_width=True
        ):

            st.session_state.page = 2

            st.session_state.result_ready = False
            st.session_state.result_type = None
            st.session_state.result_data = None

            st.rerun()


    with right:

        st.markdown(
            """
            <div class="intro-card">

                <div class="description-title">
                    Project Description
                </div>

                <div class="description">
                    Yarn quality inspection system using AI and
                    computer vision to identify yarn defects from
                    images, camera input, and videos.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # TEAM DETAILS
    # --------------------------------------------------------

    st.write("")


    team_col, gmail_col, guide_col = st.columns(
        [1.5, 1.5, 1],
        gap="medium"
    )


    with team_col:

        st.markdown(
            """
            <div class="team-card">

                <div class="team-heading">
                    TEAM MEMBERS
                </div>

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
            <div class="team-card">

                <div class="team-heading">
                    GMAIL
                </div>

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
            <div class="team-card">

                <div class="team-heading">
                    GUIDE NAME
                </div>

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

    # --------------------------------------------------------
    # TOP TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            🧶 YarnX – The Future of Yarn Inspection
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BACK BUTTON
    # --------------------------------------------------------

    if st.button("← Back to Project"):

        st.session_state.page = 1

        st.session_state.result_ready = False
        st.session_state.result_type = None
        st.session_state.result_data = None

        st.rerun()


    st.write("")


    # --------------------------------------------------------
    # TWO COLUMNS
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
            """
            <div class="inspection-card">

                <div class="section-title">
                    📥 INPUT
                </div>

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


            if uploaded_image:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")


                st.markdown(
                    '<div class="media-label">'
                    'ORIGINAL IMAGE'
                    '</div>',
                    unsafe_allow_html=True
                )


                # Smaller preview
                st.image(
                    image,
                    width=380
                )


                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing image..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=CONF_THRESHOLD,
                            verbose=False
                        )[0]


                    best = get_best_detection(
                        result
                    )


                    # GOOD
                    if best is None:

                        st.session_state.result_ready = True
                        st.session_state.result_type = "good"
                        st.session_state.result_data = None


                    # BAD
                    else:

                        annotated = result.plot()

                        annotated = cv2.cvtColor(
                            annotated,
                            cv2.COLOR_BGR2RGB
                        )


                        st.session_state.result_ready = True
                        st.session_state.result_type = "bad"
                        st.session_state.result_data = {

                            "defect": best[0],

                            "confidence": best[1],

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


                st.markdown(
                    '<div class="media-label">'
                    'CAMERA CAPTURE'
                    '</div>',
                    unsafe_allow_html=True
                )


                st.image(
                    image,
                    width=380
                )


                if st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing captured image..."
                    ):

                        result = model.predict(
                            source=np.array(image),
                            conf=CONF_THRESHOLD,
                            verbose=False
                        )[0]


                    best = get_best_detection(
                        result
                    )


                    if best is None:

                        st.session_state.result_ready = True
                        st.session_state.result_type = "good"
                        st.session_state.result_data = None


                    else:

                        annotated = result.plot()

                        annotated = cv2.cvtColor(
                            annotated,
                            cv2.COLOR_BGR2RGB
                        )


                        st.session_state.result_ready = True
                        st.session_state.result_type = "bad"
                        st.session_state.result_data = {

                            "defect": best[0],

                            "confidence": best[1],

                            "image": annotated

                        }


                    st.rerun()


        # ====================================================
        # VIDEO
        # ====================================================

        else:

            uploaded_video = st.file_uploader(
                "Upload Video",
                type=[
                    "mp4",
                    "mov",
                    "avi",
                    "mkv"
                ],
                key="video_upload"
            )


            if uploaded_video:

                st.markdown(
                    '<div class="media-label">'
                    'ORIGINAL VIDEO'
                    '</div>',
                    unsafe_allow_html=True
                )


                # Smaller video preview
                st.video(
                    uploaded_video,
                    format="video/mp4"
                )


                if st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing video... Please wait."
                    ):

                        # ------------------------------------
                        # SAVE INPUT VIDEO
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
                        # CREATE OUTPUT AVI
                        # ------------------------------------

                        output_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".avi"
                        )

                        output_file.close()


                        fourcc = cv2.VideoWriter_fourcc(
                            *"MJPG"
                        )


                        writer = cv2.VideoWriter(
                            output_file.name,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        if not writer.isOpened():

                            cap.release()

                            st.error(
                                "❌ Could not create processed video."
                            )

                            st.stop()


                        any_defect = False

                        detected_defects = {}


                        progress = st.progress(0)


                        frame_number = 0


                        # ====================================
                        # PROCESS VIDEO
                        # ====================================

                        while True:

                            ret, frame = cap.read()


                            if not ret:
                                break


                            # YOLO expects RGB
                            rgb_frame = cv2.cvtColor(
                                frame,
                                cv2.COLOR_BGR2RGB
                            )


                            result = model.predict(
                                source=rgb_frame,
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
                            # DRAW BOXES
                            # --------------------------------

                            annotated_frame = result.plot()


                            # result.plot() already BGR
                            writer.write(
                                annotated_frame
                            )


                            frame_number += 1


                            if total_frames > 0:

                                progress.progress(
                                    min(
                                        frame_number /
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
                            # READ PROCESSED VIDEO
                            # --------------------------------

                            with open(
                                output_file.name,
                                "rb"
                            ) as f:

                                processed_video = f.read()


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
                                    processed_video
                            }


                        # ------------------------------------
                        # CLEAN INPUT
                        # ------------------------------------

                        try:

                            os.remove(
                                input_file.name
                            )

                            os.remove(
                                output_file.name
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
            <div class="inspection-card">

                <div class="section-title">
                    🤖 INSPECTION RESULT
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # WAITING
        # ====================================================

        if not st.session_state.result_ready:

            st.markdown(
                """
                <div class="waiting">

                    <div class="waiting-icon">
                        ⏳
                    </div>

                    <div class="waiting-title">
                        WAITING FOR ANALYSIS
                    </div>

                    <div class="waiting-text">
                        Analyze chesaka result ikkada display avvali.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # GOOD
        # ====================================================

        elif st.session_state.result_type == "good":

            st.markdown(
                """
                <div class="good-result">

                    <h2>
                        🟢 GOOD QUALITY
                    </h2>

                    <p>
                        No defect detected.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # BAD IMAGE / CAMERA
        # ====================================================

        elif st.session_state.result_type == "bad":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="bad-result">

                    <h2>
                        🔴 BAD QUALITY
                    </h2>

                    <p>
                        Defect Detected
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="defect-box">

                    <b>Defect:</b>
                    {data["defect"]}

                    <br><br>

                    <span class="confidence">
                        Confidence:
                        {data["confidence"] * 100:.2f}%
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="media-label">'
                'IMAGE WITH DEFECT DETECTION'
                '</div>',
                unsafe_allow_html=True
            )


            # Smaller result image
            st.image(
                data["image"],
                width=380
            )


        # ====================================================
        # BAD VIDEO
        # ====================================================

        elif st.session_state.result_type == "bad_video":

            data = st.session_state.result_data


            st.markdown(
                """
                <div class="bad-result">

                    <h2>
                        🔴 BAD QUALITY
                    </h2>

                    <p>
                        Defect Detected
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="defect-box">

                    <b>Defect:</b>
                    {data["defect"]}

                    <br><br>

                    <span class="confidence">
                        Confidence:
                        {data["confidence"] * 100:.2f}%
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="media-label">'
                'PROCESSED VIDEO WITH DEFECT BOXES'
                '</div>',
                unsafe_allow_html=True
            )


            # Processed video
            st.video(
                data["video"]
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
