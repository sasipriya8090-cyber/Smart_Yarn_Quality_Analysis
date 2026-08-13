import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="YarnX – The Future of Yarn Inspection",
    page_icon="🧶",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    padding: 15px;
    border: 2px solid #222;
    margin-bottom: 15px;
}

.box {
    border: 2px solid #222;
    padding: 25px;
    min-height: 450px;
}

.predict-btn button {
    font-size: 20px;
    font-weight: bold;
}

.result-title {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.good {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    padding: 15px;
    border: 2px solid green;
    margin-top: 15px;
}

.bad {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    padding: 15px;
    border: 2px solid red;
    margin-top: 15px;
}

.defect-box {
    border: 2px solid #222;
    padding: 15px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1 – HOME / INTRODUCTION
# =========================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([35, 65])

    # -----------------------------------------------------
    # LEFT SECTION
    # -----------------------------------------------------

    with left:

        st.markdown(
            """
            <div class="box">

            <h2 style="text-align:center;">
            AI Career for Women (AICW)
            </h2>

            <h3 style="text-align:center;">
            Capstone Project
            </h3>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "PREDICT",
            use_container_width=True
        ):
            st.session_state.page = "inspection"
            st.rerun()


    # -----------------------------------------------------
    # RIGHT SECTION
    # -----------------------------------------------------

    with right:

        st.markdown(
            """
            <div class="box">

            <h2>Project Description</h2>

            <p style="font-size:18px;">
            Yarn quality inspection system using AI and
            computer vision to identify yarn defects from
            images, camera input, and videos.
            </p>

            <hr>

            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("### TEAM MEMBERS")

            st.write("1. Gutti...")
            st.write("2. Somasani...")
            st.write("3. Galidevara...")
            st.write("4. Rambala...")

        with col2:

            st.markdown("### GMAIL")

            st.write("email 1")
            st.write("email 2")
            st.write("email 3")
            st.write("email 4")

        with col3:

            st.markdown("### GUIDE NAME")

            st.write("Md. Abdul Aziz")

            st.markdown("### DESIGNATION")

            st.write("Co Lead & Trainer AICW")


# =========================================================
# PAGE 2 – INSPECTION
# =========================================================

elif st.session_state.page == "inspection":

    st.markdown(
        '<div class="main-title">🧶 YarnX – The Future of Yarn Inspection</div>',
        unsafe_allow_html=True
    )

    # Back button

    if st.button("⬅ Back"):

        st.session_state.page = "home"
        st.rerun()


    left, right = st.columns([35, 65])


    # =====================================================
    # INPUT SECTION
    # =====================================================

    with left:

        st.markdown(
            "<h2>📥 INPUT</h2>",
            unsafe_allow_html=True
        )

        st.markdown("### Select Input Type:")

        input_type = st.radio(
            "",
            [
                "🖼️ Image",
                "📷 Camera",
                "🎥 Video"
            ]
        )


        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        if input_type == "🖼️ Image":

            uploaded_file = st.file_uploader(
                "Upload Yarn Image",
                type=["jpg", "jpeg", "png", "webp"]
            )

            if uploaded_file:

                image = Image.open(uploaded_file)

                st.image(
                    image,
                    caption="INPUT PREVIEW",
                    use_container_width=True
                )

                analyze = st.button(
                    "🔍 Analyze Image",
                    use_container_width=True
                )

                if analyze:

                    with st.spinner("Analyzing yarn..."):

                        results = model.predict(
                            image,
                            conf=0.25
                        )

                    result = results[0]

                    # YOLO output image
                    plotted_image = result.plot()

                    # -------------------------------------
                    # RESULT
                    # -------------------------------------

                    with right:

                        st.markdown(
                            '<div class="result-title">🤖 INSPECTION RESULT</div>',
                            unsafe_allow_html=True
                        )

                        st.image(
                            plotted_image,
                            caption="ANALYZED IMAGE",
                            use_container_width=True
                        )


                        defects = []

                        if result.boxes is not None:

                            for box in result.boxes:

                                class_id = int(box.cls[0])

                                confidence = float(
                                    box.conf[0]
                                )

                                defect_name = model.names[class_id]

                                defects.append(
                                    (
                                        defect_name,
                                        confidence
                                    )
                                )


                        # ---------------------------------
                        # QUALITY
                        # ---------------------------------

                        if len(defects) > 0:

                            st.markdown(
                                '<div class="bad">❌ BAD QUALITY</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                '<div class="defect-box"><h3>Detected Defects</h3></div>',
                                unsafe_allow_html=True
                            )

                            for name, confidence in defects:

                                st.write(
                                    f"🔴 **Defect:** {name}"
                                )

                                st.write(
                                    f"📊 **Confidence:** "
                                    f"{confidence * 100:.2f}%"
                                )

                        else:

                            st.markdown(
                                '<div class="good">✅ GOOD QUALITY</div>',
                                unsafe_allow_html=True
                            )


        # =================================================
        # CAMERA
        # =================================================

        elif input_type == "📷 Camera":

            camera_image = st.camera_input(
                "Capture Yarn Image"
            )

            if camera_image:

                image = Image.open(camera_image)

                st.image(
                    image,
                    caption="CAMERA CAPTURE",
                    use_container_width=True
                )

                analyze = st.button(
                    "🔍 Analyze Camera Image",
                    use_container_width=True
                )

                if analyze:

                    with st.spinner("Analyzing yarn..."):

                        results = model.predict(
                            image,
                            conf=0.25
                        )

                    result = results[0]

                    plotted_image = result.plot()


                    with right:

                        st.markdown(
                            '<div class="result-title">🤖 INSPECTION RESULT</div>',
                            unsafe_allow_html=True
                        )

                        st.image(
                            plotted_image,
                            caption="ANALYZED CAMERA IMAGE",
                            use_container_width=True
                        )


                        defects = []

                        if result.boxes is not None:

                            for box in result.boxes:

                                class_id = int(box.cls[0])

                                confidence = float(
                                    box.conf[0]
                                )

                                defect_name = model.names[class_id]

                                defects.append(
                                    (
                                        defect_name,
                                        confidence
                                    )
                                )


                        if len(defects) > 0:

                            st.markdown(
                                '<div class="bad">❌ BAD QUALITY</div>',
                                unsafe_allow_html=True
                            )

                            for name, confidence in defects:

                                st.write(
                                    f"🔴 **Defect:** {name}"
                                )

                                st.write(
                                    f"📊 **Confidence:** "
                                    f"{confidence * 100:.2f}%"
                                )

                        else:

                            st.markdown(
                                '<div class="good">✅ GOOD QUALITY</div>',
                                unsafe_allow_html=True
                            )


        # =================================================
        # VIDEO
        # =================================================

        elif input_type == "🎥 Video":

            uploaded_video = st.file_uploader(
                "Upload Yarn Video",
                type=["mp4", "avi", "mov", "mkv"]
            )

            if uploaded_video:

                st.video(uploaded_video)

                analyze = st.button(
                    "🔍 Analyze Video",
                    use_container_width=True
                )

                if analyze:

                    with st.spinner(
                        "Analyzing video frames..."
                    ):

                        # Save uploaded video
                        temp_input = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )

                        temp_input.write(
                            uploaded_video.read()
                        )

                        temp_input.close()


                        cap = cv2.VideoCapture(
                            temp_input.name
                        )


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

                        fps = cap.get(
                            cv2.CAP_PROP_FPS
                        )

                        if fps <= 0:
                            fps = 25


                        output_path = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        ).name


                        fourcc = cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )

                        writer = cv2.VideoWriter(
                            output_path,
                            fourcc,
                            fps,
                            (width, height)
                        )


                        all_defects = []


                        # ---------------------------------
                        # PROCESS EVERY FRAME
                        # ---------------------------------

                        while True:

                            ret, frame = cap.read()

                            if not ret:
                                break


                            results = model.predict(
                                frame,
                                conf=0.25,
                                verbose=False
                            )

                            result = results[0]


                            # YOLO boxes + labels
                            processed_frame = result.plot()


                            writer.write(
                                processed_frame
                            )


                            if result.boxes is not None:

                                for box in result.boxes:

                                    class_id = int(
                                        box.cls[0]
                                    )

                                    confidence = float(
                                        box.conf[0]
                                    )

                                    defect_name = model.names[
                                        class_id
                                    ]


                                    defect = (
                                        defect_name,
                                        confidence
                                    )


                                    if defect_name not in [
                                        d[0]
                                        for d in all_defects
                                    ]:

                                        all_defects.append(
                                            defect
                                        )


                        cap.release()
                        writer.release()


                    # -------------------------------------
                    # VIDEO RESULT
                    # -------------------------------------

                    with right:

                        st.markdown(
                            '<div class="result-title">🤖 INSPECTION RESULT</div>',
                            unsafe_allow_html=True
                        )

                        st.video(
                            output_path
                        )


                        if len(all_defects) > 0:

                            st.markdown(
                                '<div class="bad">❌ BAD QUALITY</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                "### Detected Defects"
                            )

                            for name, confidence in all_defects:

                                st.write(
                                    f"🔴 **Defect:** {name}"
                                )

                                st.write(
                                    f"📊 **Confidence:** "
                                    f"{confidence * 100:.2f}%"
                                )

                        else:

                            st.markdown(
                                '<div class="good">✅ GOOD QUALITY</div>',
                                unsafe_allow_html=True
                            )
