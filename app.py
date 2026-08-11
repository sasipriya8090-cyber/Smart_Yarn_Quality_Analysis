import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Smart Yarn Quality Analysis",
    page_icon="🧶",
    layout="wide"
)

st.title("🧶 Smart Yarn Quality Analysis")
st.write("AI-based yarn quality analysis using YOLO.")

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ==========================================
# CONFIDENCE THRESHOLD
# ==========================================

QUALITY_THRESHOLD = 0.50

# ==========================================
# SELECT INPUT
# ==========================================

input_type = st.radio(
    "Select Input Type",
    ["Image", "Video"],
    horizontal=True
)

# =========================================================
# IMAGE ANALYSIS
# =========================================================

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload Yarn Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Uploaded Yarn Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button("🔍 Analyze Image"):

            with st.spinner("Analyzing yarn..."):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as temp_file:

                    image.convert("RGB").save(
                        temp_file.name
                    )

                    image_path = temp_file.name

                result = model.predict(
                    source=image_path,
                    conf=0.01,
                    verbose=False
                )[0]

                os.remove(image_path)

            # ==========================================
            # CHECK DETECTIONS
            # ==========================================

            if len(result.boxes) == 0:

                st.success("🟢 GOOD QUALITY")

                st.write("**No defect detected.**")

            else:

                # Find highest confidence detection
                best_box = max(
                    result.boxes,
                    key=lambda box: float(box.conf[0])
                )

                class_id = int(best_box.cls[0])

                confidence = float(
                    best_box.conf[0]
                )

                class_name = result.names[class_id]

                confidence_percent = confidence * 100

                # ======================================
                # QUALITY DECISION
                # ======================================

                if confidence >= QUALITY_THRESHOLD:

                    # HIGH CONFIDENCE DEFECT
                    st.error("🔴 BAD QUALITY")

                    st.write(
                        f"**Defect:** {class_name}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence_percent:.2f}%"
                    )

                else:

                    # LOW CONFIDENCE
                    st.success("🟢 GOOD QUALITY")

                    st.write(
                        "**No defect detected.**"
                    )

                    st.caption(
                        "No sufficiently confident "
                        "yarn defect was detected."
                    )

                    st.metric(
                        "Highest Detection Confidence",
                        f"{confidence_percent:.2f}%"
                    )

                # ======================================
                # DETECTION IMAGE
                # ======================================

                st.subheader("Detection Result")

                result_image = result.plot()

                st.image(
                    result_image,
                    caption="YOLO Detection",
                    use_container_width=True
                )


# =========================================================
# VIDEO ANALYSIS
# =========================================================

else:

    uploaded_video = st.file_uploader(
        "Upload Yarn Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("🎥 Analyze Video"):

            with st.spinner(
                "Analyzing video..."
            ):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_file:

                    temp_file.write(
                        uploaded_video.read()
                    )

                    video_path = temp_file.name

                cap = cv2.VideoCapture(
                    video_path
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                frame_number = 0

                highest_confidence = 0.0

                highest_class = None

                progress = st.progress(0)

                # ======================================
                # PROCESS VIDEO
                # ======================================

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1

                    # Analyze every 5th frame
                    if frame_number % 5 == 0:

                        results = model.predict(
                            source=frame,
                            conf=0.01,
                            verbose=False
                        )

                        result = results[0]

                        if len(result.boxes) > 0:

                            for box in result.boxes:

                                confidence = float(
                                    box.conf[0]
                                )

                                class_id = int(
                                    box.cls[0]
                                )

                                class_name = (
                                    result.names[
                                        class_id
                                    ]
                                )

                                if confidence > highest_confidence:

                                    highest_confidence = (
                                        confidence
                                    )

                                    highest_class = (
                                        class_name
                                    )

                    # Progress
                    if total_frames > 0:

                        progress.progress(
                            min(
                                frame_number /
                                total_frames,
                                1.0
                            )
                        )

                cap.release()

                os.remove(video_path)

            # ==========================================
            # VIDEO RESULT
            # ==========================================

            st.subheader(
                "🎥 Video Analysis Result"
            )

            confidence_percent = (
                highest_confidence * 100
            )

            if (
                highest_confidence
                >= QUALITY_THRESHOLD
            ):

                st.error(
                    "🔴 BAD QUALITY"
                )

                st.write(
                    f"**Defect:** {highest_class}"
                )

                st.metric(
                    "Highest Confidence",
                    f"{confidence_percent:.2f}%"
                )

            else:

                st.success(
                    "🟢 GOOD QUALITY"
                )

                st.write(
                    "**No defect detected.**"
                )

                if highest_class is not None:

                    st.caption(
                        "No sufficiently confident "
                        "yarn defect was detected."
                    )

                st.metric(
                    "Highest Detection Confidence",
                    f"{confidence_percent:.2f}%"
                )

            st.info(
                f"Frames processed: {frame_number}"
            )
