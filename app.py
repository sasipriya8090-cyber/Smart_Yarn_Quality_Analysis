import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Smart Yarn Quality Analysis",
    page_icon="🧶",
    layout="wide"
)

st.title("🧶 Smart Yarn Quality Analysis")

st.write(
    "AI-based yarn defect detection using YOLO."
)


# ==============================
# MODEL
# ==============================

MODEL_PATH = "best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ==============================
# INPUT TYPE
# ==============================

input_type = st.radio(
    "Select Input Type",
    ["Image", "Video"],
    horizontal=True
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

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

                    image.convert("RGB").save(temp_file.name)
                    image_path = temp_file.name

                # Low threshold for inspection
                result = model.predict(
                    source=image_path,
                    conf=0.01,
                    verbose=False
                )[0]

                os.remove(image_path)

            # =====================================
            # NO DETECTION
            # =====================================

            if len(result.boxes) == 0:

                st.success("🟢 NO DEFECT DETECTED")

                st.write(
                    "No trained yarn defect was detected."
                )

            else:

                # Find highest confidence detection
                best_box = max(
                    result.boxes,
                    key=lambda box: float(box.conf[0])
                )

                class_id = int(best_box.cls[0])
                confidence = float(best_box.conf[0])
                class_name = result.names[class_id]

                confidence_percent = confidence * 100

                # =================================
                # QUALITY DECISION
                # =================================

                if confidence_percent >= 50:

                    st.error("🔴 DEFECT DETECTED")

                    quality_text = "Poor Quality"

                elif confidence_percent >= 10:

                    st.warning("🟡 POTENTIAL DEFECT")

                    quality_text = "Needs Inspection"

                else:

                    st.info("🔵 LOW-CONFIDENCE DETECTION")

                    quality_text = "Not Confirmed"

                # =================================
                # RESULT
                # =================================

                st.subheader("Prediction Result")

                st.write(
                    f"**Quality Status:** {quality_text}"
                )

                st.write(
                    f"**Defect:** {class_name}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence_percent:.2f}%"
                )

                result_image = result.plot()

                st.subheader("Detection Result")

                st.image(
                    result_image,
                    caption="YOLO Detection",
                    use_container_width=True
                )

                # =================================
                # ALL DETECTIONS
                # =================================

                st.subheader("All Detected Objects")

                for box in result.boxes:

                    cid = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = result.names[cid]

                    st.write(
                        f"• **{name}** — {conf * 100:.2f}%"
                    )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

else:

    uploaded_video = st.file_uploader(
        "Upload Yarn Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("🎥 Analyze Video"):

            with st.spinner("Analyzing video..."):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_file:

                    temp_file.write(
                        uploaded_video.read()
                    )

                    video_path = temp_file.name

                cap = cv2.VideoCapture(video_path)

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                frame_number = 0
                detected_frames = 0

                highest_confidence = 0.0
                detected_classes = set()

                progress_bar = st.progress(0)

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1

                    # Analyze every 5th frame
                    if frame_number % 5 == 0:

                        result = model.predict(
                            source=frame,
                            conf=0.01,
                            verbose=False
                        )[0]

                        if len(result.boxes) > 0:

                            detected_frames += 1

                            for box in result.boxes:

                                class_id = int(
                                    box.cls[0]
                                )

                                confidence = float(
                                    box.conf[0]
                                )

                                class_name = (
                                    result.names[class_id]
                                )

                                detected_classes.add(
                                    class_name
                                )

                                highest_confidence = max(
                                    highest_confidence,
                                    confidence
                                )

                    if total_frames > 0:

                        progress_bar.progress(
                            min(
                                frame_number / total_frames,
                                1.0
                            )
                        )

                cap.release()

                os.remove(video_path)

            # =====================================
            # VIDEO RESULT
            # =====================================

            st.subheader("🎥 Video Analysis Result")

            if detected_frames == 0:

                st.success("🟢 NO DEFECT DETECTED")

                st.write(
                    "No trained yarn defect was detected "
                    "in the analyzed video frames."
                )

            else:

                confidence_percent = (
                    highest_confidence * 100
                )

                if confidence_percent >= 50:

                    st.error("🔴 DEFECT DETECTED")

                    quality_text = "Poor Quality"

                elif confidence_percent >= 10:

                    st.warning("🟡 POTENTIAL DEFECT")

                    quality_text = "Needs Inspection"

                else:

                    st.info("🔵 LOW-CONFIDENCE DETECTION")

                    quality_text = "Not Confirmed"

                st.write(
                    f"**Quality Status:** {quality_text}"
                )

                st.write("**Detected Defects:**")

                for defect in detected_classes:

                    st.write(
                        f"• {defect}"
                    )

                st.metric(
                    "Highest Confidence",
                    f"{confidence_percent:.2f}%"
                )

            st.info(
                f"Frames processed: {frame_number}"
            )
