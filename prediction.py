from ultralytics import YOLO

MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)


def predict_yarn_quality(image_path, confidence=0.25):
    """
    Predict yarn defects from an image.

    Returns:
        result: YOLO prediction result
    """

    results = model.predict(
        source=image_path,
        conf=confidence,
        verbose=False
    )

    return results[0]
