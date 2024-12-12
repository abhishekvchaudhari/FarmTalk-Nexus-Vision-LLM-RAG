from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import tempfile
import os

# Load the YOLO model
model = YOLO(r"./best.pt")

def detect_disease(uploaded_image):
    """
    Detect disease in the uploaded image using YOLO model.

    Args:
        uploaded_image: A file-like object or a valid file path.

    Returns:
        str: The detected disease label or 'no_detections' if none found.
    """
    try:
        # Handle the input based on type
        if isinstance(uploaded_image, str) and os.path.exists(uploaded_image):
            # If uploaded_image is a file path
            image_path = uploaded_image
        else:
            # If uploaded_image is a file-like object (e.g., from Streamlit)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                try:
                    img = Image.open(uploaded_image)
                    img.save(tmp.name, format="JPEG")
                    image_path = tmp.name
                except UnidentifiedImageError:
                    return "Invalid image format. Please upload a valid image."

        # Perform YOLO inference
        results = model.predict(source=image_path, conf=0.25)  # Adjust confidence if needed

        # Extract unique labels from the results
        unique_labels = set()
        for result in results:
            for box in result.boxes:
                unique_labels.add(model.names[int(box.cls)])  # Get class name by index

        # Clean up temporary file if created
        if not isinstance(uploaded_image, str):
            os.unlink(image_path)

        # Return the detected label or 'no_detections'
        return list(unique_labels)[0] if unique_labels else "no_detections"

    except Exception as e:
        return f"Error processing the image: {e}"

