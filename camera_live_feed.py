import cv2
import os
import streamlit as st

def use_lifecam():
    """Function to capture images using the Microsoft LifeCam VX-5000."""
    camera_index = 1  # Replace this with the correct index for your camera
    save_dir = "captured_images"
    os.makedirs(save_dir, exist_ok=True)
    image_count = 1

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Error: Could not open Microsoft LifeCam VX-5000.")
        return None

    st.text("Press 'Capture Image' to save an image. Turn off the checkbox to stop the camera.")
    frame_placeholder = st.empty()  # Placeholder for displaying frames
    capture_button = st.button("Capture Image")

    captured_image_path = None
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to capture video feed.")
            break

        # Display the video feed in Streamlit
        frame_placeholder.image(frame, channels="BGR", caption="Live Camera Feed")

        # Capture and save the frame when "Capture Image" button is clicked
        if capture_button:
            captured_image_path = os.path.join(save_dir, f"image_{image_count}.jpg")
            cv2.imwrite(captured_image_path, frame)
            st.success(f"Image saved as {captured_image_path}")
            image_count += 1
            break  # Exit the loop after capturing

    # Release resources
    cap.release()
    return captured_image_path
