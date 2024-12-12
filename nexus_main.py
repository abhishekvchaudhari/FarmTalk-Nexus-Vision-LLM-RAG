import os
import re

# Configure the environment variable to address the OpenMP duplicate library conflict.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from rag_framework import *
from yolo_inference import detect_disease
from PIL import Image, UnidentifiedImageError
from camera_live_feed import use_lifecam 
from audio_prompts import audio_to_text, text_to_audio
import base64  

from dotenv import load_dotenv
load_dotenv()

# Load API key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

UPLOAD_DIR =  r"./research_papers"

# Ensure the UPLOAD_DIR directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def handle_uploaded_file(uploaded_file):
    # Function to handle the uploaded file
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File {uploaded_file.name} uploaded successfully.")

def main():
    st.set_page_config("FarmTalk : NEXUS")
    disease = None
    st.markdown(
        """
        <style>
        .main-header {
            text-align: center; 
            font-size: 36px; 
            font-weight: bold; 
            color: #FFD700; /* Golden text color */
            background-color: #800000; /* Maroon background color */
            padding: 10px; /* Add some padding for spacing */
            width: 100%; /* Full width across the page */
            display: block; /* Ensure it behaves as a block-level element */
        }
        .sub-header {
            text-align: center; 
            font-size: 18px; 
            font-weight: bold; 
            color: #FFD700; /* Golden text color */
            background-color: #800000; /* Maroon background color */
            padding: 5px; /* Add some padding for spacing */
            width: 100%; /* Full width across the page */
            display: block; /* Ensure it behaves as a block-level element */
        }
        .detected-disease { 
            font-size: 16px; 
            font-weight: bold; 
            color: #333; 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header Section
    st.markdown(
        """
        <div class="main-header">Farm Talk: NEXUS</div>
        <div class="sub-header">A Real-Time Platform for Plant Disease Identification and Management</div>
        """,
        unsafe_allow_html=True,
    )


    
    docs = upload_data()
    create_vector_store(docs)

    st.markdown(
        """
        <style>
        .smaller-font {
            font-size:17px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="smaller-font">Team: Abhishek Chaudhari, Guruvasanth Amirthapandian Sureshkumar, Shesha Sai Kumar Reddy Sadu</p>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.disease_info = {}
        st.session_state.potential_causes = []
        st.session_state.root_cause = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    with st.sidebar:
        st.markdown(
            """
            <style>
            .center-image {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="center-image">', unsafe_allow_html=True)
        st.image("img/m_logo.png", width=200)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.title("Disease Prediction")
        input_method = st.radio("Select input method:", ("Upload Image", "Use Camera"))


        if input_method == "Upload Image":
            # Image upload functionality
            uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                with st.spinner("Processing..."):
                    # Pass the uploaded file directly to the detection function
                    disease = detect_disease(uploaded_image)
                    if disease != "no_detections":
                        st.session_state.disease = disease
                        st.session_state.disease_info[disease] = {"disease": disease}
                        st.success(f"Disease detected: {disease}")
                
                    else:
                        st.warning("No disease detected.")


        elif input_method == "Use Camera":
            with st.spinner("Accessing camera..."):
                # Capture an image from the live camera
                captured_image_path = use_lifecam()  # Ensure this function captures and saves the image properly
                if captured_image_path:
                    try:
                        # Open the saved image and pass it to the detection function
                        with open(captured_image_path, "rb") as f:
                            disease = detect_disease(f)  # Directly pass the file object
                        if disease != "no_detections":
                            st.success(f"Disease detected: {disease}")
                        else:
                            st.warning("No disease detected.")
                    except Exception as e:
                        st.error(f"Error processing the captured image: {e}")
                else:
                    st.error("Failed to capture image from the camera.")
                    
    st.markdown(
    """
    <style>
    .small-title {
        text-align: center;
        font-size: 20px; /* Adjust the size here */
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #333; /* Text color */
    }
    </style>
    <div class="small-title">Chat with Audio/Text</div>
    """,
    unsafe_allow_html=True,
)
    # Allow the user to choose between text or audio input
    input_method = st.radio("Select Input Method:", ("Text", "Audio"))

    if input_method == "Text":
        if prompt := st.chat_input("Enter your query here: "):
            with st.spinner("Working on your query..."):
                llm = create_llm_model()
                embeddings = NVIDIAEmbeddings()
                faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

                response = get_response(llm=llm, 
                                        vector_DB=faiss_index, 
                                        question=prompt + f"\n\nContext: The detected disease is '{disease}'." if disease else prompt, 
                                        chat_history=st.session_state.chat_history)
                
                print("st.session_state.chat_history::: ", st.session_state.chat_history)
                # Display the conversation
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])

                # Append messages to session state
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                st.session_state.chat_history.extend([(prompt, response["answer"])])

                if "root_cause" in st.session_state and "remedies" not in st.session_state:
                    st.session_state.remedies = response["answer"]
                    #st.sidebar.success(f"Remedies: {response['answer']}")

                # Check for disease-related context in the prompt
                for disease, info in st.session_state.disease_info.items():
                    if disease in prompt:
                        st.session_state.messages.append({"role": "assistant", "content": f"The recent root cause for {disease} was {info['root_cause']} due to {', '.join(info['potential_causes'])}"})
                        break

    elif input_method == "Audio":
        # Record audio prompt
        record_audio = st.button("Record Audio")
        if record_audio:
            user_prompt = audio_to_text()  # Convert audio to text
            if user_prompt:
                st.success(f"You said: {user_prompt}")

                with st.spinner("Working on your query..."):
                    llm = create_llm_model()
                    embeddings = NVIDIAEmbeddings()
                    faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

                    response = get_response(llm=llm, 
                                            vector_DB=faiss_index, 
                                            question=user_prompt + f"\n\nContext: The detected disease is '{disease}'." if disease else user_prompt,  
                                            chat_history=st.session_state.chat_history)
                    
                    # Display the conversation
                    st.markdown(f"**User (Audio):** {user_prompt}")
                    st.markdown(f"**Assistant:** {response['answer']}")

                    # Append messages to session state
                    st.session_state.messages.append({"role": "user", "content": user_prompt})
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    st.session_state.chat_history.extend([(user_prompt, response["answer"])])

                    if "root_cause" in st.session_state and "remedies" not in st.session_state:
                        st.session_state.remedies = response["answer"]
                       # st.sidebar.success(f"Remedies: {response['answer']}")

                    # Generate audio response
                    audio_path = text_to_audio(response["answer"])
                    st.success("Playing the generated audio response...")

                    # Embed the audio with autoplay
                    audio_html = f"""
                    <audio autoplay>
                        <source src="data:audio/mpeg;base64,{base64.b64encode(open(audio_path, "rb").read()).decode()}" type="audio/mpeg">
                        Your browser does not support the audio element.
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)

                    with st.sidebar:
                        st.markdown("### Video Response")
                        hardcoded_video_path = r"./video/avatar.mp4"

                        try:
                            with open(hardcoded_video_path, "rb") as video_file:
                                video_bytes = video_file.read()

                            # Encode the video in base64 for embedding
                            video_base64 = base64.b64encode(video_bytes).decode()

                            # Embed the video with autoplay
                            video_html = f"""
                            <video width="100%" autoplay controls>
                                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                            """
                            st.markdown(video_html, unsafe_allow_html=True)
                        except FileNotFoundError:
                            st.error("Video file not found. Please check the file path.")


                    # Check for disease-related context in the prompt
                    for disease, info in st.session_state.disease_info.items():
                        if disease in user_prompt:
                            st.session_state.messages.append({"role": "assistant", "content": f"The recent root cause for {disease} was {info['root_cause']} due to {', '.join(info['potential_causes'])}"})
                            break


    with st.sidebar:
        st.title("Upload Documents for Vector database")

        uploaded_file = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"])
        if uploaded_file:
            handle_uploaded_file(uploaded_file)
            docs = upload_data()
            create_vector_store(docs)


    # Footer Section
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <hr>
            <small>University of Minnesota | Robotics Institute</small>
        </div>
        """,
        unsafe_allow_html=True,
    )



if __name__ == "__main__":
    main()
