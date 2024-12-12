import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

def audio_to_text():
    """Record audio and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
            return None
        except sr.RequestError:
            st.error("Speech recognition service is unavailable.")
            return None

def text_to_audio(text, file_name="response.mp3"):
    """Convert text to audio and save it."""
    tts = gTTS(text)
    audio_path = os.path.join(os.getcwd(), file_name)
    tts.save(audio_path)
    return audio_path