import streamlit as st
import whisper
import numpy as np
import tempfile
import os

st.title("StreamTranscribe App")

# Upload audio-file to streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()
st.text("Whisper Model Loaded")

if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        with st.spinner("Transcribing Audio..."):
            try:
                # Save audio data to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_audio:
                    tmp_audio.write(audio_file.read())
                    tmp_audio_path = tmp_audio.name
                
                # Load audio from the temporary file
                audio_array = whisper.load_audio(tmp_audio_path)
                
                # Transcribe audio
                transcription = model.transcribe(audio_array)
                
                st.success("Transcription Complete")
                st.markdown(transcription["text"])
            except Exception as e:
                st.error(f"Transcription failed: {str(e)}")
            finally:
                # Delete temporary file after use
                if tmp_audio_path:
                    os.unlink(tmp_audio_path)
    else:
        st.sidebar.error("Please Upload an Audio File")

# Display original audio file
if audio_file is not None:
    st.sidebar.header("Play Original Audio File")
    st.sidebar.audio(audio_file)
