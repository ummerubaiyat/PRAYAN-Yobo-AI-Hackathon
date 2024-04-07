import streamlit as st
from banglaspeech2text import Speech2Text
import tempfile
import os

st.title("Bangla Speech to Text App")

# Upload audio file to Streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

# Load Bangla Speech to Text model
@st.cache_resource
def load_model():
    return Speech2Text("large", cache_path="cache", use_gpu=True)

stt = load_model()
st.text("Bangla Speech to Text Model Loaded")

if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        with st.spinner("Transcribing Audio..."):
            try:
                # Save audio data to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_audio:
                    tmp_audio.write(audio_file.read())
                    tmp_audio_path = tmp_audio.name
                
                # Transcribe audio
                transcription = stt.transcribe(tmp_audio_path)
                
                st.success("Transcription Complete")
                st.write(transcription)
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
    st.sidebar.audio(audio_file, format='audio/mp3')
