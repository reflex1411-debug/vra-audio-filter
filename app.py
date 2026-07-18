import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile
import os
import base64

# --- PAGE SETUP ---
st.set_page_config(page_title="Neilio's VRA Toolkit", layout="wide")
LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
if "session_tracks" not in st.session_state: st.session_state.session_tracks = {}

# --- PROCESSING FUNCTIONS ---
def get_sos(low, high, fs):
    nyq = 0.5 * fs
    return butter(8, [low/nyq, high/nyq], btype='band', output='sos')

def process_audio(file_source, low=None, high=None, trim=0.0):
    # Load file
    if isinstance(file_source, str):
        audio = AudioSegment.from_file(file_source)
    else:
        file_source.seek(0)
        audio = AudioSegment.from_file(io.BytesIO(file_source.read()))
    
    # Trim & Convert
    audio = audio[int(trim*1000):]
    audio = audio.set_frame_rate(44100).set_channels(1)
    data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    
    # Filter
    if low and high:
        sos = get_sos(low, high, 44100)
        data = sosfilt(sos, data)
        
    # Export
    buf = io.BytesIO()
    sf.write(buf, data, 44100, format='WAV')
    buf.seek(0)
    return buf

# --- UI INTERFACE ---
st.title("🎧 Neilio's VRA Clinical Toolkit")

# Sidebar/Top Controls
with st.container():
    col1, col2 = st.columns(2)
    files = [f for f in os.listdir(LIBRARY_DIR)] + list(st.session_state.session_tracks.keys())
    selected = col1.selectbox("Select Track", ["-- Select --"] + files)
    
    new_files = col2.file_uploader("Batch Upload", accept_multiple_files=True)
    if new_files:
        for f in new_files: st.session_state.session_tracks[f.name] = f.read()

if selected != "-- Select --":
    active = os.path.join(LIBRARY_DIR, selected) if selected in os.listdir(LIBRARY_DIR) else io.BytesIO(st.session_state.session_tracks[selected])
    
    trim = st.slider("Trim Start (s)", 0.0, 10.0, 0.0)
    offset = st.slider("Jumpback Offset (s)", 0.0, 5.0, 2.0)
    
    # Manifest of Frequencies
    manifest = [
        ("Full-Range", None, None), 
        ("500Hz", 420, 595), 
        ("1000Hz", 841, 1189),
        ("2000Hz", 1682, 2378)
    ]
    
    st.write("### Presentation Channels")
    cols = st.columns(len(manifest))
    
    for i, (lab, low, high) in enumerate(manifest):
        with cols[i]:
            st.markdown(f"#### {lab}")
            buf = process_audio(active, low, high, trim)
            
            # Simple Player
            st.audio(buf)
            
            # Action Buttons
            if st.button(f"Save {lab}", key=f"btn_{lab}"):
                st.download_button("Download", buf, file_name=f"{lab}.wav")
