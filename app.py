import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile
import os

# Set wide layout
st.set_page_config(page_title="Neilio's VRA Toolkit", page_icon="🎧", layout="wide", initial_sidebar_state="collapsed")

# Setup library
LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
if "session_tracks" not in st.session_state: st.session_state.session_tracks = {}

# Custom Audiometer Structural Frame CSS
st.markdown("""
    <style>
        .stApp { background-color: #0f172a !important; }
        audio { height: 32px !important; margin-bottom: 8px !important; margin-top: 2px !important; width: 100%; }
    </style>
    <div style="background: linear-gradient(180deg, #334155 0%, #1e293b 100%); padding: 15px 25px; border-radius: 8px 8px 0px 0px; border: 2px solid #475569; border-bottom: none; display: flex; align-items: center; justify-content: space-between;">
        <div style="text-align: left;">
            <h1 style="color: #f8fafc; margin: 0; font-family: monospace; font-size: 1.4rem; font-weight: 700;">NEILIO'S VRA CLINICAL STIMULI GENERATOR</h1>
            <div style="color: #94a3b8; font-size: 0.75rem; font-family: monospace; margin-top: 2px;">MODEL VRA-11 // RMS-CALIBRATED OUTPUT MATRIX</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Processing Functions
def butter_filter_sos(low, high, fs):
    nyq = 0.5 * fs
    return butter(8, [low/nyq, high/nyq], btype='band', output='sos')

def process_audio_buffer(file_source, lowcut, highcut, trim_seconds=0.0):
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f: data, fs = sf.read(io.BytesIO(f.read()))
    else:
        file_source.seek(0)
        data, fs = sf.read(io.BytesIO(file_source.read()))
    
    if trim_seconds > 0: data = data[int(trim_seconds * fs):]
    
    if lowcut and highcut:
        sos = butter_filter_sos(lowcut, highcut, fs)
        data = sosfilt(sos, data)
        
    buf = io.BytesIO()
    sf.write(buf, data, fs, format='WAV')
    buf.seek(0)
    return buf

# Audiometer Channel Renderer
def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset):
    import base64
    b64 = base64.b64encode(audio_buffer.getvalue()).decode()
    html = f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
        <div style="font-family: monospace; color: #f8fafc; font-weight: bold;">{label}</div>
        <audio id="a_{element_key}" src="data:audio/wav;base64,{b64}" controls style="width:100%;"></audio>
        <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button onclick="document.getElementById('a_{element_key}').play()" style="flex:1; background:#10b981; color:white; border:none; border-radius:4px;">▶ PLAY</button>
            <button onclick="document.getElementById('a_{element_key}').pause()" style="flex:1; background:#ef4444; color:white; border:none; border-radius:4px;">⏸ PAUSE</button>
        </div>
        <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button onclick="window.parent.loop=Math.max(0, document.getElementById('a_{element_key}').currentTime-{preroll_offset}); this.innerHTML='⚙️ MARKED';" style="flex:1; background:#f59e0b; color:black; border:none; border-radius:4px;">🔴 MARK POINT</button>
            <button onclick="document.getElementById('a_{element_key}').currentTime=window.parent.loop; document.getElementById('a_{element_key}').play();" style="flex:1; background:#38bdf8; color:black; border:none; border-radius:4px;">↩️ JUMP BACK</button>
        </div>
    </div>
    """
    st.components.v1.html(html, height=165)

# Main UI
with st.container(border=True):
    # Track selection
    files = [f for f in os.listdir(LIBRARY_DIR)] + list(st.session_state.session_tracks.keys())
    selected = st.selectbox("Select Track", ["-- Select --"] + files)
    
    if selected != "-- Select --":
        active = os.path.join(LIBRARY_DIR, selected) if selected in os.listdir(LIBRARY_DIR) else io.BytesIO(st.session_state.session_tracks[selected])
        
        c1, c2 = st.columns(2)
        trim = c1.slider("Trim Intro (s)", 0.0, 10.0, 0.0)
        offset = c2.slider("Jumpback Offset (s)", 0.0, 5.0, 2.0)
        
        manifest = [("Full-Range", None, None), ("500Hz", 400, 600), ("1000Hz", 800, 1200)]
        
        cols = st.columns(len(manifest))
        for i, (lab, low, high) in enumerate(manifest):
            with cols[i]:
                buf = process_audio_buffer(active, low, high, trim)
                render_audiometer_channel(lab, buf, lab, offset)
                st.download_button(f"Save {lab}", buf, file_name=f"{lab}.wav")

    # Batch Upload
    new_files = st.file_uploader("Batch Upload", accept_multiple_files=True)
    if new_files:
        for f in new_files: st.session_state.session_tracks[f.name] = f.read()
        st.rerun()
