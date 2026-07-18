import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile
import os
import base64

# Set wide layout
st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ensure local persistence folder exists
LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
if "session_tracks" not in st.session_state: st.session_state.session_tracks = {}
if "favorites" not in st.session_state: st.session_state.favorites = []

# --- CSS INJECTION ---
st.markdown("""
    <style>
        .stApp { background-color: #0f172a !important; }
        .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
        audio { height: 32px !important; margin-bottom: 8px !important; margin-top: 2px !important; width: 100%; }
    </style>
    <div style="background: linear-gradient(180deg, #334155 0%, #1e293b 100%); padding: 15px 25px; border-radius: 8px 8px 0px 0px; border: 2px solid #475569; border-bottom: none; display: flex; align-items: center; justify-content: space-between; box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="text-align: left;">
                <h1 style="color: #f8fafc; margin: 0; font-family: monospace; font-size: 1.4rem; letter-spacing: 1px; font-weight: 700;">NEILIO'S VRA CLINICAL STIMULI GENERATOR</h1>
                <div style="color: #94a3b8; font-size: 0.75rem; font-family: monospace; margin-top: 2px;">MODEL VRA-11 // RMS-CALIBRATED OUTPUT MATRIX</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- PROCESSING FUNCTIONS ---
def butter_filter_sos(cutoff_low, cutoff_high, fs, filter_type='band', order=8):
    nyq = 0.5 * fs
    if filter_type == 'low': return butter(order, cutoff_high/nyq, btype='low', output='sos')
    elif filter_type == 'high': return butter(order, cutoff_low/nyq, btype='high', output='sos')
    return butter(order, [cutoff_low/nyq, cutoff_high/nyq], btype='band', output='sos')

def rms_normalize(data, target_db=-20.0, peak_limit=0.95):
    current_rms = np.sqrt(np.mean(data**2))
    if current_rms == 0: return data
    gain = (10 ** (target_db / 20.0)) / current_rms
    data = data * gain
    max_peak = np.max(np.abs(data))
    return (data / max_peak) * peak_limit if max_peak > peak_limit else data

@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda _: None})
def process_audio_buffer(file_bytes, lowcut, highcut, filter_type, order, trim_seconds):
    # Load
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_frame_rate(44100).set_channels(1)
    fs = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    
    # Trim
    if trim_seconds > 0:
        start_sample = int(trim_seconds * fs)
        data = data[start_sample:]
        
    # Filter
    if filter_type != 'raw':
        sos = butter_filter_sos(lowcut, highcut, fs, filter_type, order)
        data = sosfilt(sos, data)
        
    data = rms_normalize(data)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset):
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    html_code = f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
        <div style="font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold; margin-bottom: 6px;">{label}</div>
        <audio id="audio_{element_key}" src="data:audio/wav;base64,{audio_base64}" controls style="width:100%;"></audio>
        <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button onclick="document.getElementById('audio_{element_key}').play()" style="flex:1; background:#10b981; color:white; border:none; padding:6px; border-radius:4px; font-family:monospace; font-size:0.75rem;">▶ PLAY</button>
            <button onclick="document.getElementById('audio_{element_key}').pause()" style="flex:1; background:#ef4444; color:white; border:none; padding:6px; border-radius:4px; font-family:monospace; font-size:0.75rem;">⏸ PAUSE</button>
        </div>
        <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button onclick="window.parent.loop=Math.max(0, document.getElementById('audio_{element_key}').currentTime-{preroll_offset}); this.innerHTML='⚙️ MARKED';" style="flex:1; background:#f59e0b; border:none; padding:6px; border-radius:4px; font-family:monospace; font-size:0.75rem;">🔴 MARK POINT</button>
            <button onclick="document.getElementById('audio_{element_key}').currentTime=window.parent.loop; document.getElementById('audio_{element_key}').play();" style="flex:1; background:#38bdf8; border:none; padding:6px; border-radius:4px; font-family:monospace; font-size:0.75rem;">↩️ JUMP BACK</button>
        </div>
    </div>
    """
    st.components.v1.html(html_code, height=165)

# --- MAIN UI ---
stored_files = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))]
all_tracks = stored_files + list(st.session_state.session_tracks.keys())
selected = st.selectbox("Select Active Track", ["-- Select --"] + all_tracks)

if selected != "-- Select --":
    file_bytes = open(os.path.join(LIBRARY_DIR, selected), 'rb').read() if selected in stored_files else st.session_state.session_tracks[selected]
    
    trim = st.slider("Trim Intro (s)", 0.0, 10.0, 0.0)
    offset = st.slider("Jumpback Offset (s)", 0.0, 5.0, 2.0)
    
    stimuli_manifest = [
        ("Full-Range", None, None, "raw", 8),
        ("Low-Pass", None, 1000, "low", 8),
        ("High-Pass", 1000, None, "high", 8),
        ("500Hz NBN", 420, 595, "band", 8),
        ("1000Hz NBN", 841, 1189, "band", 8),
        ("2000Hz NBN", 1682, 2378, "band", 8),
        ("4000Hz NBN", 3364, 4757, "band", 8),
        ("500Hz FRESH", 450, 550, "band", 20),
        ("1000Hz FRESH", 900, 1100, "band", 20),
        ("2000Hz FRESH", 1800, 2200, "band", 20),
        ("4000Hz FRESH", 3600, 4400, "band", 20)
    ]
    
    tab1, tab2 = st.tabs(["🎛️ LIVE PRESENTATION", "📦 BATCH EXPORT"])
    
    with tab1:
        left, mid, right = st.columns(3)
        for i, (lab, low, high, typ, ordr) in enumerate(stimuli_manifest):
            buf = process_audio_buffer(file_bytes, low, high, typ, ordr, trim)
            col = [left, mid, right][i % 3]
            with col:
                render_audiometer_channel(lab, buf, lab, offset)
                
    with tab2:
        for (lab, low, high, typ, ordr) in stimuli_manifest:
            buf = process_audio_buffer(file_bytes, low, high, typ, ordr, trim)
            st.download_button(f"Save {lab}", buf, file_name=f"{lab}.wav")

# Batch Upload
new_files = st.file_uploader("Upload", accept_multiple_files=True)
if new_files:
    for f in new_files: st.session_state.session_tracks[f.name] = f.read()
    st.rerun()
