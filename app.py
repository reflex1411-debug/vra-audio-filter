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
        @keyframes pulse { 0% { height: 4px; } 100% { height: 16px; } }
    </style>
""", unsafe_allow_html=True)

# Faceplate Header
st.markdown("""
    <div style="background: linear-gradient(180deg, #334155 0%, #1e293b 100%); padding: 15px 25px; border-radius: 8px 8px 0px 0px; border: 2px solid #475569; border-bottom: none; display: flex; align-items: center; justify-content: space-between; box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="display: flex; align-items: center; gap: 20px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="38" height="38"><path d="M70,30 C60,25 45,35 40,45 C38,42 32,30 20,25 C25,38 35,45 38,48 C30,52 15,55 5,52 C18,58 32,58 39,53 C41,60 45,75 42,90 C48,75 50,60 48,52 C55,48 85,32 95,30 C85,32 75,32 70,30 Z" fill="#94a3b8"/></svg>
            <div style="text-align: left;">
                <h1 style="color: #f8fafc; margin: 0; font-family: monospace; font-size: 1.4rem; letter-spacing: 1px; font-weight: 700;">NEILIO'S VRA CLINICAL STIMULI GENERATOR</h1>
                <div style="color: #94a3b8; font-size: 0.75rem; font-family: monospace; margin-top: 2px;">MODEL VRA-11 // RMS-CALIBRATED OUTPUT MATRIX</div>
            </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center; background: #0f172a; padding: 6px 12px; border-radius: 4px; border: 1px solid #334155;">
            <div style="display: flex; align-items: center; gap: 5px; font-family: monospace; font-size: 0.7rem; color: #64748b;"><div style="width: 8px; height: 8px; background-color: #22c55e; border-radius: 50%; box-shadow: 0 0 8px #22c55e;"></div> SYS_READY</div>
            <div style="width: 1px; height: 12px; background: #334155;"></div>
            <div style="display: flex; align-items: center; gap: 5px; font-family: monospace; font-size: 0.7rem; color: #64748b;"><div style="width: 8px; height: 8px; background-color: #38bdf8; border-radius: 50%; box-shadow: 0 0 8px #38bdf8;"></div> RMS_FIXED (-20dBFS)</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---
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

def process_audio_buffer(file_source, lowcut=None, highcut=None, filter_type='band', order=8, trim_seconds=0.0):
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f: file_bytes = f.read()
    else:
        file_bytes = file_source.read()
        file_source.seek(0)
    
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_frame_rate(44100).set_channels(1)
    fs = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    
    if trim_seconds > 0:
        start_sample = int(trim_seconds * fs)
        data = data[start_sample:]
        
    if filter_type != 'raw':
        sos = butter_filter_sos(lowcut, highcut, fs, filter_type, order)
        data = sosfilt(sos, data)
        
    data = rms_normalize(data)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset, filter_type):
    # Determine visualizer bars based on filter type
    bars = ""
    if filter_type == 'raw': bars = "14px, 14px, 14px, 14px, 14px"
    elif filter_type == 'low': bars = "16px, 12px, 8px, 4px, 2px"
    elif filter_type == 'high': bars = "2px, 4px, 8px, 12px, 16px"
    else: bars = "4px, 8px, 16px, 16px, 8px, 4px"

    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    html_code = f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        <div style="font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold; margin-bottom: 6px;">
            {label}
            <div style="float: right; display: flex; align-items: flex-end; height: 16px; gap: 2px;">
                {"".join([f'<div style="width: 3px; height: {h}; background: #38bdf8; animation: pulse 0.5s infinite alternate;"></div>' for h in bars.split(', ')])}
            </div>
        </div>
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
    st.components.v1.html(html_code, height=170)

# --- UI ---
with st.container(border=True):
    ui_mode = st.radio("", ["🎛️ LIVE LINE-IN", "📦 BULK EXPORT"], horizontal=True, label_visibility="collapsed")
    st.markdown("<hr style='margin: 8px 0; border-color: #1e293b;' />", unsafe_allow_html=True)
    
    # Track Selection Logic
    stored_files = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    all_tracks_list = stored_files + list(st.session_state.session_tracks.keys())
    selected_track_name = st.selectbox("", ["-- Select Track --"] + all_tracks_list, label_visibility="collapsed")
    
    if selected_track_name != "-- Select Track --":
        active_target = os.path.join(LIBRARY_DIR, selected_track_name) if selected_track_name in stored_files else io.BytesIO(st.session_state.session_tracks[selected_track_name])
        trim_seconds = st.slider("Trim Start (s)", 0.0, 30.0, 0.0, step=0.5)
        preroll_offset = st.slider("Jumpback Offset (s)", 0.0, 5.0, 2.0, step=0.1)
        
        stimuli_manifest = [
            ("Full-Range", None, None, "raw", 8),
            ("Low-Pass", None, 1000, "low", 8),
            ("High-Pass", 1000, None, "high", 8),
            ("500Hz NBN", 420, 595, "band", 8),
            ("1000Hz NBN", 841, 1189, "band", 8),
            ("500Hz FRESH", 450, 550, "band", 20),
            ("1000Hz FRESH", 900, 1100, "band", 20)
        ]
        
        if "LIVE" in ui_mode:
            left, mid, right = st.columns(3)
            for i, (lab, low, high, typ, ordr) in enumerate(stimuli_manifest):
                buf = process_audio_buffer(active_target, low, high, typ, ordr, trim_seconds)
                col = [left, mid, right][i % 3]
                with col:
                    render_audiometer_channel(lab, buf, lab, preroll_offset, typ)
        else:
            for item in stimuli_manifest:
                buf = process_audio_buffer(active_target, item[1], item[2], item[3], item[4], trim_seconds)
                st.download_button(f"Save {item[0]}", buf, file_name=f"{item[0]}.wav")

# Batch Upload
new_uploads = st.file_uploader("Upload", type=["mp3", "wav"], accept_multiple_files=True)
if new_uploads:
    for upload in new_uploads: st.session_state.session_tracks[upload.name] = upload.read()
    st.rerun()
