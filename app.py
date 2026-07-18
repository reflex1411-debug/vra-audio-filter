import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile
import os
import base64

# --- SET UP ---
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

# --- CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #0f172a !important; }
        .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
        audio { height: 32px !important; margin-bottom: 8px !important; margin-top: 2px !important; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---
def butter_filter_sos(low, high, fs, filter_type='band', order=8):
    nyq = 0.5 * fs
    if filter_type == 'low': return butter(order, high/nyq, btype='low', output='sos')
    elif filter_type == 'high': return butter(order, low/nyq, btype='high', output='sos')
    return butter(order, [low/nyq, high/nyq], btype='band', output='sos')

def rms_normalize(data, target_db=-20.0, peak_limit=0.95):
    current_rms = np.sqrt(np.mean(data**2))
    if current_rms == 0: return data
    gain = (10 ** (target_db / 20.0)) / current_rms
    data = data * gain
    max_peak = np.max(np.abs(data))
    return (data / max_peak) * peak_limit if max_peak > peak_limit else data

@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda _: None})
def process_audio_buffer(file_bytes, lowcut, highcut, filter_type, order, trim_seconds):
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_frame_rate(44100).set_channels(1)
    fs = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    if trim_seconds > 0: data = data[int(trim_seconds * fs):]
    if filter_type != 'raw':
        sos = butter_filter_sos(lowcut, highcut, fs, filter_type, order)
        data = sosfilt(sos, data)
    data = rms_normalize(data)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset, border_color="#38bdf8"):
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    html_code = f"""
    <div style="background-color: #1e293b; border-top: 5px solid {border_color}; border-left: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155; border-radius: 0 0 6px 6px; padding: 10px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
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

# --- UI ---
st.markdown("<h1>NEILIO'S VRA CLINICAL GENERATOR</h1>", unsafe_allow_html=True)
stored_files = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))]
all_tracks = stored_files + list(st.session_state.session_tracks.keys())

# Favorites Bar
if st.session_state.favorites:
    cols = st.columns(len(st.session_state.favorites))
    for i, fav in enumerate(st.session_state.favorites):
        if cols[i].button(f"⭐ {fav}"): st.session_state.selected = fav

selected = st.selectbox("Select Active Track", ["-- Select --"] + all_tracks, key="selected")

if selected != "-- Select --":
    file_bytes = open(os.path.join(LIBRARY_DIR, selected), 'rb').read() if selected in stored_files else st.session_state.session_tracks[selected]
    if st.button("⭐ Toggle Favorite"):
        if selected in st.session_state.favorites: st.session_state.favorites.remove(selected)
        else: st.session_state.favorites.append(selected)
    
    trim = st.slider("Trim Start (s)", 0.0, 10.0, 0.0)
    offset = st.slider("Jumpback Offset (s)", 0.0, 5.0, 2.0)
    
    stimuli_manifest = [
        ("Full-Range", None, None, "raw", 8),
        ("500Hz NBN", 420, 595, "band", 8),
        ("1000Hz NBN", 841, 1189, "band", 8),
        ("2000Hz NBN", 1682, 2378, "band", 8),
        ("4000Hz NBN", 3364, 4757, "band", 8),
        ("500Hz FRESH", 450, 550, "band", 20),
        ("1000Hz FRESH", 900, 1100, "band", 20),
        ("2000Hz FRESH", 1800, 2200, "band", 20),
        ("4000Hz FRESH", 3600, 4400, "band", 20)
    ]
    
    freq_colors = {"500": "#f59e0b", "1000": "#38bdf8", "2000": "#a3e635", "4000": "#a855f7", "Full": "#94a3b8"}
    
    tab1, tab2 = st.tabs(["🎛️ LIVE PRESENTATION", "📦 BATCH EXPORT"])
    
    with tab1:
        cols = st.columns(3)
        for i, (lab, low, high, typ, ordr) in enumerate(stimuli_manifest):
            color = next((val for key, val in freq_colors.items() if key in lab), "#38bdf8")
            buf = process_audio_buffer(file_bytes, low, high, typ, ordr, trim)
            with cols[i % 3]:
                render_audiometer_channel(lab, buf, lab, offset, border_color=color)
                
    with tab2:
        for (lab, low, high, typ, ordr) in stimuli_manifest:
            buf = process_audio_buffer(file_bytes, low, high, typ, ordr, trim)
            st.download_button(f"Save {lab}", buf, file_name=f"{lab}.wav")

# Batch Upload
new_files = st.file_uploader("Batch Upload", accept_multiple_files=True)
if new_files:
    for f in new_files: st.session_state.session_tracks[f.name] = f.read()
    st.rerun()
