import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile
import os
import base64
from streamlit_local_storage import LocalStorage

# Set wide layout
st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Local Storage
local_storage = LocalStorage()

LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
if "session_tracks" not in st.session_state: st.session_state.session_tracks = {}

# Load favorites from local storage (Persistence)
stored_favs = local_storage.getItem("favorites")
if "favorites" not in st.session_state:
    st.session_state.favorites = stored_favs if stored_favs else []

# CSS Injection
st.markdown("""
    <style>
        .stApp { background-color: #0f172a !important; }
        .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
        audio { height: 32px !important; margin-bottom: 8px !important; margin-top: 2px !important; width: 100%; }
        .audiogram-ruler { display: flex; justify-content: space-between; font-family: monospace; font-size: 0.75rem; color: #fbbf24; margin-bottom: 10px; padding: 0 40px; border-bottom: 1px solid #fbbf24; }
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
    </div>
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

def process_audio_buffer(file_source, lowcut=None, highcut=None, filter_type='band', order=8, trim_seconds=0.0):
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f: file_bytes = f.read()
    else:
        file_bytes = file_source.read()
        file_source.seek(0)
    audio = AudioSegment.from_file(io.BytesIO(file_bytes)).set_frame_rate(44100).set_channels(1)
    data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    if trim_seconds > 0: data = data[int(trim_seconds * 44100):]
    if filter_type != 'raw':
        sos = butter_filter_sos(lowcut, highcut, 44100, filter_type, order)
        data = sosfilt(sos, data)
    data = rms_normalize(data)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, 44100, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset):
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    html_code = f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        <div style="font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold; margin-bottom: 6px;">{label}</div>
        <audio id="audio_{element_key}" src="data:audio/wav;base64,{audio_base64}" controls style="width:100%;"></audio>
        <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button onclick="document.getElementById('audio_{element_key}').play()" style="flex:1; background:#10b981; color:white; border:none; padding:6px; border-radius:4px; font-family:monospace; font-size:0.75rem;">▶ PLAY</button>
            <button onclick="document.getElementById('audio_{element_key}').pause()" style="flex:1; background:#ef4444; color:white; border:none; padding:6px; border-radius:4px; font-family:monospace; font-size:0.75rem;">⏸ PAUSE</button>
        </div>
        <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button onclick="var clickTime = document.getElementById('audio_{element_key}').currentTime; window.parent.sharedVraLoopPoint = Math.max(0, clickTime - {preroll_offset}); this.innerHTML='⚙️ MARKED';" style="flex:1; background:#f59e0b; border:none; padding:6px; border-radius:4px; font-family:monospace; font-size:0.75rem;">🔴 MARK POINT</button>
            <button onclick="document.getElementById('audio_{element_key}').currentTime=window.parent.sharedVraLoopPoint; document.getElementById('audio_{element_key}').play();" style="flex:1; background:#38bdf8; border:none; padding:6px; border-radius:4px; font-family:monospace; font-size:0.75rem;">↩️ JUMP BACK</button>
        </div>
    </div>
    """
    st.components.v1.html(html_code, height=165)

# --- UI ---
with st.container(border=True):
    ui_mode = st.radio("", ["🎛️ LIVE LINE-IN", "📦 BULK EXPORT"], horizontal=True, label_visibility="collapsed")
    st.markdown("<hr style='margin: 8px 0; border-color: #1e293b;' />", unsafe_allow_html=True)
    
    stored_files = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    all_tracks_list = stored_files + list(st.session_state.session_tracks.keys())
    
    # Favorites System with LocalStorage
    if st.session_state.favorites:
        cols = st.columns(max(len(st.session_state.favorites), 1))
        for i, fav in enumerate(st.session_state.favorites):
            if cols[i].button(f"🎵 {fav[:20]}...", key=f"fav_{i}"): st.session_state.override = fav

    selected_track_name = st.selectbox("", ["-- Select Track --"] + all_tracks_list, label_visibility="collapsed", index=0 if "override" not in st.session_state else all_tracks_list.index(st.session_state.override)+1)
    if "override" in st.session_state: del st.session_state.override
    
    # Favorites Toggle
    if selected_track_name != "-- Select Track --":
        if st.button("⭐ Add/Remove Favorite"):
            if selected_track_name in st.session_state.favorites: st.session_state.favorites.remove(selected_track_name)
            else: st.session_state.favorites.append(selected_track_name)
            local_storage.setItem("favorites", st.session_state.favorites)
            st.rerun()

    new_uploads = st.file_uploader("Upload", type=["mp3", "wav"], accept_multiple_files=True)
    if new_uploads:
        for u in new_uploads: st.session_state.session_tracks[u.name] = u.read()
        st.rerun()

    if selected_track_name != "-- Select Track --":
        active_target = os.path.join(LIBRARY_DIR, selected_track_name) if selected_track_name in stored_files else io.BytesIO(st.session_state.session_tracks[selected_track_name])
        c1, c2 = st.columns(2)
        trim_seconds = c1.slider("Trim Start (s)", 0.0, 30.0, 0.0, step=0.5)
        preroll_offset = c2.slider("Jumpback Offset (s)", 0.0, 5.0, 2.0, step=0.1)
        
        manifest = [
            {"label": "Full-Range", "low": None, "high": None, "type": "raw", "suffix": "Full"},
            {"label": "Low-Pass", "low": None, "high": 1000, "type": "low", "suffix": "LP"},
            {"label": "High-Pass", "low": 1000, "high": None, "type": "high", "suffix": "HP"},
            {"label": "500Hz NBN", "low": 420, "high": 595, "type": "band", "suffix": "500"},
            {"label": "1000Hz NBN", "low": 841, "high": 1189, "type": "band", "suffix": "1000"},
            {"label": "2000Hz NBN", "low": 1682, "high": 2378, "type": "band", "suffix": "2000"},
            {"label": "4000Hz NBN", "low": 3364, "high": 4757, "type": "band", "suffix": "4000"}
        ]
        
        if "LIVE" in ui_mode:
            st.subheader("Broadband & Filters")
            cols1 = st.columns(3)
            for i, item in enumerate(manifest[:3]):
                buf = process_audio_buffer(active_target, item["low"], item["high"], item["type"], trim_seconds=trim_seconds)
                render_audiometer_channel(item["label"], buf, item["suffix"], preroll_offset)
            
            st.divider()
            st.markdown("<div class='audiogram-ruler'><span>500Hz</span><span>1kHz</span><span>2kHz</span><span>4kHz</span></div>", unsafe_allow_html=True)
            cols2 = st.columns(4)
            for i, item in enumerate(manifest[3:]):
                buf = process_audio_buffer(active_target, item["low"], item["high"], item["type"], trim_seconds=trim_seconds)
                with cols2[i]: render_audiometer_channel(item["label"], buf, item["suffix"], preroll_offset)
        else:
            for item in manifest:
                buf = process_audio_buffer(active_target, item["low"], item["high"], item["type"], trim_seconds=trim_seconds)
                st.download_button(f"Save {item['label']}", buf, file_name=f"{item['label']}.wav")
