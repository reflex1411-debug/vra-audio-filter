import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
from pydub import effects
import io
import zipfile
import os
import base64
import yt_dlp
from streamlit_local_storage import LocalStorage

# ==============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ==============================================================================

st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

local_storage = LocalStorage()
LIBRARY_DIR = "library"

if not os.path.exists(LIBRARY_DIR):
    os.makedirs(LIBRARY_DIR)

if "session_tracks" not in st.session_state:
    st.session_state.session_tracks = {}

stored_favs = local_storage.getItem("favorites")
if "favorites" not in st.session_state:
    st.session_state.favorites = stored_favs if stored_favs else []

# ==============================================================================
# 2. CSS & STYLE INJECTION
# ==============================================================================

st.markdown("""
    <style>
        .stApp { background-color: #0f172a !important; }
        .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
        .card { 
            background-color: #1e293b; border: 1px solid #334155; 
            border-radius: 12px; padding: 16px; margin-bottom: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); text-align: center; 
        }
        .audiogram-ruler {
            display: flex; justify-content: space-between; font-family: monospace; font-size: 1rem;
            color: #fbbf24; margin: 20px 0; padding: 0 40px; border-bottom: 2px solid #fbbf24;
        }
    </style>
    
    <div style="background: linear-gradient(180deg, #334155 0%, #1e293b 100%); padding: 20px 30px; border-radius: 16px; border: 2px solid #475569; display: flex; align-items: center; justify-content: space-between; box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="text-align: left;">
                <h1 style="color: #f8fafc; margin: 0; font-family: monospace; font-size: 1.8rem; letter-spacing: 1px; font-weight: 700;">NEILIO'S VRA CLINICAL STIMULI GENERATOR</h1>
                <div style="color: #94a3b8; font-size: 0.9rem; font-family: monospace; margin-top: 4px;">MODEL VRA-11 // MASTER ARCHIVE EDITION // RMS-CALIBRATED OUTPUT MATRIX</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. AUDIO PROCESSING ENGINE
# ==============================================================================

def download_youtube_audio(url, cookie_path=None):
    ydl_opts = {
        'format': 'bestaudio', 
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'outtmpl': 'library/yt_download.%(ext)s',
        'nocheckcertificate': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    if cookie_path: ydl_opts['cookiefile'] = cookie_path
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
        return "library/yt_download.wav"
    except Exception as e:
        st.error(f"YouTube Download Failed: {e}")
        return None

def butter_filter_sos(low, high, fs, filter_type='band', order=8):
    nyq = 0.5 * fs
    if filter_type == 'low': sos = butter(order, high/nyq, btype='low', output='sos')
    elif filter_type == 'high': sos = butter(order, low/nyq, btype='high', output='sos')
    else: sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sos

def calculate_rms(data): return np.sqrt(np.mean(data**2))

def rms_normalize(data, target_db=-20.0, peak_limit=0.95):
    current_rms = calculate_rms(data)
    if current_rms == 0: return data
    target_linear = 10 ** (target_db / 20.0)
    gain = target_linear / current_rms
    normalized_data = data * gain
    max_peak = np.max(np.abs(normalized_data))
    if max_peak > peak_limit: normalized_data = (normalized_data / max_peak) * peak_limit
    return normalized_data

def generate_calibration_tone(freq=1000, duration=10.0, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    data = np.sin(2 * np.pi * freq * t).astype(np.float32)
    data = rms_normalize(data, target_db=-20.0)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def process_audio_buffer(file_source, lowcut=None, highcut=None, filter_type='band', order=8, trim=0.0, compress=False, noise_gain=0.0):
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f: file_bytes = f.read()
        is_mp3 = file_source.lower().endswith('.mp3')
    else:
        file_bytes = file_source.read()
        is_mp3 = file_source.name.lower().endswith('.mp3')
        file_source.seek(0)
    
    if is_mp3:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3").set_frame_rate(44100).set_channels(1)
        data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
        fs = 44100
    else: data, fs = sf.read(io.BytesIO(file_bytes))
        
    if trim > 0:
        start_sample = int(trim * fs)
        if start_sample < len(data): data = data[start_sample:]
        
    sos = butter_filter_sos(lowcut, highcut, fs, filter_type=filter_type, order=order)
    if filter_type != 'raw':
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]): filtered_data[:, channel] = sosfilt(sos, data[:, channel])
        else: filtered_data = sosfilt(sos, data)
    else: filtered_data = data

    if noise_gain > 0:
        noise = np.random.normal(0, 0.05, len(filtered_data))
        if filter_type != 'raw': noise = sosfilt(sos, noise)
        filtered_data = filtered_data + (noise * noise_gain)

    if compress:
        int_data = (filtered_data * 32767).astype(np.int16)
        audio_seg = AudioSegment(int_data.tobytes(), frame_rate=int(fs), sample_width=2, channels=1)
        audio_seg = effects.compress_dynamic_range(audio_seg, threshold=-25.0, ratio=8.0, attack=5.0, release=150.0)
        filtered_data = np.array(audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0

    normalized_data = rms_normalize(filtered_data, target_db=-20.0)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, normalized_data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

# ==============================================================================
# 4. UI LAYOUT BUILDER
# ==============================================================================

tab1, tab2 = st.tabs(["🎛️ LIVE LINE-IN PRESENTATION DESK", "📦 EXPORT & YOUTUBE DOWNLOADER"])

with tab1:
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        compress_toggle = col1.checkbox("Enable Dynamic Range Compression")
        noise_gain = col2.slider("Noise Floor Gain (NBN)", 0.0, 0.5, 0.0, 0.05)
        
        all_tracks = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))]
        sel = st.selectbox("Select Signal:", ["-- Select --"] + all_tracks)
        
        if sel != "-- Select --":
            active_source = os.path.join(LIBRARY_DIR, sel)
            trim = st.slider("Trim Start (s)", 0.0, 10.0, 0.0, 0.5)
            
            manifest = [
                {"label": "Broadband", "low": 20, "high": 20000, "type": "raw", "suffix": "BB"},
                {"label": "Low-Pass", "low": 20, "high": 1000, "type": "low", "suffix": "LP"},
                {"label": "High-Pass", "low": 1000, "high": 20000, "type": "high", "suffix": "HP"},
                {"label": "500Hz BPF", "low": 420, "high": 595, "type": "band", "suffix": "500"},
                {"label": "1000Hz BPF", "low": 841, "high": 1189, "type": "band", "suffix": "1000"},
                {"label": "2000Hz BPF", "low": 1682, "high": 2378, "type": "band", "suffix": "2000"},
                {"label": "4000Hz BPF", "low": 3364, "high": 4757, "type": "band", "suffix": "4000"}
            ]
            
            cols = st.columns(3)
            for i, item in enumerate(manifest):
                with cols[i % 3]:
                    st.markdown(f"<div class='card'><strong>{item['label']}</strong>", unsafe_allow_html=True)
                    buf = process_audio_buffer(active_source, item["low"], item["high"], item["type"], trim=trim, compress=compress_toggle, noise_gain=noise_gain)
                    st.audio(buf, format="audio/wav")
                    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("YouTube Downloader")
    yt_url = st.text_input("🔗 URL:")
    if yt_url and st.button("Download"):
        with st.spinner("Downloading..."):
            if download_youtube_audio(yt_url): st.success("Downloaded.")
