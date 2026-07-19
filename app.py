import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import os
import base64
from streamlit_local_storage import LocalStorage

# ==============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ==============================================================================

# Set wide layout for the clinical faceplate
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
            background: #1e293b; border: 1px solid #334155; 
            border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 10px;
        }
        .audiogram-ruler {
            display: flex; justify-content: space-between; font-family: monospace; font-size: 1rem;
            color: #fbbf24; margin-bottom: 12px; padding: 0 40px; border-bottom: 2px solid #fbbf24;
        }
    </style>
    
    <div style="background: linear-gradient(180deg, #334155 0%, #1e293b 100%); padding: 20px 30px; border-radius: 16px; border: 2px solid #475569; display: flex; align-items: center; justify-content: space-between; box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="display: flex; align-items: center; gap: 20px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="45" height="45">
                <path d="M70,30 C60,25 45,35 40,45 C38,42 32,30 20,25 C25,38 35,45 38,48 C30,52 15,55 5,52 C18,58 32,58 39,53 C41,60 45,75 42,90 C48,75 50,60 48,52 C55,48 85,32 95,30 C85,32 75,32 70,30 Z" fill="#94a3b8"/>
            </svg>
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

def butter_filter_sos(low, high, fs, filter_type='band', order=8):
    """Calculates Butterworth filter coefficients."""
    nyq = 0.5 * fs
    if filter_type == 'low': sos = butter(order, high/nyq, btype='low', output='sos')
    elif filter_type == 'high': sos = butter(order, low/nyq, btype='high', output='sos')
    else: sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sos

def calculate_rms(data):
    """Calculates Root Mean Square of audio data."""
    return np.sqrt(np.mean(data**2))

def rms_normalize(data, target_db=-20.0, peak_limit=0.95):
    """Normalizes audio to target RMS level."""
    current_rms = calculate_rms(data)
    if current_rms == 0: return data
    target_linear = 10 ** (target_db / 20.0)
    gain = target_linear / current_rms
    normalized_data = data * gain
    max_peak = np.max(np.abs(normalized_data))
    if max_peak > peak_limit:
        normalized_data = (normalized_data / max_peak) * peak_limit
    return normalized_data

def generate_calibration_tone(freq=1000, duration=10.0, fs=44100):
    """Generates a 1kHz calibration sine wave."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    data = np.sin(2 * np.pi * freq * t).astype(np.float32)
    data = rms_normalize(data, target_db=-20.0)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def process_audio_buffer(file_source, lowcut=None, highcut=None, filter_type='band', order=8, trim=0.0, compress=False):
    """Loads, filters, and normalizes audio buffers with software compression."""
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
        
    if filter_type != 'raw':
        sos = butter_filter_sos(lowcut, highcut, fs, filter_type=filter_type, order=order)
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]): filtered_data[:, channel] = sosfilt(sos, data[:, channel])
        else: filtered_data = sosfilt(sos, data)
    else: filtered_data = data

    # 3-5dB Soft Limiter (Dynamic Range Compression)
    if compress:
        filtered_data = np.tanh(filtered_data * 2.5) * 0.45

    normalized_data = rms_normalize(filtered_data, target_db=-20.0)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, normalized_data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

# ==============================================================================
# 4. UI LOGIC & LAYOUT
# ==============================================================================

with st.container(border=True):
    with st.expander("🛠️ SYSTEM CALIBRATION & TRANSDUCER CHECK"):
        if st.button("🔊 GENERATE 1kHz CALIBRATION TONE (-20dBFS)"):
            cal_buffer = generate_calibration_tone()
            st.audio(cal_buffer, format="audio/wav")
            st.success("Calibration active.")

    ui_mode = st.radio("", ["🎛️ LIVE LINE-IN PRESENTATION DESK", "📦 BULK EXPORT & FILE DOWNLOAD CENTER"], horizontal=True, label_visibility="collapsed")
    compress_toggle = st.checkbox("Enable 3-5dB Dynamic Range Compression")
    st.markdown("<hr style='margin: 8px 0; border-color: #1e293b;' />", unsafe_allow_html=True)

    all_tracks = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    search = st.text_input("🔍 Search Library (Filter by name):", placeholder="Start typing to filter tracks...")
    filtered = [t for t in all_tracks if search.lower() in t.lower()]
    sel = st.selectbox("Library Selection:", ["-- Select Track from Bank --"] + filtered)
    
    if sel != "-- Select Track from Bank --":
        st.markdown(f"<div style='background: #0f172a; border: 2px solid #38bdf8; border-radius: 12px; padding: 20px; text-align: center; color: #38bdf8; font-family: monospace; font-size: 1.3rem; margin: 15px 0;'>ACTIVE SIGNAL: {sel}</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        trim = c1.slider("Trim Start (s)", 0.0, 10.0, 0.0, 0.5)
        preroll = c2.slider("Pre-roll (s)", 0.0, 5.0, 2.0, 0.1)

        active_source = os.path.join(LIBRARY_DIR, sel)
        manifest = [
            {"label": "Broadband", "low": 20, "high": 20000, "type": "raw", "suffix": "BB"},
            {"label": "Low-Pass (≤1kHz)", "low": 20, "high": 1000, "type": "low", "suffix": "LP"},
            {"label": "High-Pass (>1kHz)", "low": 1000, "high": 20000, "type": "high", "suffix": "HP"},
            {"label": "500Hz BPF", "low": 420, "high": 595, "type": "band", "suffix": "500"},
            {"label": "1000Hz BPF", "low": 841, "high": 1189, "type": "band", "suffix": "1000"},
            {"label": "2000Hz BPF", "low": 1682, "high": 2378, "type": "band", "suffix": "2000"},
            {"label": "4000Hz BPF", "low": 3364, "high": 4757, "type": "band", "suffix": "4000"}
        ]
        
        # Rendering
        if "LIVE LINE-IN" in ui_mode:
            r1_c1, r1_c2, r1_c3 = st.columns(3)
            # Row 1 Rendering
            for i, item in enumerate([m for m in manifest if "BPF" not in m["label"] and "raw" not in m["type"] or m["label"] == "Broadband"]):
                with [r1_c1, r1_c2, r1_c3][i % 3]:
                    st.markdown(f"<div class='card'><strong>{item['label']}</strong>", unsafe_allow_html=True)
                    buf = process_audio_buffer(active_source, item["low"], item["high"], item["type"], trim=trim, compress=compress_toggle)
                    st.audio(buf, format="audio/wav")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='audiogram-ruler'><span>500Hz</span><span>1kHz</span><span>2kHz</span><span>4kHz</span></div>", unsafe_allow_html=True)
            
            r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
            # Row 2 Rendering
            for i, item in enumerate([m for m in manifest if "BPF" in m["label"]]):
                with [r2_c1, r2_c2, r2_c3, r2_c4][i]:
                    st.markdown(f"<div class='card'><strong>{item['label']}</strong>", unsafe_allow_html=True)
                    buf = process_audio_buffer(active_source, item["low"], item["high"], item["type"], trim=trim, compress=compress_toggle)
                    st.audio(buf, format="audio/wav")
                    st.markdown("</div>", unsafe_allow_html=True)

        else: # EXPORT MODE
            if st.button("📦 DOWNLOAD FULL SET (.ZIP)"):
                zip_b = io.BytesIO()
                with zipfile.ZipFile(zip_b, "w") as z:
                    for item in manifest:
                        buf = process_audio_buffer(active_source, item["low"], item["high"], item["type"], trim=trim, compress=compress_toggle)
                        z.writestr(f"{sel}_{item['suffix']}.wav", buf.getvalue())
                st.download_button("Click to Save Archive", zip_b.getvalue(), f"{sel}_set.zip", "application/zip")

for _ in range(55): st.write("\n")
