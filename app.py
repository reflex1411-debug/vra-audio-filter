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

# ==============================================================================
# 1. CONFIGURATION AND INITIALIZATION
# ==============================================================================

# Initialize layout with Apple Minimalist aesthetic preference
st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define visual theme dictionary for Apple Minimalist and Clinical Dark modes
THEMES = {
    "Apple Minimalist": {
        "bg": "#f5f5f7", "card": "#ffffff", "text": "#1d1d1f", 
        "border": "#d2d2d7", "accent": "#0071e3", "shadow": "0 4px 12px rgba(0,0,0,0.05)"
    },
    "Clinical Dark": {
        "bg": "#0f172a", "card": "#1e293b", "text": "#f8fafc", 
        "border": "#334155", "accent": "#38bdf8", "shadow": "0 4px 12px rgba(0,0,0,0.3)"
    }
}

# Ensure session state persistence for themes
if "theme" not in st.session_state: 
    st.session_state.theme = "Apple Minimalist"
current_theme = THEMES[st.session_state.theme]

# Storage and Library Setup
local_storage = LocalStorage()
LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
if "session_tracks" not in st.session_state: st.session_state.session_tracks = {}
if "favorites" not in st.session_state: 
    st.session_state.favorites = local_storage.getItem("favorites") or []

# ==============================================================================
# 2. STYLING INJECTION (CSS)
# ==============================================================================

st.markdown(f"""
    <style>
        .stApp {{ background-color: {current_theme['bg']} !important; color: {current_theme['text']}; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
        .block-container {{ padding: 2rem !important; }}
        audio {{ height: 40px !important; margin-bottom: 8px !important; width: 100%; }}
        .audiogram-ruler {{ 
            display: flex; justify-content: space-between; 
            font-family: -apple-system, sans-serif; font-size: 1.1rem; 
            color: {current_theme['accent']}; margin: 30px 0; padding: 0 40px; 
            border-bottom: 2px solid {current_theme['border']}; 
        }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. AUDIO PROCESSING ENGINE
# ==============================================================================

def butter_filter_sos(low, high, fs, filter_type='band', order=8):
    """Generates Second-Order Sections (SOS) for Butterworth filtering."""
    nyq = 0.5 * fs
    if filter_type == 'low': sos = butter(order, high/nyq, btype='low', output='sos')
    elif filter_type == 'high': sos = butter(order, low/nyq, btype='high', output='sos')
    else: sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sos

def get_stats(data):
    """Calculates peak-to-RMS ratio for clinical audit."""
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))
    return 20 * np.log10(peak / rms) if rms > 0 else 0

def rms_normalize(data, target_db=-20.0):
    """Normalizes signals to maintain consistent -20dBFS RMS levels."""
    rms = np.sqrt(np.mean(data**2))
    return data * (10**(target_db/20.0) / rms) if rms > 0 else data

def apply_compression(data, threshold_db=-20.0):
    """Applies hard-clip compression to tighten dynamic range."""
    max_amp = 10**(threshold_db/20.0)
    return np.clip(data, -max_amp, max_amp)

def process_audio_buffer(file_source, lowcut, highcut, filter_type, trim=0.0, compress=False):
    """Main signal processing pipeline: Load -> Trim -> Filter -> Compress -> Normalize."""
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f: bytes_data = f.read()
        is_mp3 = file_source.lower().endswith('.mp3')
    else:
        file_source.seek(0)
        bytes_data = file_source.read()
        is_mp3 = file_source.name.lower().endswith('.mp3')
    
    if is_mp3:
        audio = AudioSegment.from_file(io.BytesIO(bytes_data), format="mp3").set_frame_rate(44100).set_channels(1)
        data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
        fs = 44100
    else: data, fs = sf.read(io.BytesIO(bytes_data))
    
    if trim > 0: data = data[int(trim*fs):]
    if filter_type != 'raw':
        data = sosfilt(butter_filter_sos(lowcut, highcut, fs, filter_type), data)
    if compress: data = apply_compression(data)
    data = rms_normalize(data)
    
    buf = io.BytesIO()
    sf.write(buf, data, fs, format='WAV')
    buf.seek(0)
    return buf

# ==============================================================================
# 4. COMPONENT RENDERING
# ==============================================================================

def render_channel(label, buf, key, preroll):
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    html = f"""
    <div style="background:{current_theme['card']}; border:1px solid {current_theme['border']}; border-radius:12px; padding:12px; text-align:center; box-shadow: {current_theme['shadow']};">
        <div style="font-size:0.9rem; font-weight:600; margin-bottom:8px; color:{current_theme['text']};">{label}</div>
        <div style="display:flex; height:50px; margin-bottom:8px; gap:4px;">
            <div style="width:30%; background:{current_theme['bg']}; border:1px solid {current_theme['border']}; border-radius:6px;">
                <svg viewBox="0 0 100 50"><path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="{current_theme['border']}" stroke-width="4"/><line id="n_{key}" x1="50" y1="45" x2="50" y2="10" stroke="#ef4444" stroke-width="2" style="transform-origin:50% 45%; transform:rotate(-45deg);"/></svg>
            </div>
            <div id="v_{key}" style="width:70%; background:{current_theme['bg']}; border:1px solid {current_theme['border']}; border-radius:6px; display:flex; align-items:flex-end; gap:1px;">
                {''.join(['<div class="b_'+key+'" style="flex:1; background:'+current_theme['accent']+'; height:10%;"></div>' for _ in range(16)])}
            </div>
        </div>
        <audio id="a_{key}" src="data:audio/wav;base64,{audio_b64}" style="width:100%; height:30px;"></audio>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:3px; margin-top:5px;">
            <button onclick="let a=document.getElementById('a_{key}'); if(a.paused){{a.play();}}else{{a.pause();}}" style="font-size:9px; cursor:pointer;">▶️/⏸️</button>
            <button onclick="let a=document.getElementById('a_{key}'); a.pause(); a.currentTime=0;" style="font-size:9px; cursor:pointer;">⏹️</button>
            <button onclick="window.parent.lp_{key} = document.getElementById('a_{key}').currentTime - {preroll};" style="font-size:9px; cursor:pointer;">🔴MARK</button>
            <button onclick="let a=document.getElementById('a_{key}'); a.currentTime = window.parent.lp_{key}; a.play();" style="font-size:9px; cursor:pointer;">🐇JUMP</button>
        </div>
        <script>
            (function(){{
                let a=document.getElementById('a_{key}'); let ana;
                a.onplay = async () => {{
                    if(!ana){{ let ctx=new AudioContext(); ana=ctx.createAnalyser(); ctx.createMediaElementSource(a).connect(ana); ana.fftSize=64; }}
                    let d=new Uint8Array(ana.frequencyBinCount);
                    function u(){{ 
                        ana.getByteFrequencyData(d); 
                        document.querySelectorAll('.b_{key}').forEach((b,i)=>{{ b.style.height=(d[i]/2.5)+'%'; }});
                        document.getElementById('n_{key}').style.transform='rotate('+(d[0]/2-45)+'deg)';
                        if(!a.paused) requestAnimationFrame(u); 
                    }} u();
                }}
            }})();
        </script>
    </div>
    """
    st.components.v1.html(html, height=260)

# ==============================================================================
# 5. UI LAYOUT LOGIC
# ==============================================================================

# Theme selector
st.session_state.theme = st.selectbox("Select UI Aesthetic", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme))

# Upper Instrument Control Panel
with st.expander("🛠️ INSTRUMENT CONTROL PANEL", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("🔊 Calibration Tone"): st.audio(generate_calibration_tone(), format="audio/wav")
    compress = c2.checkbox("Enable 3-5dB Compression")
    trim = c3.slider("Trim Start (s)", 0.0, 5.0, 0.0)
    preroll = c4.slider("Jump Pre-roll (s)", 0.0, 5.0, 2.0)

# Track Selection
all_files = [f for f in os.listdir(LIBRARY_DIR) if f.endswith(('.mp3', '.wav'))]
sel = st.selectbox("Signal Selection", ["-- Select --"] + all_files)

# Logic for data handling
if sel != "-- Select --":
    src = os.path.join(LIBRARY_DIR, sel)
    if os.path.exists(src):
        with open(src, 'rb') as f: data, _ = sf.read(io.BytesIO(f.read()))
        
        # Row 1: Broadband, LP, HP
        r1 = st.columns(3)
        manifest_r1 = [
            {"l": "Broadband", "low": 20, "high": 20000, "type": "raw", "s": "BB"},
            {"l": "Low-Pass", "low": 20, "high": 1000, "type": "low", "s": "LP"},
            {"l": "High-Pass", "low": 1000, "high": 20000, "type": "high", "s": "HP"}
        ]
        for i, item in enumerate(manifest_r1):
            with r1[i]:
                render_channel(item["l"], process_audio_buffer(src, item["low"], item["high"], item["type"], trim, compress), item["s"], preroll)
        
        # Active Signal Banner (Placed between rows as requested)
        st.markdown(f"""
            <div style="background:{current_theme['accent']}; color:white; padding:15px; text-align:center; 
            border-radius:10px; font-weight:bold; margin: 20px 0; font-size: 1.2rem;">
            ACTIVE SIGNAL: {sel} // Dynamic Range: {get_stats(data):.2f} dB
            </div>
        """, unsafe_allow_html=True)
        
        # Ruler
        st.markdown("<div class='audiogram-ruler'><span>500Hz</span><span>1kHz</span><span>2kHz</span><span>4kHz</span></div>", unsafe_allow_html=True)
        
        # Row 2: 500Hz, 1000Hz, 2000Hz, 4000Hz
        r2 = st.columns(4)
        manifest_r2 = [
            {"l": "500Hz", "low": 420, "high": 595, "type": "band", "s": "500"},
            {"l": "1000Hz", "low": 841, "high": 1189, "type": "band", "s": "1000"},
            {"l": "2000Hz", "low": 1682, "high": 2378, "type": "band", "s": "2000"},
            {"l": "4000Hz", "low": 3364, "high": 4757, "type": "band", "s": "4000"}
        ]
        for i, item in enumerate(manifest_r2):
            with r2[i]:
                render_channel(item["l"], process_audio_buffer(src, item["low"], item["high"], item["type"], trim, compress), item["s"], preroll)

# Padding to ensure UI stability
for _ in range(40): st.write("\n")
