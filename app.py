import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
import io
import base64
from streamlit_local_storage import LocalStorage
import os

# ==============================================================================
# 1. CONFIGURATION AND INITIALIZATION
# ==============================================================================
st.set_page_config(page_title="Neilio's VRA Toolkit", page_icon="🎧", layout="wide", initial_sidebar_state="collapsed")

THEME = {
    "bg": "#0f172a", "card": "#1e293b", "text": "#f8fafc", 
    "border": "#334155", "accent": "#38bdf8", "bar": "#10b981"
}

LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)

# CSS for Clinical Dark Aesthetic
st.markdown(f"""
    <style>
        .stApp {{ background-color: {THEME['bg']} !important; color: {THEME['text']}; font-family: monospace; }}
        .card {{ background: {THEME['card']}; border: 1px solid {THEME['border']}; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .audiogram-ruler {{ display: flex; justify-content: space-between; font-family: monospace; font-size: 1.1rem; color: #fbbf24; margin: 25px 0; padding: 0 40px; border-bottom: 2px solid #fbbf24; }}
        button {{ width: 100%; padding: 5px; font-size: 10px !important; cursor: pointer; }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. AUDIO PROCESSING ENGINE
# ==============================================================================

def get_butter_sos(low, high, fs, ftype):
    nyq = 0.5 * fs
    if ftype == 'low': return butter(8, high/nyq, btype='low', output='sos')
    if ftype == 'high': return butter(8, low/nyq, btype='high', output='sos')
    return butter(8, [low/nyq, high/nyq], btype='band', output='sos')

def process_audio(file_source, low, high, ftype, trim=0.0, comp=False):
    """Processes audio buffer and returns bytes."""
    with open(file_source, 'rb') as f: data, fs = sf.read(io.BytesIO(f.read()))
    if trim > 0: data = data[int(trim*fs):]
    if ftype != 'raw': data = sosfilt(get_butter_sos(low, high, fs, ftype), data)
    if comp: data = np.clip(data, -0.2, 0.2)
    rms = np.sqrt(np.mean(data**2))
    if rms > 0: data = data * (10**(-20/20) / rms)
    buf = io.BytesIO()
    sf.write(buf, data, fs, format='WAV')
    buf.seek(0)
    return buf

# ==============================================================================
# 3. COMPONENT RENDERING
# ==============================================================================

def render_channel(label, buf, key, preroll):
    """Renders channel card with FFT and playback controls."""
    b64 = base64.b64encode(buf.getvalue()).decode()
    html = f"""
    <div class="card">
        <div style="font-weight:600; font-size:0.9rem; margin-bottom:10px;">{label}</div>
        <div id="v_{key}" style="background:#0f172a; border:1px solid {THEME['border']}; border-radius:4px; height:40px; display:flex; align-items:flex-end; gap:1px; margin-bottom:10px;">
            {''.join(['<div class="b_'+key+'" style="flex:1; background:'+THEME['bar']+'; height:10%;"></div>' for _ in range(16)])}
        </div>
        <audio id="a_{key}" src="data:audio/wav;base64,{b64}" controls style="width:100%; height:30px;"></audio>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:3px; margin-top:8px;">
            <button onclick="document.getElementById('a_{key}').play()">▶️</button>
            <button onclick="let a=document.getElementById('a_{key}'); a.pause(); a.currentTime=0;">⏹️</button>
            <button onclick="window.parent.lp_{key} = document.getElementById('a_{key}').currentTime - {preroll};">🔴MARK</button>
            <button onclick="let a=document.getElementById('a_{key}'); a.currentTime = window.parent.lp_{key}; a.play();">🐇JUMP</button>
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
                        if(!a.paused) requestAnimationFrame(u); 
                    }} u();
                }}
            }})();
        </script>
    </div>
    """
    st.components.v1.html(html, height=260)

# ==============================================================================
# 4. UI LAYOUT
# ==============================================================================

# Control Panel
with st.expander("🛠️ INSTRUMENT CONTROL PANEL", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.button("🔊 Calibration Tone")
    compress = col2.checkbox("3-5dB Compression")
    trim = col3.slider("Trim Start (s)", 0.0, 5.0, 0.0)
    preroll = col4.slider("Pre-roll (s)", 0.0, 5.0, 2.0)

# Track selection
files = [f for f in os.listdir(LIBRARY_DIR) if f.endswith(('.mp3', '.wav'))]
sel = st.selectbox("Select Signal", ["-- Select --"] + files)

if sel != "-- Select --":
    src = os.path.join(LIBRARY_DIR, sel)
    
    # Row 1
    r1 = st.columns(3)
    manifest_r1 = [
        {"l":"Broadband", "low":20, "high":20000, "t":"raw", "s":"BB"},
        {"l":"Low-Pass", "low":20, "high":1000, "t":"low", "s":"LP"},
        {"l":"High-Pass", "low":1000, "high":20000, "t":"high", "s":"HP"}
    ]
    for i, item in enumerate(manifest_r1):
        with r1[i]:
            render_channel(item["l"], process_audio(src, item["low"], item["high"], item["t"], trim, compress), item["s"], preroll)

    # Banner
    st.markdown(f"""
        <div style="background:{THEME['accent']}; color:white; padding:15px; text-align:center; 
        border-radius:8px; font-weight:bold; margin: 25px 0; font-size: 1.2rem;">
        ACTIVE SIGNAL: {sel}
        </div>
    """, unsafe_allow_html=True)
    
    # Ruler
    st.markdown("<div class='audiogram-ruler'><span>500Hz</span><span>1kHz</span><span>2kHz</span><span>4kHz</span></div>", unsafe_allow_html=True)
    
    # Row 2
    r2 = st.columns(4)
    manifest_r2 = [
        {"l":"500Hz", "low":420, "high":595, "t":"band", "s":"500"},
        {"l":"1000Hz", "low":841, "high":1189, "t":"band", "s":"1000"},
        {"l":"2000Hz", "low":1682, "high":2378, "t":"band", "s":"2000"},
        {"l":"4000Hz", "low":3364, "high":4757, "t":"band", "s":"4000"}
    ]
    for i, item in enumerate(manifest_r2):
        with r2[i]:
            render_channel(item["l"], process_audio(src, item["low"], item["high"], item["t"], trim, compress), item["s"], preroll)
