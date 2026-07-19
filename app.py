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
# CONFIGURATION & THEMES
# ==============================================================================

st.set_page_config(page_title="Neilio's VRA Toolkit", page_icon="🎧", layout="wide", initial_sidebar_state="collapsed")

THEMES = {
    "Clinical Dark": {"bg": "#0f172a", "card": "#1e293b", "text": "#f8fafc", "border": "#334155", "accent": "#38bdf8"},
    "Apple Minimalist": {"bg": "#f5f5f7", "card": "#ffffff", "text": "#1d1d1f", "border": "#d2d2d7", "accent": "#0071e3"}
}

if "theme" not in st.session_state: st.session_state.theme = "Apple Minimalist"
current_theme = THEMES[st.session_state.theme]

local_storage = LocalStorage()
LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
if "session_tracks" not in st.session_state: st.session_state.session_tracks = {}
stored_favs = local_storage.getItem("favorites")
if "favorites" not in st.session_state: st.session_state.favorites = stored_favs if stored_favs else []

# ==============================================================================
# CSS & STYLE INJECTION
# ==============================================================================

st.markdown(f"""
    <style>
        .stApp {{ background-color: {current_theme['bg']} !important; color: {current_theme['text']}; }}
        .block-container {{ padding: 2rem !important; }}
        audio {{ height: 40px !important; margin-bottom: 8px !important; width: 100%; }}
        .audiogram-ruler {{ display: flex; justify-content: space-between; font-family: -apple-system, sans-serif; font-size: 1.1rem; color: {current_theme['accent']}; margin: 20px 0; padding: 0 40px; border-bottom: 2px solid {current_theme['border']}; }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# AUDIO PROCESSING ENGINE
# ==============================================================================

def butter_filter_sos(low, high, fs, filter_type='band', order=8):
    nyq = 0.5 * fs
    if filter_type == 'low': sos = butter(order, high/nyq, btype='low', output='sos')
    elif filter_type == 'high': sos = butter(order, low/nyq, btype='high', output='sos')
    else: sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sos

def get_stats(data):
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))
    return 20 * np.log10(peak / rms) if rms > 0 else 0

def rms_normalize(data, target_db=-20.0):
    rms = np.sqrt(np.mean(data**2))
    return data * (10**(target_db/20.0) / rms) if rms > 0 else data

def apply_compression(data, threshold_db=-20.0):
    max_amp = 10**(threshold_db/20.0)
    return np.clip(data, -max_amp, max_amp)

def process_audio_buffer(file_source, lowcut, highcut, filter_type, trim=0.0, compress=False):
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f: bytes_data = f.read()
        is_mp3 = file_source.lower().endswith('.mp3')
    else:
        bytes_data = file_source.read()
        is_mp3 = file_source.name.lower().endswith('.mp3')
        file_source.seek(0)
    
    if is_mp3:
        audio = AudioSegment.from_file(io.BytesIO(bytes_data), format="mp3").set_frame_rate(44100).set_channels(1)
        data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
        fs = 44100
    else: data, fs = sf.read(io.BytesIO(bytes_data))
    if trim > 0: data = data[int(trim*fs):]
    if filter_type != 'raw': data = sosfilt(butter_filter_sos(lowcut, highcut, fs, filter_type), data)
    if compress: data = apply_compression(data)
    data = rms_normalize(data)
    buf = io.BytesIO()
    sf.write(buf, data, fs, format='WAV')
    buf.seek(0)
    return buf

def render_audiometer_channel(label, audio_buffer, element_key, preroll):
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    html_code = f"""
    <div style="background-color: {current_theme['card']}; border: 1px solid {current_theme['border']}; border-radius: 12px; padding: 12px; text-align: center; color: {current_theme['text']}; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <div style="font-family: -apple-system, sans-serif; font-size: 0.9rem; font-weight: 600; margin-bottom: 8px;">{label}</div>
        <div style="display:flex; justify-content:center; gap:5px; margin-bottom:8px;">
            <div style="width:30%; background:{current_theme['bg']}; border-radius:6px; border:1px solid {current_theme['border']}; padding:2px;">
                <svg viewBox="0 0 100 50"><path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="{current_theme['border']}" stroke-width="4"/><line id="needle_{element_key}" x1="50" y1="45" x2="50" y2="10" stroke="#ef4444" stroke-width="2" style="transform-origin: 50% 45%; transform: rotate(-45deg);"/></svg>
            </div>
            <div id="vu_container_{element_key}" style="width:70%; background:{current_theme['bg']}; border:1px solid {current_theme['border']}; border-radius:6px; display:flex; align-items:flex-end; gap:1px; height:40px;">
                {"".join(['<div class="vu_bar_' + element_key + '" style="flex:1; height:10%; background:' + current_theme['accent'] + ';"></div>' for _ in range(32)])}
            </div>
        </div>
        <audio id="audio_{element_key}" src="data:audio/wav;base64,{audio_base64}" controls style="width:100%;"></audio>
        <script>
            (function() {{
                const a = document.getElementById('audio_{element_key}');
                const b = document.querySelectorAll('.vu_bar_{element_key}');
                const n = document.getElementById('needle_{element_key}');
                let ctx, ana, rot = -45;
                a.onplay = async () => {{
                    if(!ctx) {{ ctx = new (window.AudioContext || window.webkitAudioContext)(); ana = ctx.createAnalyser(); let src = ctx.createMediaElementSource(a); src.connect(ana); ana.connect(ctx.destination); ana.fftSize=2048; }}
                    if(ctx.state === 'suspended') await ctx.resume();
                    let d = new Uint8Array(ana.frequencyBinCount);
                    function u() {{
                        if(!a.paused) {{
                            ana.getByteFrequencyData(d);
                            let sum = 0;
                            b.forEach((el, i) => {{
                                let val = (d[i*4] || 0) / 255.0; sum += val;
                                el.style.height = (10 + (Math.min(1.0, val * 1.5) * 80)) + "%";
                            }});
                            rot += ((-45 + (sum/32*90)) - rot) * 0.3;
                            n.style.transform = 'rotate(' + rot + 'deg)';
                            requestAnimationFrame(u);
                        }}
                    }}
                    u();
                }};
            }})();
        </script>
    </div>
    """
    st.components.v1.html(html_code, height=300)

# ==============================================================================
# UI LOGIC & LAYOUT
# ==============================================================================

st.session_state.theme = st.selectbox("UI Aesthetic Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme))

all_tr = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))] + list(st.session_state.session_tracks.keys())
search = st.text_input("🔍 Search Library:")
sel = st.selectbox("Library Selection:", ["-- Select --"] + [t for t in all_tr if search.lower() in t.lower()])

if sel != "-- Select --":
    # 1. BANNER & CONTROLS
    st.markdown(f"<div style='background:{current_theme['card']}; border:2px solid {current_theme['accent']}; padding:15px; text-align:center; color:{current_theme['accent']}; font-weight:bold; border-radius:10px;'>ACTIVE SIGNAL: {sel}</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⭐ Toggle Favorite"):
            if sel in st.session_state.favorites: st.session_state.favorites.remove(sel)
            else: st.session_state.favorites.append(sel)
            local_storage.setItem("favorites", st.session_state.favorites); st.rerun()
    with c2:
        ui = st.radio("Mode", ["LIVE", "EXPORT"], horizontal=True)
    with c3:
        if st.button("🔊 Calibration Tone"): st.audio(generate_calibration_tone(), format="audio/wav")

    # 2. SETTINGS
    if sel in os.listdir(LIBRARY_DIR): src_p = os.path.join(LIBRARY_DIR, sel)
    else: src_p = io.BytesIO(st.session_state.session_tracks[sel])
    
    if isinstance(src_p, str): data, _ = sf.read(src_p)
    else: data, _ = sf.read(src_p); src_p.seek(0)
    
    st.write(f"**Audit:** Dynamic Range: {get_stats(data):.2f} dB")
    compress = st.checkbox("Apply Compression")
    trim = st.slider("Trim Start (s)", 0.0, 10.0, 0.0)
    preroll = st.slider("Pre-roll (s)", 0.0, 5.0, 2.0)

    # 3. VISUALIZERS
    manifest = [
        {"l": "Broadband", "low": 20, "high": 20000, "type": "raw", "s": "BB"},
        {"l": "Low-Pass", "low": 20, "high": 1000, "type": "low", "s": "LP"},
        {"l": "High-Pass", "low": 1000, "high": 20000, "type": "high", "s": "HP"},
        {"l": "500Hz", "low": 420, "high": 595, "type": "band", "s": "500"},
        {"l": "1000Hz", "low": 841, "high": 1189, "type": "band", "s": "1000"},
        {"l": "2000Hz", "low": 1682, "high": 2378, "type": "band", "s": "2000"},
        {"l": "4000Hz", "low": 3364, "high": 4757, "type": "band", "s": "4000"}
    ]
    
    if ui == "LIVE":
        r1, r2, r3 = st.columns(3)
        for i, item in enumerate(manifest[:3]):
            with [r1, r2, r3][i]:
                buf = process_audio_buffer(src_p, item["low"], item["high"], item["type"], trim, compress)
                render_audiometer_channel(item["l"], buf, item["s"], preroll)
        st.markdown("<div class='audiogram-ruler'><span>500Hz</span><span>1kHz</span><span>2kHz</span><span>4kHz</span></div>", unsafe_allow_html=True)
        r_bpf = st.columns(4)
        for i, item in enumerate(manifest[3:]):
            with r_bpf[i]:
                buf = process_audio_buffer(src_p, item["low"], item["high"], item["type"], trim, compress)
                render_audiometer_channel(item["l"], buf, item["s"], preroll)
    else:
        if st.button("📦 DOWNLOAD FULL SET (.ZIP)"):
            z_b = io.BytesIO()
            with zipfile.ZipFile(z_b, "w") as z:
                for item in manifest:
                    buf = process_audio_buffer(src_p, item["low"], item["high"], item["type"], trim, compress)
                    z.writestr(f"{sel}_{item['s']}.wav", buf.getvalue())
            st.download_button("Save Archive", z_b.getvalue(), f"{sel}_set.zip", "application/zip")

for _ in range(30): st.write("\n")
