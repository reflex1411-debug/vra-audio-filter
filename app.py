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
# CONFIGURATION & INITIALIZATION
# ==============================================================================

st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

local_storage = LocalStorage()
LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
if "session_tracks" not in st.session_state: st.session_state.session_tracks = {}
stored_favs = local_storage.getItem("favorites")
if "favorites" not in st.session_state: st.session_state.favorites = stored_favs if stored_favs else []

# ==============================================================================
# CSS & STYLE INJECTION
# ==============================================================================

st.markdown("""
    <style>
        .stApp { background-color: #0f172a !important; }
        .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
        audio { height: 40px !important; margin-bottom: 12px !important; margin-top: 4px !important; width: 100%; }
        .audiogram-ruler { display: flex; justify-content: space-between; font-family: monospace; font-size: 1.2rem; color: #fbbf24; margin-top: 20px; margin-bottom: 20px; padding: 0 40px; border-bottom: 2px solid #fbbf24; }
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

def rms_normalize(data, target_db=-20.0):
    rms = np.sqrt(np.mean(data**2))
    return data * (10**(target_db/20.0) / rms) if rms > 0 else data

def process_audio_buffer(file_source, lowcut, highcut, filter_type, trim=0.0):
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
    data = rms_normalize(data)
    buf = io.BytesIO()
    sf.write(buf, data, fs, format='WAV')
    buf.seek(0)
    return buf

def render_audiometer_channel(label, audio_buffer, element_key, preroll):
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    html_code = f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 12px; text-align: center;">
        <div style="font-family: monospace; font-size: 1.1rem; color: #f8fafc; font-weight: bold; margin-bottom: 8px;">{label}</div>
        <div style="display:flex; justify-content:center; gap:10px; margin-bottom:10px;">
            <div style="width:40%; background:#0f172a; border-radius:8px; border:1px solid #334155; padding:5px;">
                <svg viewBox="0 0 100 50"><path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="#475569" stroke-width="4"/><line id="needle_{element_key}" x1="50" y1="45" x2="50" y2="10" stroke="#ef4444" stroke-width="2" style="transform-origin: 50% 45%; transform: rotate(-45deg);"/></svg>
            </div>
            <div id="vu_container_{element_key}" style="width:60%; background:#0f172a; border:1px solid #334155; display:flex; align-items:flex-end; gap:1px; height:50px;">
                {"".join(['<div class="vu_bar_' + element_key + '" style="flex:1; height:10%; background:#10b981;"></div>' for _ in range(32)])}
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
                    if(!ctx) {{ ctx = new AudioContext(); ana = ctx.createAnalyser(); ctx.createMediaElementSource(a).connect(ana); ana.fftSize=2048; }}
                    let d = new Uint8Array(ana.frequencyBinCount);
                    function u() {{
                        if(!a.paused) {{
                            ana.getByteFrequencyData(d);
                            let s = 0;
                            b.forEach((el, i) => {{
                                let val = (d[i*4] || 0) / 255.0;
                                s += val;
                                // 4k boost: index 24-32
                                let boost = (i > 24) ? 1.8 : (i > 16) ? 1.4 : 1.0;
                                let h = 10 + (Math.min(1.0, val * boost) * 90);
                                el.style.height = h + "%";
                            }});
                            rot += ((-45 + (s/32*90)) - rot) * 0.3;
                            n.style.transform = 'rotate(' + rot + 'deg)';
                            requestAnimationFrame(u);
                        }}
                    }}
                    u();
                }}
            }})();
        </script>
    </div>
    """
    st.components.v1.html(html_code, height=330)

# ==============================================================================
# UI LOGIC & LAYOUT
# ==============================================================================

with st.container(border=True):
    with st.expander("🛠️ SYSTEM CALIBRATION"):
        if st.button("🔊 GENERATE 1kHz CALIBRATION TONE"):
            st.audio(generate_calibration_tone(), format="audio/wav")

    ui = st.radio("", ["🎛️ LIVE PRESENTATION", "📦 EXPORT"], horizontal=True, label_visibility="collapsed")
    all_tr = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))] + list(st.session_state.session_tracks.keys())
    search = st.text_input("🔍 Search Library:", placeholder="Filter tracks...")
    filt = [t for t in all_tr if search.lower() in t.lower()]
    sel = st.selectbox("Select Track:", ["-- Select --"] + filt)
    
    if st.session_state.favorites:
        fav_cols = st.columns(max(len(st.session_state.favorites), 1))
        for i, f in enumerate(st.session_state.favorites):
            if fav_cols[i].button(f"🎵 {f[:15]}...", key=f"fav_{i}"): st.session_state.selected_track_override = f; st.rerun()

    if sel != "-- Select --":
        st.markdown(f"<div style='background: #0f172a; border:2px solid #38bdf8; padding:20px; text-align:center; color:#38bdf8; font-family:monospace; margin:15px 0;'>ACTIVE SIGNAL: {sel}</div>", unsafe_allow_html=True)
        if st.button("⭐ Toggle Favorite"):
            if sel in st.session_state.favorites: st.session_state.favorites.remove(sel)
            else: st.session_state.favorites.append(sel)
            local_storage.setItem("favorites", st.session_state.favorites); st.rerun()

        c1, c2 = st.columns(2)
        trim = c1.slider("Trim Start (s)", 0.0, 10.0, 0.0, 0.5)
        preroll = c2.slider("Pre-roll (s)", 0.0, 5.0, 2.0, 0.1)

        src = os.path.join(LIBRARY_DIR, sel) if sel in os.listdir(LIBRARY_DIR) else io.BytesIO(st.session_state.session_tracks[sel])
        manifest = [
            {"l": "Broadband", "low": 20, "high": 20000, "type": "raw", "s": "BB"},
            {"l": "Low-Pass (≤1kHz)", "low": 20, "high": 1000, "type": "low", "s": "LP"},
            {"l": "High-Pass (>1kHz)", "low": 1000, "high": 20000, "type": "high", "s": "HP"},
            {"l": "500Hz BPF", "low": 420, "high": 595, "type": "band", "s": "500"},
            {"l": "1000Hz BPF", "low": 841, "high": 1189, "type": "band", "s": "1000"},
            {"l": "2000Hz BPF", "low": 1682, "high": 2378, "type": "band", "s": "2000"},
            {"l": "4000Hz BPF", "low": 3364, "high": 4757, "type": "band", "s": "4000"}
        ]
        
        if "LIVE" in ui:
            r1, r2, r3 = st.columns(3)
            # Row 1: BB, LP, HP
            for i, item in enumerate(manifest[:3]):
                with [r1, r2, r3][i]:
                    buf = process_audio_buffer(src, item["low"], item["high"], item["type"], trim=trim)
                    render_audiometer_channel(item["l"], buf, item["s"], preroll)
            
            st.markdown("<div class='audiogram-ruler'><span>500Hz</span><span>1kHz</span><span>2kHz</span><span>4kHz</span></div>", unsafe_allow_html=True)
            # Row 2: BPFs
            r_bpf1, r_bpf2, r_bpf3, r_bpf4 = st.columns(4)
            for i, item in enumerate(manifest[3:]):
                with [r_bpf1, r_bpf2, r_bpf3, r_bpf4][i]:
                    buf = process_audio_buffer(src, item["low"], item["high"], item["type"], trim=trim)
                    render_audiometer_channel(item["l"], buf, item["s"], preroll)
        else:
            if st.button("📦 DOWNLOAD FULL SET (.ZIP)"):
                z_b = io.BytesIO()
                with zipfile.ZipFile(z_b, "w") as z:
                    for item in manifest:
                        buf = process_audio_buffer(src, item["low"], item["high"], item["type"], trim=trim)
                        z.writestr(f"{sel}_{item['s']}.wav", buf.getvalue())
                st.download_button("Save Archive", z_b.getvalue(), f"{sel}_set.zip", "application/zip")

for _ in range(50): st.write("\n")
