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

audio { height: 40px !important; margin-bottom: 12px !important; margin-top: 4px !important; width: 100%; }

.audiogram-ruler {
    display: flex; justify-content: space-between; font-family: monospace; font-size: 1rem;
    color: #fbbf24; margin: 20px 0; padding: 0 40px; border-bottom: 2px solid #fbbf24;
}

/* Dot Matrix Narrow Marquee */
.marquee {
    width: 60%;
    overflow: hidden;
    white-space: nowrap;
    box-sizing: border-box;
    background: #000;
    border: 2px solid #38bdf8;
    border-radius: 4px;
    padding: 8px 15px;
    height: 50px;
    display: flex;
    align-items: center;
}

.marquee span {
    display: inline-block;
    padding-left: 100%;
    animation: marquee 12s linear infinite;
    font-family: 'Courier New', Courier, monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
    text-shadow: 0 0 8px #38bdf8;
    text-transform: uppercase;
}

@keyframes marquee {
    0% { transform: translate(0, 0); }
    100% { transform: translate(-100%, 0); }
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

def download_youtube_audio(url, cookie_path=None):
    ydl_opts = {
        'format': 'bestaudio',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'outtmpl': 'library/yt_download.%(ext)s',
        'nocheckcertificate': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    if cookie_path:
        ydl_opts['cookiefile'] = cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "library/yt_download.wav"
    except Exception as e:
        st.error(f"YouTube Download Failed: {e}")
        return None

def butter_filter_sos(low, high, fs, filter_type='band', order=8):
    nyq = 0.5 * fs
    if filter_type == 'low':
        sos = butter(order, high/nyq, btype='low', output='sos')
    elif filter_type == 'high':
        sos = butter(order, low/nyq, btype='high', output='sos')
    else:
        sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sos

def calculate_rms(data):
    return np.sqrt(np.mean(data**2))

def rms_normalize(data, target_db=-20.0, peak_limit=0.95):
    current_rms = calculate_rms(data)
    if current_rms == 0:
        return data

    target_linear = 10 ** (target_db / 20.0)
    gain = target_linear / current_rms
    normalized_data = data * gain

    max_peak = np.max(np.abs(normalized_data))
    if max_peak > peak_limit:
        normalized_data = (normalized_data / max_peak) * peak_limit

    return normalized_data

def generate_calibration_tone(freq=1000, duration=10.0, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    data = np.sin(2 * np.pi * freq * t).astype(np.float32)
    data = rms_normalize(data, target_db=-20.0)

    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, fs, format='WAV', subtype='PCM_16')
    virtual_file.seek(0)
    return virtual_file

def process_audio_buffer(file_source, lowcut=None, highcut=None, filter_type='band', order=8, trim=0.0, compress=False, noise_gain=0.0):
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f:
            file_bytes = f.read()
        is_mp3 = file_source.lower().endswith('.mp3')
    else:
        file_bytes = file_source.read()
        is_mp3 = file_source.name.lower().endswith('.mp3')

    file_source.seek(0)

    if is_mp3:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3").set_frame_rate(44100).set_channels(1)
        data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
        fs = 44100
    else:
        data, fs = sf.read(io.BytesIO(file_bytes))

    if trim > 0:
        start_sample = int(trim * fs)
        if start_sample < len(data):
            data = data[start_sample:]

    sos = butter_filter_sos(lowcut, highcut, fs, filter_type=filter_type, order=order)

    if filter_type != 'raw':
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]):
                filtered_data[:, channel] = sosfilt(sos, data[:, channel])
        else:
            filtered_data = sosfilt(sos, data)
    else:
        filtered_data = data

    if noise_gain > 0:
        noise = np.random.normal(0, 0.05, len(filtered_data))
        if filter_type != 'raw':
            noise = sosfilt(sos, noise)
        filtered_data = filtered_data + (noise * noise_gain)

    if compress:
        int_data = (filtered_data * 32767).astype(np.int16)
        audio_seg = AudioSegment(int_data.tobytes(), frame_rate=int(fs), sample_width=2, channels=1)
        audio_seg = effects.compress_dynamic_range(audio_seg, threshold=-25.0, ratio=8.0, attack=5.0, release=150.0)
        filtered_data = np.array(audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0

    normalized_data = rms_normalize(filtered_data, target_db=-20.0)

    virtual_file = io.BytesIO()
    sf.write(virtual_file, normalized_data, fs, format='WAV', subtype='PCM_16')
    virtual_file.seek(0)
    return virtual_file

def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset, fft_gain):
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    audio_src = f"data:audio/wav;base64,{audio_base64}"

    html_code = f"""
    <div class="card">

        <div style="font-family: monospace; font-size: 1.1rem; color: #f8fafc; font-weight: bold; margin-bottom: 12px; letter-spacing: 0.5px;">
            {label}
        </div>

        <div id="vu_container_{element_key}" style="background-color: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; margin-bottom: 12px; display: flex; align-items: flex-end; justify-content: center; gap: 1px; height: 36px;">
            {"".join(['<div class="vu_bar_' + element_key + '" style="flex: 1; height: 10%; background-color: #10b981; border-radius: 1px; transition: height 0.05s ease;"></div>' for _ in range(32)])}
        </div>

        <!-- Audio element without native controls -->
        <audio id="audio_{element_key}" src="{audio_src}" style="width:100%;"></audio>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px;">
            <button onclick="document.getElementById('audio_{element_key}').play()" 
                style="background-color: #10b981; color: white; border: none; padding: 25px 5px; border-radius: 16px; font-family: monospace; font-size: 1rem; cursor: pointer; font-weight: bold;">
                ▶️ PLAY
            </button>

            <button onclick="var a = document.getElementById('audio_{element_key}'); a.pause(); a.currentTime = 0;" 
                style="background-color: #ef4444; color: white; border: none; padding: 25px 5px; border-radius: 16px; font-family: monospace; font-size: 1rem; cursor: pointer; font-weight: bold;">
                ⏹️ STOP
            </button>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;">
            <button onclick="var clickTime = document.getElementById('audio_{element_key}').currentTime; 
                             window.parent.sharedVraLoopPoint = Math.max(0, clickTime - {preroll_offset}); 
                             this.innerHTML='⚙️ MARKED'; 
                             setTimeout(()=>{{this.innerHTML='🎯 MARK'}}, 1500);" 
                style="background-color: #f59e0b; color: #0f172a; border: none; padding: 25px 5px; border-radius: 16px; font-family: monospace; font-size: 1rem; cursor: pointer; font-weight: bold;">
                🎯 MARK
            </button>

            <button onclick="if(window.parent.sharedVraLoopPoint !== undefined) {{ 
                                var a = document.getElementById('audio_{element_key}'); 
                                a.currentTime = window.parent.sharedVraLoopPoint; 
                                a.play(); }}" 
                style="background-color: #38bdf8; color: #0f172a; border: none; padding: 25px 5px; border-radius: 16px; font-family: monospace; font-size: 1rem; cursor: pointer; font-weight: bold;">
                🐇 JUMP
            </button>
        </div>

        <script>
        (function() {{
            const audio = document.getElementById('audio_{element_key}');
            const bars = document.querySelectorAll('.vu_bar_{element_key}');
            let audioCtx, analyser, dataArray, source;

            audio.addEventListener('play', async () => {{
                if (!audioCtx) {{
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    await audioCtx.resume();
                    analyser = audioCtx.createAnalyser();
                    source = audioCtx.createMediaElementSource(audio);
                    source.connect(analyser);
                    analyser.connect(audioCtx.destination);
                    analyser.fftSize = 2048;
                    dataArray = new Uint8Array(analyser.frequencyBinCount);
                }}

                function update() {{
                    if (!audio.paused) {{
                        analyser.getByteFrequencyData(dataArray);
                        for (let i = 0; i < 32; i++) {{
                            const val = (dataArray[i * 4] || 0) / 255.0;
                            const bar = bars[i];
                            const currentH = parseFloat(bar.style.height);
                            const targetH = 10 + (val * 90 * {fft_gain});
                            bar.style.height = (currentH + (targetH - currentH) * 0.4) + "%";
                            bar.style.opacity = 0.3 + (val * 0.7);
                        }}
                        requestAnimationFrame(update);
                    }}
                }}
                update();
            }});
        }})();
        </script>

    </div>
    """

    st.components.v1.html(html_code, height=365)

# ==============================================================================
# 4. UI LOGIC & LAYOUT
# ==============================================================================

if "filter_order" not in st.session_state:
    st.session_state.filter_order = 8

if "fft_gain" not in st.session_state:
    st.session_state.fft_gain = 1.0

tab1, tab2, tab3 = st.tabs(["🎛️ PRESENTATION DESK", "📦 EXPORT & DOWNLOADER", "⚙️ EXPERT CONFIG"])

with tab3:
    st.subheader("⚙️ Expert Filter Settings")
    st.session_state.filter_order = st.slider("Filter Order (Butterworth steepness)", 2, 16, 8, 2)
    st.session_state.fft_gain = st.slider("FFT Visualizer Sensitivity", 0.5, 5.0, 1.0, 0.1)

with tab1:
    with st.container(border=True):
        with st.expander("🛠️ SYSTEM CALIBRATION & TRANSDUCER CHECK"):
            if st.button("🔊 GENERATE 1kHz CALIBRATION TONE (-20dBFS)"):
                cal_buffer = generate_calibration_tone()
                st.audio(cal_buffer, format="audio/wav")
                st.success("Calibration active.")

        compress_toggle = st.checkbox("Enable Dynamic Range Compression")
        noise_gain = st.slider("Noise Floor Gain (NBN)", 0.0, 0.5, 0.0, 0.05)

        all_tracks = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))]

        # Search box for library
        query = st.text_input("🔍 Search Library")
        filtered_tracks = [t for t in all_tracks if query.lower() in t.lower()]

        sel = st.selectbox("Select Signal:", ["-- Select --"] + filtered_tracks)

        if sel != "-- Select --":
            active_source = os.path.join(LIBRARY_DIR, sel)

            trim = st.slider("Trim Start (s)", 0.0, 10.0, 0.0, 0.5)
            preroll = st.slider("Pre-roll (s)", 0.0, 5.0, 2.0, 0.1)

            manifest = [
                {"label": "Broadband", "low": 20, "high": 20000, "type": "raw", "suffix": "BB"},
                {"label": "Low-Pass", "low": 20, "high": 1000, "type": "low", "suffix": "LP"},
                {"label": "High-Pass", "low": 1000, "high": 20000, "type": "high", "suffix": "HP"},
                {"label": "500Hz BPF", "low": 420, "high": 595, "type": "band", "suffix": "500"},
                {"label": "1000Hz BPF", "low": 841, "high": 1189, "type": "band", "suffix": "1000"},
                {"label": "2000Hz BPF", "low": 1682, "high": 2378, "type": "band", "suffix": "2000"},
                {"label": "4000Hz BPF", "low": 3364, "high": 4757, "type": "band", "suffix": "4000"}
            ]

            # Row 1 Render (Broadband + low/high-pass)
            row1_items = [
                m for m in manifest
                if m["label"] == "Broadband" or m["type"] in ["low", "high"]
            ]

            cols = st.columns(3)
            for i, item in enumerate(row1_items):
                with cols[i]:
                    buf = process_audio_buffer(
                        active_source,
                        item["low"],
                        item["high"],
                        item["type"],
                        order=st.session_state.filter_order,
                        trim=trim,
                        compress=compress_toggle,
                        noise_gain=noise_gain
                    )
                    render_audiometer_channel(item["label"], buf, item["suffix"], preroll, st.session_state.fft_gain)

            # Animated active signal indicator
            st.markdown(f"""
            <div class='marquee'><span>{sel}</span></div>
            """, unsafe_allow_html=True)

            st.markdown(
                "<div class='audiogram-ruler'><span>500Hz</span><span>1kHz</span><span>2kHz</span><span>4kHz</span></div>",
                unsafe_allow_html=True
            )

            # Row 2 Render (BPFs)
            cols2 = st.columns(4)
            for i, item in enumerate([m for m in manifest if "BPF" in m["label"]]):
                with cols2[i]:
                    buf = process_audio_buffer(
                        active_source,
                        item["low"],
                        item["high"],
                        item["type"],
                        order=st.session_state.filter_order,
                        trim=trim,
                        compress=compress_toggle,
                        noise_gain=noise_gain
                    )
                    render_audiometer_channel(item["label"], buf, item["suffix"], preroll, st.session_state.fft_gain)

with tab2:
    st.subheader("📦 Bulk Export & YouTube Downloader")

    yt_url = st.text_input("🔗 URL:")
    cookie_file = st.file_uploader("Upload cookies.txt", type=["txt"])

    if yt_url and st.button("Download"):
        with st.spinner("Downloading..."):
            cookie_path = "library/cookies.txt"
            if cookie_file:
                with open(cookie_path, "wb") as f:
                    f.write(cookie_file.getbuffer())

            if download_youtube_audio(yt_url, cookie_path):
                st.success("Downloaded.")
                st.rerun()

    # ZIP export for selected signal
    if 'sel' in
