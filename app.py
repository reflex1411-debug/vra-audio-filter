import base64
import io
import os
import re
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment, effects
from scipy.signal import butter, sosfiltfilt
import soundfile as sf
import streamlit as st
from streamlit_local_storage import LocalStorage
import yt_dlp

# ==============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ==============================================================================

st.set_page_config(
    page_title="Neilio's VRA Toolkit",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
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

if "cal_measured_dba" not in st.session_state:
    st.session_state.cal_measured_dba = 70.0
if "cal_dial_level" not in st.session_state:
    st.session_state.cal_dial_level = 70.0

if "selected_track" not in st.session_state:
    st.session_state.selected_track = "-- Select --"

# ==============================================================================
# 2. HELPER & ANALYSIS FUNCTIONS
# ==============================================================================


def extract_youtube_id(url):
    pattern = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def calculate_purity_metric(
    data, lowcut=None, highcut=None, filter_type="raw", fs=44100
):
    fft_vals = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1.0 / fs)
    total_energy = np.sum(fft_vals**2)

    if total_energy == 0:
        return {"val": 0.0, "label": "THD"}

    if filter_type == "band" and lowcut and highcut:
        in_band_mask = (freqs >= lowcut) & (freqs <= highcut)
        out_of_band_energy = np.sum(fft_vals[~in_band_mask] ** 2)
        leakage_pct = (out_of_band_energy / total_energy) * 100.0
        return {"val": round(float(leakage_pct), 2), "label": "Leak"}
    else:
        peak_idx = np.argmax(fft_vals)
        fundamental_energy = fft_vals[peak_idx] ** 2
        harmonic_energy = total_energy - fundamental_energy

        if fundamental_energy == 0:
            return {"val": 0.0, "label": "THD"}

        thd = (np.sqrt(harmonic_energy) / np.sqrt(fundamental_energy)) * 100.0
        return {"val": round(float(min(thd, 100.0)), 2), "label": "THD"}


def render_spectrum_plot(data, fs=44100, label=""):
    fft_vals = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1.0 / fs)

    fft_db = 20 * np.log10(np.maximum(fft_vals, 1e-6))
    fft_db = fft_db - np.max(fft_db)

    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    ax.plot(freqs, fft_db, color="#38bdf8", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_xlim(20, 20000)
    ax.set_ylim(-80, 5)

    ax.set_title(
        f"Spectral Density Matrix - {label}",
        color="#ffffff",
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
    )
    ax.set_xlabel(
        "Frequency (Hz)", color="#ffffff", fontsize=8, fontweight="bold"
    )
    ax.set_ylabel(
        "Magnitude (dB)", color="#ffffff", fontsize=8, fontweight="bold"
    )

    ax.tick_params(colors="#ffffff", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#475569")

    ax.grid(True, which="both", color="#334155", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    return fig


# ==============================================================================
# 3. CSS STYLING
# ==============================================================================

st.markdown(
    """
    <style>
        .stApp { background-color: #0f172a !important; }
        .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
        
        html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, label {
            color: #ffffff !important;
        }

        .stTextInput label, .stSelectbox label, .stSlider label, .stNumberInput label, .stCheckbox span {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        .st-emotion-cache-1e0sspq, .stExpander details summary p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        .card { 
            background-color: #1e293b; border: 1px solid #475569; 
            border-radius: 12px; padding: 16px; margin-bottom: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.4); text-align: center; 
        }
        
        audio { height: 40px !important; margin-bottom: 12px !important; margin-top: 4px !important; width: 100%; }
        
        .audiogram-ruler {
            display: flex; justify-content: space-between; font-family: monospace; font-size: 1rem;
            color: #fbbf24; margin: 20px 0; padding: 0 40px; border-bottom: 2px solid #fbbf24;
        }

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
            margin: 0 auto;
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
            color: #38bdf8 !important;
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
                <path d="M70,30 C60,25 45,35 40,45 C38,42 32,30 20,25 C25,38 35,45 38,48 C30,52 15,55 5,52 C18,58 32,58 39,53 C41,60 45,75 42,90 C48,75 50,60 48,52 C55,48 85,32 95,30 C85,32 75,32 70,30 Z" fill="#ffffff"/>
            </svg>
            <div style="text-align: left;">
                <h1 style="color: #ffffff !important; margin: 0; font-family: monospace; font-size: 1.8rem; letter-spacing: 1px; font-weight: 700;">NEILIO'S VRA CLINICAL STIMULI GENERATOR</h1>
                <div style="color: #cbd5e1 !important; font-size: 0.9rem; font-family: monospace; margin-top: 4px;">MODEL VRA-11 // MASTER ARCHIVE EDITION // RMS-CALIBRATED OUTPUT MATRIX</div>
            </div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 4. AUDIO PROCESSING ENGINE
# ==============================================================================


def download_youtube_audio(url, cookie_path=None):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": "library/%(title)s.%(ext)s",
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["mweb", "android", "web"],
            }
        },
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
    }

    if cookie_path and os.path.exists(cookie_path):
        ydl_opts["cookiefile"] = cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return os.path.splitext(filename)[0] + ".wav"
    except Exception as e:
        st.error(f"YouTube Download Error: {e}")
        return None


def butter_filter_sos(low, high, fs, filter_type="band", order=4):
    nyq = 0.5 * fs
    if filter_type == "low":
        sos = butter(order, high / nyq, btype="low", output="sos")
    elif filter_type == "high":
        sos = butter(order, low / nyq, btype="high", output="sos")
    elif filter_type == "band":
        sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    else:
        sos = None
    return sos


def calculate_rms(data):
    return np.sqrt(np.mean(data**2))


def calculate_audio_metrics(
    data, lowcut=None, highcut=None, filter_type="raw"
):
    peak = np.max(np.abs(data))
    rms = calculate_rms(data)

    peak_db = 20 * np.log10(peak) if peak > 0 else -100.0
    rms_db = 20 * np.log10(rms) if rms > 0 else -100.0
    crest_factor = peak_db - rms_db if (peak > 0 and rms > 0) else 0.0
    purity = calculate_purity_metric(
        data, lowcut=lowcut, highcut=highcut, filter_type=filter_type
    )

    return {
        "peak_db": round(float(peak_db), 2),
        "rms_db": round(float(rms_db), 2),
        "dr_span_db": round(float(crest_factor), 2),
        "purity_val": purity["val"],
        "purity_label": purity["label"],
    }


def equal_loudness_normalize(data, target_db=-20.0, peak_limit=0.98):
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


def apply_soft_knee_limiter(
    data, target_rms_db=-20.0, max_crest_factor_db=3.5, distortion_knee=1.2
):
    rms = calculate_rms(data)
    if rms == 0:
        return data

    target_linear = 10 ** (target_rms_db / 20.0)
    data_norm = data * (target_linear / rms)

    threshold = target_linear * (10 ** (max_crest_factor_db / 20.0))
    normalized_peaks = data_norm / threshold

    data_soft = np.tanh(normalized_peaks * distortion_knee) / distortion_knee
    data_soft = data_soft * threshold

    final_rms = calculate_rms(data_soft)
    if final_rms > 0:
        data_soft = data_soft * (target_linear / final_rms)

    return data_soft


@st.cache_data(show_spinner=False)
def generate_calibration_tone(freq=1000, duration=10.0, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    data = np.sin(2 * np.pi * freq * t).astype(np.float32)
    data = apply_soft_knee_limiter(
        data, target_rms_db=-20.0, max_crest_factor_db=3.0
    )
    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, fs, format="WAV", subtype="PCM_16")
    return virtual_file.getvalue()


@st.cache_data(
    show_spinner="Processing clean, natural-sounding VRA audio matrix..."
)
def process_audio_buffer(
    file_path,
    lowcut=None,
    highcut=None,
    filter_type="band",
    order=4,  # 4th order gives 48 dB/octave zero-phase cutoff without phase smearing
    trim=0.0,
    compress=True,
    comp_threshold=-22.0,
    comp_ratio=8.0,
    max_crest_factor=3.5,
    distortion_knee=1.2,
    noise_gain=0.0,
):
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    is_mp3 = file_path.lower().endswith(".mp3")

    if is_mp3:
        audio = (
            AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
            .set_frame_rate(44100)
            .set_channels(1)
        )
        data = (
            np.array(audio.get_array_of_samples(), dtype=np.float32)
            / (2**15)
        )
        fs = 44100
    else:
        data, fs = sf.read(io.BytesIO(file_bytes))

    if trim > 0:
        start_sample = int(trim * fs)
        if start_sample < len(data):
            data = data[start_sample:]

    # STEP 1: Time-Domain Zero-Phase Filtering (Prine & Natural Sounding)
    if filter_type != "raw":
        sos = butter_filter_sos(
            lowcut, highcut, fs, filter_type=filter_type, order=order
        )
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]):
                filtered_data[:, channel] = sosfiltfilt(
                    sos, data[:, channel]
                )
        else:
            filtered_data = sosfiltfilt(sos, data)
    else:
        filtered_data = data

    if noise_gain > 0:
        noise = np.random.normal(0, 0.05, len(filtered_data))
        if filter_type != "raw":
            sos_n = butter_filter_sos(
                lowcut, highcut, fs, filter_type=filter_type, order=order
            )
            noise = sosfiltfilt(sos_n, noise)
        filtered_data = filtered_data + (noise * noise_gain)

    # STEP 2: Post-Filter Soft-Knee Dynamic Clamping
    if compress:
        filtered_data = apply_soft_knee_limiter(
            filtered_data,
            target_rms_db=-20.0,
            max_crest_factor_db=max_crest_factor,
            distortion_knee=distortion_knee,
        )

    # STEP 3: Loudness Normalization
    final_data = equal_loudness_normalize(filtered_data, target_db=-20.0)

    # Compute Metrics
    metrics = calculate_audio_metrics(
        final_data, lowcut=lowcut, highcut=highcut, filter_type=filter_type
    )

    virtual_file = io.BytesIO()
    sf.write(virtual_file, final_data, fs, format="WAV", subtype="PCM_16")
    return virtual_file.getvalue(), metrics, final_data


def render_audiometer_channel(
    label,
    audio_bytes,
    metrics,
    element_key,
    preroll_offset,
    fft_gain,
    est_dba=None,
):
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_src = f"data:audio/wav;base64,{audio_base64}"

    bars_html = "".join(
        [
            f'<div class="vu_bar_{element_key}" style="flex: 1; height: 10%;'
            ' background-color: #10b981; border-radius: 1px; transition:'
            ' height 0.05s ease;"></div>'
            for _ in range(32)
        ]
    )

    dba_display = (
        f'<span style="color:#10b981;">Est. Sound Level: <b>{est_dba:.2f} dBA</b></span>'
        if est_dba is not None
        else "<span>Est. Sound Level: <b>-- dBA</b></span>"
    )

    purity_val = metrics.get("purity_val", 0.0)
    purity_lbl = metrics.get("purity_label", "THD")
    purity_color = "#10b981" if purity_val < 5.0 else "#ef4444"

    html_code = f"""
    <div class="card">
        <div style="font-family: monospace; font-size: 1.1rem; color: #ffffff; font-weight: bold; margin-bottom: 6px; letter-spacing: 0.5px;">{label}</div>
        
        <!-- Clinical Leveling, Purity & Estimated dBA Badge -->
        <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 4px 8px; margin-bottom: 4px; font-family: monospace; font-size: 0.72rem; color: #38bdf8; display: flex; justify-content: space-around;">
            <span>Span: <b style="color:#fbbf24;">±{metrics['dr_span_db']:.2f} dB</b></span>
            <span>Peak: <b>{metrics['peak_db']:.2f} dBFS</b></span>
            <span>RMS: <b>{metrics['rms_db']:.2f} dBFS</b></span>
            <span>{purity_lbl}: <b style="color:{purity_color};">{purity_val:.2f}%</b></span>
        </div>
        
        <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 4px 8px; margin-bottom: 10px; font-family: monospace; font-size: 0.8rem; display: flex; justify-content: center;">
            {dba_display}
        </div>

        <div id="vu_container_{element_key}" style="background-color: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; margin-bottom: 12px; display: flex; align-items: flex-end; justify-content: center; gap: 1px; height: 36px;">
            {bars_html}
        </div>
        <audio id="audio_{element_key}" src="{audio_src}" controls style="width:100%;"></audio>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px;">
            <button id="btn_toggle_{element_key}" onclick="togglePlayPause_{element_key}()" style="background-color: #10b981; color: white; border: none; padding: 12px 5px; border-radius: 8px; font-family: monospace; font-size: 0.9rem; cursor: pointer; font-weight: bold;">▶️ PLAY</button>
            <button onclick="stopAudio_{element_key}()" style="background-color: #ef4444; color: white; border: none; padding: 12px 5px; border-radius: 8px; font-family: monospace; font-size: 0.9rem; cursor: pointer; font-weight: bold;">⏹️ STOP</button>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;">
            <button onclick="var clickTime = document.getElementById('audio_{element_key}').currentTime; window.parent.sharedVraLoopPoint = Math.max(0, clickTime - {preroll_offset}); this.innerHTML='⚙️ MARKED'; setTimeout(()=>{{this.innerHTML='🎯 MARK'}}, 1500);" style="background-color: #f59e0b; color: #0f172a; border: none; padding: 12px 5px; border-radius: 8px; font-family: monospace; font-size: 0.9rem; cursor: pointer; font-weight: bold;">🎯 MARK</button>
            <button onclick="if(window.parent.sharedVraLoopPoint !== undefined) {{ var a = document.getElementById('audio_{element_key}'); a.currentTime = window.parent.sharedVraLoopPoint; a.play(); }}" style="background-color: #38bdf8; color: #0f172a; border: none; padding: 12px 5px; border-radius: 8px; font-family: monospace; font-size: 0.9rem; cursor: pointer; font-weight: bold;">🐇 JUMP</button>
        </div>
        <script>
            (function() {{
                const audio = document.getElementById('audio_{element_key}');
                const btnToggle = document.getElementById('btn_toggle_{element_key}');
                const bars = document.querySelectorAll('.vu_bar_{element_key}');
                let audioCtx, analyser, dataArray, source;

                window.togglePlayPause_{element_key} = function() {{
                    if (audio.paused) {{
                        audio.play();
                    }} else {{
                        audio.pause();
                    }}
                }};

                window.stopAudio_{element_key} = function() {{
                    audio.pause();
                    audio.currentTime = 0;
                }};

                audio.addEventListener('play', async () => {{
                    btnToggle.innerHTML = '⏸️ PAUSE';
                    btnToggle.style.backgroundColor = '#f59e0b';
                    
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
                                if (bar) {{
                                    const currentH = parseFloat(bar.style.height) || 10;
                                    const targetH = 10 + (val * 90 * {fft_gain});
                                    bar.style.height = (currentH + (targetH - currentH) * 0.4) + "%";
                                    bar.style.opacity = 0.3 + (val * 0.7);
                                }}
                            }}
                            requestAnimationFrame(update);
                        }}
                    }}
                    update();
                }});

                audio.addEventListener('pause', () => {{
                    btnToggle.innerHTML = '▶️ PLAY';
                    btnToggle.style.backgroundColor = '#10b981';
                }});

                audio.addEventListener('ended', () => {{
                    btnToggle.innerHTML = '▶️ PLAY';
                    btnToggle.style.backgroundColor = '#10b981';
                }});
            }})();
        </script>
    </div>
    """
    st.components.v1.html(html_code, height=380)


# ==============================================================================
# 5. UI LOGIC & LAYOUT
# ==============================================================================

if "filter_order" not in st.session_state:
    st.session_state.filter_order = 4
if "fft_gain" not in st.session_state:
    st.session_state.fft_gain = 1.0
if "comp_threshold" not in st.session_state:
    st.session_state.comp_threshold = -22.0
if "comp_ratio" not in st.session_state:
    st.session_state.comp_ratio = 8.0
if "max_crest_factor" not in st.session_state:
    st.session_state.max_crest_factor = 3.5
if "distortion_knee" not in st.session_state:
    st.session_state.distortion_knee = 1.2

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎛️ PRESENTATION DESK",
        "🎥 AD-HOC YOUTUBE PLAYER",
        "📦 EXPORT & DOWNLOADER",
        "⚙️ EXPERT CONFIG",
    ]
)

with tab4:
    st.subheader("⚙️ Expert Filter & Dynamic Span Settings")
    st.session_state.filter_order = st.slider(
        "Filter Order (Butterworth steepness per pass)", 2, 8, 4, 1
    )
    st.session_state.fft_gain = st.slider(
        "FFT Visualizer Sensitivity", 0.5, 5.0, 1.0, 0.1
    )
    st.session_state.comp_threshold = st.slider(
        "Compressor Threshold (dB)", -35.0, -10.0, -22.0, 1.0
    )
    st.session_state.comp_ratio = st.slider(
        "Compressor Ratio", 2.0, 16.0, 8.0, 1.0
    )
    st.session_state.max_crest_factor = st.slider(
        "Target Peak-to-RMS Span Ceiling (dB)", 2.0, 6.0, 3.5, 0.5
    )
    st.session_state.distortion_knee = st.slider(
        "Soft-Clipping Curve (Distortion Mitigation)", 0.8, 2.0, 1.2, 0.1
    )

with tab2:
    st.subheader("🎥 Ad-Hoc YouTube Media Player")

    adhoc_yt_url = st.text_input(
        "🔗 YouTube Media URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        key="adhoc_player_input",
    )

    if adhoc_yt_url:
        video_id = extract_youtube_id(adhoc_yt_url)
        if video_id:
            left_pad, player_col, right_pad = st.columns([1, 2, 1])
            with player_col:
                with st.container(border=True):
                    st.video(f"https://www.youtube.com/watch?v={video_id}")
                    st.success("Media loaded successfully.")
        else:
            st.error("Invalid YouTube URL format.")

with tab1:
    with st.container(border=True):
        with st.expander("🛠️ SYSTEM CALIBRATION & SOUND FIELD LEVEL CALCULATOR"):
            cal_col1, cal_col2 = st.columns(2)
            with cal_col1:
                st.session_state.cal_dial_level = st.number_input(
                    "Audiometer Dial Setting during Sound Field Measurement (dB HL):",
                    value=70.0,
                    step=5.0,
                )
            with cal_col2:
                st.session_state.cal_measured_dba = st.number_input(
                    "Measured Sound Level Meter Output (dBA):",
                    value=70.0,
                    step=0.5,
                )

            if st.button("🔊 GENERATE 1kHz CALIBRATION TONE (-20dBFS)"):
                cal_bytes = generate_calibration_tone()
                st.audio(cal_bytes, format="audio/wav")
                st.success("Calibration active.")

            target_test_dial = st.number_input(
                "🎯 Active Test Dial Setting (dB HL):",
                value=st.session_state.cal_dial_level,
                step=5.0,
            )

            calculated_dba = st.session_state.cal_measured_dba + (
                target_test_dial - st.session_state.cal_dial_level
            )
            st.info(
                f"📊 **Calculated Acoustic Output Level:** **{calculated_dba:.2f} dBA**"
            )

        compress_toggle = st.checkbox(
            "Lock Equal Loudness & Uniform Dynamic Span Across All Channels",
            value=True,
        )
        noise_gain = st.slider("Noise Floor Gain (NBN)", 0.0, 0.5, 0.0, 0.05)

        all_tracks = sorted(
            [
                f
                for f in os.listdir(LIBRARY_DIR)
                if f.lower().endswith((".mp3", ".wav"))
            ]
        )

        lib_col1, lib_col2 = st.columns([1, 1])
        with lib_col1:
            search_query = st.text_input(
                "🔍 Search Library:",
                placeholder="Type track name and press Enter...",
            )

        if search_query:
            filtered_tracks = [
                f for f in all_tracks if search_query.lower() in f.lower()
            ]
        else:
            filtered_tracks = all_tracks

        if len(filtered_tracks) == 1:
            st.session_state.selected_track = filtered_tracks[0]

        options = ["-- Select --"] + filtered_tracks
        if st.session_state.selected_track not in options:
            st.session_state.selected_track = "-- Select --"

        with lib_col2:
            sel = st.selectbox(
                "Select Signal:",
                options,
                index=options.index(st.session_state.selected_track),
                key="track_select_box",
            )
            st.session_state.selected_track = sel

        if sel != "-- Select --":
            active_source = os.path.join(LIBRARY_DIR, sel)
            trim = st.slider("Trim Start (s)", 0.0, 10.0, 0.0, 0.5)
            preroll = st.slider("Pre-roll (s)", 0.0, 5.0, 2.0, 0.1)

            manifest = [
                {
                    "label": "Broadband",
                    "low": 20,
                    "high": 20000,
                    "type": "raw",
                    "suffix": "BB",
                },
                {
                    "label": "Low-Pass",
                    "low": 20,
                    "high": 1000,
                    "type": "low",
                    "suffix": "LP",
                },
                {
                    "label": "High-Pass",
                    "low": 1000,
                    "high": 20000,
                    "type": "high",
                    "suffix": "HP",
                },
                {
                    "label": "500Hz BPF",
                    "low": 420,
                    "high": 595,
                    "type": "band",
                    "suffix": "500",
                },
                {
                    "label": "1000Hz BPF",
                    "low": 841,
                    "high": 1189,
                    "type": "band",
                    "suffix": "1000",
                },
                {
                    "label": "2000Hz BPF",
                    "low": 1682,
                    "high": 2378,
                    "type": "band",
                    "suffix": "2000",
                },
                {
                    "label": "4000Hz BPF",
                    "low": 3364,
                    "high": 4757,
                    "type": "band",
                    "suffix": "4000",
                },
            ]

            cols = st.columns(3)
            row1_items = [
                m
                for m in manifest
                if "BPF" not in m["label"]
                and "raw" not in m["type"]
                or m["label"] == "Broadband"
            ]
            for i, item in enumerate(row1_items):
                with cols[i]:
                    buf_bytes, metrics, raw_array = process_audio_buffer(
                        active_source,
                        item["low"],
                        item["high"],
                        item["type"],
                        order=st.session_state.filter_order,
                        trim=trim,
                        compress=compress_toggle,
                        comp_threshold=st.session_state.comp_threshold,
                        comp_ratio=st.session_state.comp_ratio,
                        max_crest_factor=st.session_state.max_crest_factor,
                        distortion_knee=st.session_state.distortion_knee,
                        noise_gain=noise_gain,
                    )
                    render_audiometer_channel(
                        item["label"],
                        buf_bytes,
                        metrics,
                        item["suffix"],
                        preroll,
                        st.session_state.fft_gain,
                        est_dba=calculated_dba,
                    )

            st.markdown(
                f"""
                <div class='marquee'><span>{sel}</span></div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div"
                " class='audiogram-ruler'><span>500Hz</span><span>1kHz</span><span>2kHz</span><span>4kHz</span></div>",
                unsafe_allow_html=True,
            )

            cols2 = st.columns(4)
            row2_items = [m for m in manifest if "BPF" in m["label"]]
            for i, item in enumerate(row2_items):
                with cols2[i]:
                    buf_bytes, metrics, raw_array = process_audio_buffer(
                        active_source,
                        item["low"],
                        item["high"],
                        item["type"],
                        order=st.session_state.filter_order,
                        trim=trim,
                        compress=compress_toggle,
                        comp_threshold=st.session_state.comp_threshold,
                        comp_ratio=st.session_state.comp_ratio,
                        max_crest_factor=st.session_state.max_crest_factor,
                        distortion_knee=st.session_state.distortion_knee,
                        noise_gain=noise_gain,
                    )
                    render_audiometer_channel(
                        item["label"],
                        buf_bytes,
                        metrics,
                        item["suffix"],
                        preroll,
                        st.session_state.fft_gain,
                        est_dba=calculated_dba,
                    )

            with st.expander("📈 SIGNAL PURITY & SPECTRAL ANALYSIS INSPECTOR"):
                spec_channel = st.selectbox(
                    "Inspect Band Frequency Response:",
                    [m["label"] for m in manifest],
                )
                selected_item = next(
                    m for m in manifest if m["label"] == spec_channel
                )

                _, _, band_array = process_audio_buffer(
                    active_source,
                    selected_item["low"],
                    selected_item["high"],
                    selected_item["type"],
                    order=st.session_state.filter_order,
                    trim=trim,
                    compress=compress_toggle,
                    comp_threshold=st.session_state.comp_threshold,
                    comp_ratio=st.session_state.comp_ratio,
                    max_crest_factor=st.session_state.max_crest_factor,
                    distortion_knee=st.session_state.distortion_knee,
                    noise_gain=noise_gain,
                )

                fig = render_spectrum_plot(
                    band_array, label=selected_item["label"]
                )
                st.pyplot(fig)

with tab3:
    st.subheader("📦 Bulk Export & YouTube Downloader")
    yt_url = st.text_input("🔗 URL:")
    cookie_file = st.file_uploader("Upload cookies.txt", type=["txt"])

    if yt_url and st.button("Download"):
        with st.spinner("Downloading..."):
            cookie_path = None
            if cookie_file:
                cookie_path = "library/cookies.txt"
                with open(cookie_path, "wb") as f:
                    f.write(cookie_file.getbuffer())

            if download_youtube_audio(yt_url, cookie_path):
                st.success("Downloaded.")
                st.rerun()

    if sel != "-- Select --":
        if st.button("📦 DOWNLOAD FULL SET (.ZIP)"):
            zip_b = io.BytesIO()
            with zipfile.ZipFile(zip_b, "w") as z:
                for item in manifest:
                    buf_bytes, _, _ = process_audio_buffer(
                        active_source,
                        item["low"],
                        item["high"],
                        item["type"],
                        order=st.session_state.filter_order,
                        trim=trim,
                        compress=compress_toggle,
                        comp_threshold=st.session_state.comp_threshold,
                        comp_ratio=st.session_state.comp_ratio,
                        max_crest_factor=st.session_state.max_crest_factor,
                        distortion_knee=st.session_state.distortion_knee,
                        noise_gain=noise_gain,
                    )
                    z.writestr(f"{sel}_{item['suffix']}.wav", buf_bytes)
            st.download_button(
                "Click to Save Archive",
                zip_b.getvalue(),
                f"{sel}_set.zip",
                "application/zip",
            )
