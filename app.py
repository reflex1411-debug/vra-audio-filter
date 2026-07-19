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

# Set wide layout to establish a comprehensive dual-channel audiometer faceplate
# This ensures that the instrument UI has adequate horizontal space for layout
st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize LocalStorage client for persisting user preferences across clinical sessions
# This allows the 'Favorites' deck to persist across browser reloads
local_storage = LocalStorage()

# Define the library directory constant for file path references
LIBRARY_DIR = "library"

# Ensure local persistence folder exists for storing library files
# This is where the permanent stimuli bank lives in the GitHub repository
if not os.path.exists(LIBRARY_DIR):
    os.makedirs(LIBRARY_DIR)

# Initialize system memory cache for tracking current loaded session tracks
# These are the ephemeral files uploaded during the current live session
if "session_tracks" not in st.session_state:
    st.session_state.session_tracks = {}

# Initialize system memory cache for favorite tracks tracking
# Load favorites from local storage, fallback to empty list if none exist
stored_favs = local_storage.getItem("favorites")
if "favorites" not in st.session_state:
    st.session_state.favorites = stored_favs if stored_favs else []

# ==============================================================================
# CSS & STYLE INJECTION (Verbose Formatting)
# ==============================================================================

# The following CSS blocks define the clinical "look and feel" of the instrument.
# We inject these styles to maintain a consistent VRA-11 look across the clinic.
st.markdown("""
    <style>
        /* Base page grounding mimicking hardware metal casing */
        .stApp {
            background-color: #0f172a !important;
        }
        /* Tighten gaps between elements for a dense instrument cluster feel */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
        }
        /* Make internal media players ultra-compact to match instrument aesthetic */
        audio {
            height: 40px !important;
            margin-bottom: 12px !important;
            margin-top: 4px !important;
            width: 100%;
        }
        /* Audiogram ruler styling for visual alignment */
        .audiogram-ruler {
            display: flex;
            justify-content: space-between;
            font-family: monospace;
            font-size: 1rem;
            color: #fbbf24;
            margin-bottom: 12px;
            padding: 0 40px;
            border-bottom: 2px solid #fbbf24;
        }
    </style>
    
    <!-- Faceplate Main Header -->
    <div style="background: linear-gradient(180deg, #334155 0%, #1e293b 100%); padding: 20px 30px; border-radius: 16px; border: 2px solid #475569; display: flex; align-items: center; justify-content: space-between; box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="display: flex; align-items: center; gap: 20px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="45" height="45">
                <path d="M70,30 C60,25 45,35 40,45 C38,42 32,30 20,25 C25,38 35,45 38,48 C30,52 15,55 5,52 C18,58 32,58 39,53 C41,60 45,75 42,90 C48,75 50,60 48,52 C55,48 85,32 95,30 C85,32 75,32 70,30 Z" fill="#94a3b8"/>
            </svg>
            <div style="text-align: left;">
                <h1 style="color: #f8fafc; margin: 0; font-family: monospace; font-size: 1.8rem; letter-spacing: 1px; font-weight: 700;">NEILIO'S VRA CLINICAL STIMULI GENERATOR</h1>
                <div style="color: #94a3b8; font-size: 0.9rem; font-family: monospace; margin-top: 4px;">MODEL VRA-11 // RMS-CALIBRATED OUTPUT MATRIX</div>
            </div>
        </div>
        <!-- Simulated Hardware Status LED Matrix -->
        <div style="display: flex; gap: 15px; align-items: center; background: #0f172a; padding: 10px 20px; border-radius: 8px; border: 1px solid #334155;">
            <div style="display: flex; align-items: center; gap: 8px; font-family: monospace; font-size: 0.85rem; color: #64748b;">
                <div style="width: 10px; height: 10px; background-color: #22c55e; border-radius: 50%; box-shadow: 0 0 10px #22c55e;"></div> SYS_READY
            </div>
            <div style="width: 1px; height: 16px; background: #334155;"></div>
            <div style="display: flex; align-items: center; gap: 8px; font-family: monospace; font-size: 0.85rem; color: #64748b;">
                <div style="width: 10px; height: 10px; background-color: #38bdf8; border-radius: 50%; box-shadow: 0 0 10px #38bdf8;"></div> RMS_FIXED (-20dBFS)
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# AUDIO PROCESSING ENGINE (Expanded & Documented)
# ==============================================================================

def butter_filter_sos(cutoff_low, cutoff_high, fs, filter_type='band', order=8):
    """
    Calculates Butterworth filter coefficients for signal conditioning.
    This provides the necessary frequency shaping for Narrowband Noise (NBN).
    """
    nyq = 0.5 * fs
    if filter_type == 'low':
        normal_cutoff = cutoff_high / nyq
        sos = butter(order, normal_cutoff, btype='low', output='sos')
    elif filter_type == 'high':
        normal_cutoff = cutoff_low / nyq
        sos = butter(order, normal_cutoff, btype='high', output='sos')
    else:
        low = cutoff_low / nyq
        high = cutoff_high / nyq
        sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def calculate_rms(data):
    """
    Calculates Root Mean Square of audio data to determine energy levels.
    Essential for ensuring consistent clinical presentation across stimuli.
    """
    return np.sqrt(np.mean(data**2))

def rms_normalize(data, target_db=-20.0, peak_limit=0.95):
    """
    Normalizes audio to a target RMS level, ensuring clinical repeatability.
    This prevents stimulus intensity variation between different source files.
    """
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
    """
    Generates a pure sine wave at 1000Hz for system calibration.
    This is the reference signal for verifying transducer output levels.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    data = np.sin(2 * np.pi * freq * t).astype(np.float32)
    data = rms_normalize(data, target_db=-20.0)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def process_audio_buffer(file_source, lowcut=None, highcut=None, filter_type='band', order=8, trim_seconds=0.0):
    """
    Loads, processes, filters, and normalizes audio buffers for output.
    Handles MP3 decoding and WAV processing internally.
    """
    # Data Ingestion Logic
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f: file_bytes = f.read()
        is_mp3 = file_source.lower().endswith('.mp3')
    else:
        file_bytes = file_source.read()
        is_mp3 = file_source.name.lower().endswith('.mp3')
        file_source.seek(0)
    
    # Decoding Logic
    if is_mp3:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
        audio = audio.set_frame_rate(44100).set_channels(1)
        fs = audio.frame_rate
        data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    else:
        data, fs = sf.read(io.BytesIO(file_bytes))
        
    # Trimming Logic
    if trim_seconds > 0:
        start_sample = int(trim_seconds * fs)
        if start_sample < len(data):
            if len(data.shape) > 1: data = data[start_sample:, :]
            else: data = data[start_sample:]
        
    # Filtering Logic
    if filter_type != 'raw':
        sos = butter_filter_sos(lowcut, highcut, fs, filter_type=filter_type, order=order)
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]):
                filtered_data[:, channel] = sosfilt(sos, data[:, channel])
        else:
            filtered_data = sosfilt(sos, data)
    else:
        filtered_data = data

    # Normalization Logic
    if len(filtered_data.shape) > 1:
        for channel in range(filtered_data.shape[1]):
            filtered_data[:, channel] = rms_normalize(filtered_data[:, channel], target_db=-20.0)
    else:
        filtered_data = rms_normalize(filtered_data, target_db=-20.0)
        
    # Final Export
    virtual_file = io.BytesIO()
    sf.write(virtual_file, filtered_data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset, filter_complexity):
    """
    Injects HTML5 audio components with interactive playback controls.
    Includes custom JavaScript Web Audio Analyser for realistic VU behavior.
    """
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    audio_src = f"data:audio/wav;base64,{audio_base64}"
    
    html_code = f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); text-align: center;">
        <div style="font-family: monospace; font-size: 1.1rem; color: #f8fafc; font-weight: bold; margin-bottom: 12px; letter-spacing: 0.5px;">{label}</div>
        
        <!-- Hardware Signal VU Display Subsystem -->
        <div id="vu_container_{element_key}" style="background-color: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; margin-bottom: 12px; display: flex; align-items: flex-end; justify-content: center; gap: 1px; height: 36px;">
            {"".join(['<div class="vu_bar_' + element_key + '" style="flex: 1; height: 10%; background-color: #10b981; border-radius: 1px; transition: height 0.05s ease;"></div>' for _ in range(32)])}
        </div>

        <audio id="audio_{element_key}" src="{audio_src}" controls style="width:100%;"></audio>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px;">
            <button onclick="var a = document.getElementById('audio_{element_key}'); if(a.paused) {{ a.play(); }} else {{ a.pause(); }}" style="background-color: #10b981; color: white; border: none; padding: 25px 5px; border-radius: 16px; font-family: monospace; font-size: 1rem; cursor: pointer; font-weight: bold;">▶️ PLAY / ⏸️ PAUSE</button>
            <button onclick="var a = document.getElementById('audio_{element_key}'); a.pause(); a.currentTime = 0;" style="background-color: #ef4444; color: white; border: none; padding: 25px 5px; border-radius: 16px; font-family: monospace; font-size: 1rem; cursor: pointer; font-weight: bold;">⏹️ STOP</button>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;">
            <button onclick="var clickTime = document.getElementById('audio_{element_key}').currentTime; window.parent.sharedVraLoopPoint = Math.max(0, clickTime - {preroll_offset}); this.innerHTML='⚙️ MARKED'; setTimeout(()=>{{this.innerHTML='🔴 MARK'}}, 1500);" style="background-color: #f59e0b; color: #0f172a; border: none; padding: 25px 5px; border-radius: 16px; font-family: monospace; font-size: 1rem; cursor: pointer; font-weight: bold;">🔴 MARK</button>
            <button onclick="if(window.parent.sharedVraLoopPoint !== undefined) {{ var a = document.getElementById('audio_{element_key}'); a.currentTime = window.parent.sharedVraLoopPoint; a.play(); }}" style="background-color: #38bdf8; color: #0f172a; border: none; padding: 25px 5px; border-radius: 16px; font-family: monospace; font-size: 1rem; cursor: pointer; font-weight: bold;">🐇 JUMP</button>
        </div>

        <script>
            (function() {{
                const audio = document.getElementById('audio_{element_key}');
                const bars = document.querySelectorAll('.vu_bar_{element_key}');
                let audioCtx, analyser, dataArray, source;
                
                audio.addEventListener('play', () => {{
                    if (!audioCtx) {{
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                        analyser = audioCtx.createAnalyser();
                        source = audioCtx.createMediaElementSource(audio);
                        source.connect(analyser);
                        analyser.connect(audioCtx.destination);
                        analyser.fftSize = 64; 
                        dataArray = new Uint8Array(analyser.frequencyBinCount);
                    }}
                    function update() {{
                        if (!audio.paused) {{
                            analyser.getByteFrequencyData(dataArray);
                            bars.forEach((bar, i) => {{
                                // Linear mapping of bins to bars with inertia decay
                                const val = (dataArray[i] || 0) / 255.0;
                                const currentH = parseFloat(bar.style.height);
                                const targetH = 10 + (val * 85);
                                bar.style.height = (currentH + (targetH - currentH) * 0.4) + "%";
                                bar.style.opacity = 0.3 + (val * 0.7);
                            }});
                            requestAnimationFrame(update);
                        }}
                    }}
                    update();
                }}, {{once: true}});
            }})();
        </script>
    </div>
    """
    st.components.v1.html(html_code, height=365)

# ==============================================================================
# UI LOGIC & LAYOUT (Restored Verbose Construction)
# ==============================================================================

with st.container(border=True):
    
    # System Calibration Panel
    with st.expander("🛠️ SYSTEM CALIBRATION & TRANSDUCER CHECK"):
        st.write("Generate a 1kHz pure sine tone to calibrate your sound level meter or transducer output level.")
        if st.button("🔊 GENERATE 1kHz CALIBRATION TONE (-20dBFS)"):
            cal_buffer = generate_calibration_tone()
            st.audio(cal_buffer, format="audio/wav")
            st.success("Calibration tone active. Signal normalized to -20dBFS.")

    # Functional System Mode Toggle Strip
    st.markdown("<div style='font-family: monospace; font-size: 1.0rem; color: #64748b; margin-bottom: 2px;'>[CONSOLE FUNCTION CONFIGURATION]</div>", unsafe_allow_html=True)
    ui_mode = st.radio(
        "",
        ["🎛️ LIVE LINE-IN PRESENTATION DESK", "📦 BULK EXPORT & FILE DOWNLOAD CENTER"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown("<hr style='margin: 8px 0; border-color: #1e293b;' />", unsafe_allow_html=True)

    # Initialize master file options tracking arrays
    stored_files = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    all_tracks_list = stored_files + list(st.session_state.session_tracks.keys())
    
    # --- FAVORITES DECK BAR ROW ---
    if st.session_state.favorites:
        st.markdown("<div style='font-family: monospace; font-size: 0.9rem; color: #fbbf24; margin-bottom: 4px;'>⭐ [FAVORITES SPEED-DIAL DECK] Pinned Stimuli Links</div>", unsafe_allow_html=True)
        fav_cols = st.columns(max(len(st.session_state.favorites), 1))
        for f_idx, fav_name in enumerate(st.session_state.favorites):
            with fav_cols[f_idx]:
                if st.button(f"🎵 {fav_name[:25]}...", use_container_width=True, key=f"fav_btn_{fav_name}"):
                    st.session_state.selected_track_override = fav_name
                    st.rerun()
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

    top_col1, top_col2 = st.columns([1, 1])
    
    with top_col1:
        st.markdown("<div style='font-family: monospace; font-size: 1.0rem; color: #94a3b8; margin-bottom: 5px;'>[AUDIO STIMULI BANK] SELECT ACTIVE TRACK</div>", unsafe_allow_html=True)
        
        default_index = 0
        if "selected_track_override" in st.session_state and st.session_state.selected_track_override in all_tracks_list:
            default_index = all_tracks_list.index(st.session_state.selected_track_override) + 1
            
        dropdown_options = ["-- Select Track from Bank --"] + all_tracks_list
        selected_track_name = st.selectbox("", options=dropdown_options, index=default_index, label_visibility="collapsed", key="master_bank_dropdown")
        
    with top_col2:
        st.markdown("<div style='font-family: monospace; font-size: 1.0rem; color: #94a3b8; margin-bottom: 5px;'>[PORT PORTAL] BATCH IMPORT NEW FILES TO BANK</div>", unsafe_allow_html=True)
        new_uploads = st.file_uploader("", type=["mp3", "wav"], accept_multiple_files=True, label_visibility="collapsed", key="uploader_portal")
        
        if new_uploads:
            triggered_rerun = False
            for upload in new_uploads:
                if upload.name not in st.session_state.session_tracks:
                    st.session_state.session_tracks[upload.name] = upload.read()
                    triggered_rerun = True
            if triggered_rerun:
                st.rerun()

    active_target = None
    base_name = ""
    
    if selected_track_name and selected_track_name != "-- Select Track from Bank --":
        is_fav = selected_track_name in st.session_state.favorites
        fav_label = "⭐ REMOVE FROM FAVOURITES" if is_fav else "☆ ADD TO FAVOURITES"
        if st.button(fav_label, key="toggle_fav_action"):
            if is_fav: st.session_state.favorites.remove(selected_track_name)
            else: st.session_state.favorites.append(selected_track_name)
            local_storage.setItem("favorites", st.session_state.favorites)
            st.session_state.selected_track_override = selected_track_name
            st.rerun()
            
        if selected_track_name in stored_files:
            active_target = os.path.join(LIBRARY_DIR, selected_track_name)
            base_name = selected_track_name.rsplit('.', 1)[0]
        else:
            class NamedBytesIO(io.BytesIO):
                def __init__(self, buffer, name):
                    super().__init__(buffer)
                    self.name = name
            active_target = NamedBytesIO(st.session_state.session_tracks[selected_track_name], selected_track_name)
            base_name = selected_track_name.rsplit('.', 1)[0]

    if active_target is not None:
        try:
            if isinstance(active_target, str):
                if active_target.lower().endswith('.mp3'):
                    total_duration = len(AudioSegment.from_file(active_target, format="mp3")) / 1000.0
                else:
                    data, fs = sf.read(active_target)
                    total_duration = len(data) / float(fs)
            else:
                active_target.seek(0)
                if active_target.name.lower().endswith('.mp3'):
                    total_duration = len(AudioSegment.from_file(active_target, format="mp3")) / 1000.0
                else:
                    data, fs = sf.read(active_target)
                    total_duration = len(data) / float(fs)
                active_target.seek(0)
        except:
            total_duration = 60.0
            if not isinstance(active_target, str): active_target.seek(0)

        # Dual Control Dial Block: Signal Gate + Pre-Roll Alignment Dial
        slider_col1, slider_col2 = st.columns(2)
        with slider_col1:
            st.markdown("<div style='font-family: monospace; font-size: 1.0rem; color: #38bdf8; margin-top: 10px; margin-bottom: -5px;'>[SIGNAL TRIMMING GATE] CUT START POSITION (SECONDS)</div>", unsafe_allow_html=True)
            trim_seconds = st.slider("", min_value=0.0, max_value=min(total_duration - 2.0, 30.0), value=0.0, step=0.5, label_visibility="collapsed", key="trim_slider")
        with slider_col2:
            st.markdown("<div style='font-family: monospace; font-size: 1.0rem; color: #fbbf24; margin-top: 10px; margin-bottom: -5px;'>[ALIGNMENT LATENCY DIAL] JUMP BACK PRE-ROLL (SECONDS)</div>", unsafe_allow_html=True)
            preroll_offset = st.slider("", min_value=0.0, max_value=5.0, value=2.0, step=0.1, label_visibility="collapsed", key="latency_dial")
        
        stimuli_manifest = [
            {"label": "Broadband", "low": None, "high": None, "type": "raw", "suffix": "Broadband", "order": 8, "comp": 3},
            {"label": "Low-Pass (≤1000 Hz)", "low": None, "high": 1000, "type": "low", "suffix": "LowPass_1kHz", "order": 8, "comp": 2},
            {"label": "High-Pass (>1000 Hz)", "low": 1000, "high": None, "type": "high", "suffix": "HighPass_1kHz", "order": 8, "comp": 2},
            {"label": "500Hz BPF", "low": 420, "high": 595, "type": "band", "suffix": "500Hz_BPF", "order": 8, "comp": 1},
            {"label": "1000Hz BPF", "low": 841, "high": 1189, "type": "band", "suffix": "1000Hz_BPF", "order": 8, "comp": 1},
            {"label": "2000Hz BPF", "low": 1682, "high": 2378, "type": "band", "suffix": "2000Hz_BPF", "order": 8, "comp": 1},
            {"label": "4000Hz BPF", "low": 3364, "high": 4757, "type": "band", "suffix": "4000Hz_BPF", "order": 8, "comp": 1}
        ]
        
        # --- VIEW MODE 1: LIVE PRESENTATION MODE DESK ---
        if "LIVE LINE-IN" in ui_mode:
            r1_c1, r1_c2, r1_c3 = st.columns(3)
            
            broadband_items = [i for i in stimuli_manifest if "BPF" not in i["label"]]
            for i, item in enumerate(broadband_items):
                col = [r1_c1, r1_c2, r1_c3][i]
                with col:
                    processed_buffer = process_audio_buffer(active_target, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                    if not isinstance(active_target, str): active_target.seek(0)
                    render_audiometer_channel(item["label"], processed_buffer, item["suffix"], preroll_offset, item["comp"])

            # --- DYNAMIC SIGNAL MONITOR CONSOLE ---
            selected_track_name = st.session_state.get("master_bank_dropdown", "-- Select Track from Bank --")
            display_name = selected_track_name if selected_track_name != "-- Select Track from Bank --" else "NO SIGNAL SELECTED"
            st.markdown(f"""
                <div style="background: #0f172a; border: 2px solid #38bdf8; border-radius: 12px; padding: 20px; margin: 15px 0; text-align: center; color: #38bdf8; font-family: 'Courier New', monospace; font-size: 1.3rem; font-weight: bold; box-shadow: 0 0 15px rgba(56,189,248,0.2);">
                    ACTIVE SIGNAL: {display_name}
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='audiogram-ruler'><span>500Hz</span><span>1kHz</span><span>2kHz</span><span>4kHz</span></div>", unsafe_allow_html=True)
            r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
            
            nbn_items = [i for i in stimuli_manifest if "BPF" in i["label"]]
            for i, item in enumerate(nbn_items):
                col = [r2_c1, r2_c2, r2_c3, r2_c4][i]
                with col:
                    processed_buffer = process_audio_buffer(active_target, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                    if not isinstance(active_target, str): active_target.seek(0)
                    render_audiometer_channel(item["label"], processed_buffer, item["suffix"], preroll_offset, item["comp"])

        # --- VIEW MODE 2: EXPORT & DOWNLOAD ARCHIVE SECTION ---
        else:
            st.markdown("<h4 style='margin: 0 0 10px 0; font-family: monospace; color: #f8fafc; font-size: 1.2rem;'>📦 DISK COMPILATION EXPEDITIONS</h4>", unsafe_allow_html=True)
            dl_col1, dl_col2 = st.columns([1, 2])
            
            with dl_col1:
                st.markdown("<div style='background-color: #1e293b; padding: 10px 15px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 1.0rem; color: #f8fafc; font-weight: bold;'>[BATCH UTILITY] PACK COMPLETE MATRICES</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.caption("Bundles all processed, gated, and RMS-normalized variants into one zip file.")
                    with st.spinner("Compiling compressed archive..."):
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                            for item in stimuli_manifest:
                                track_data = process_audio_buffer(active_target, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                                if not isinstance(active_target, str): active_target.seek(0)
                                zip_file.writestr(f"{base_name}_{item['suffix']}.wav", track_data.getvalue())
                        zip_buffer.seek(0)
                        st.download_button("📦 DOWNLOAD COMPLETE SET (.ZIP)", data=zip_buffer, file_name=f"{base_name}_VRA_Complete_Set.zip", mime="application/zip", use_container_width=True, type="primary")
            
            with dl_col2:
                st.markdown("<div style='background-color: #1e293b; padding: 10px 15px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 1.0rem; color: #f8fafc; font-weight: bold;'>[INDIVIDUAL SELECTION] EXTRACT SPECIFIC MANIFEST ARTIFACTS</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    for item in stimuli_manifest:
                        processed_buffer = process_audio_buffer(active_target, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                        if not isinstance(active_target, str): active_target.seek(0)
                        btn_col1, btn_col2 = st.columns([3, 1])
                        with btn_col1: st.markdown(f"<div style='font-family: monospace; font-size: 1.0rem; color: #e2e8f0; padding-top: 10px;'>→ {base_name}_{item['suffix']}.wav</div>", unsafe_allow_html=True)
                        with btn_col2: st.download_button("📥 SAVE WAV", data=processed_buffer, file_name=f"{base_name}_{item['suffix']}.wav", mime="audio/wav", use_container_width=True, key=f"archive_dl_{item['suffix']}")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
