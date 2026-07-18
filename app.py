import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile
import os

# Set wide layout to establish a comprehensive dual-channel audiometer faceplate
st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ensure local persistence folder exists for storing library files
LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR):
    os.makedirs(LIBRARY_DIR)

# Initialize system memory cache for tracking current loaded session tracks
if "session_tracks" not in st.session_state:
    st.session_state.session_tracks = {}

# Initialize system memory cache for favorite tracks tracking
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# Custom Audiometer Structural Frame & LED Header CSS Injection
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
            height: 32px !important;
            margin-bottom: 8px !important;
            margin-top: 2px !important;
            width: 100%;
        }
    </style>
    
    <!-- Faceplate Main Header -->
    <div style="background: linear-gradient(180deg, #334155 0%, #1e293b 100%); padding: 15px 25px; border-radius: 8px 8px 0px 0px; border: 2px solid #475569; border-bottom: none; display: flex; align-items: center; justify-content: space-between; box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="display: flex; align-items: center; gap: 20px;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="38" height="38">
                <path d="M70,30 C60,25 45,35 40,45 C38,42 32,30 20,25 C25,38 35,45 38,48 C30,52 15,55 5,52 C18,58 32,58 39,53 C41,60 45,75 42,90 C48,75 50,60 48,52 C55,48 85,32 95,30 C85,32 75,32 70,30 Z" fill="#94a3b8"/>
            </svg>
            <div style="text-align: left;">
                <h1 style="color: #f8fafc; margin: 0; font-family: monospace; font-size: 1.4rem; letter-spacing: 1px; font-weight: 700;">NEILIO'S VRA CLINICAL STIMULI GENERATOR</h1>
                <div style="color: #94a3b8; font-size: 0.75rem; font-family: monospace; margin-top: 2px;">MODEL VRA-11 // RMS-CALIBRATED OUTPUT MATRIX</div>
            </div>
        </div>
        <!-- Simulated Hardware Status LED Matrix -->
        <div style="display: flex; gap: 12px; align-items: center; background: #0f172a; padding: 6px 12px; border-radius: 4px; border: 1px solid #334155;">
            <div style="display: flex; align-items: center; gap: 5px; font-family: monospace; font-size: 0.7rem; color: #64748b;">
                <div style="width: 8px; height: 8px; background-color: #22c55e; border-radius: 50%; box-shadow: 0 0 8px #22c55e;"></div> SYS_READY
            </div>
            <div style="width: 1px; height: 12px; background: #334155;"></div>
            <div style="display: flex; align-items: center; gap: 5px; font-family: monospace; font-size: 0.7rem; color: #64748b;">
                <div style="width: 8px; height: 8px; background-color: #38bdf8; border-radius: 50%; box-shadow: 0 0 8px #38bdf8;"></div> RMS_FIXED (-20dBFS)
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Advanced Filtering Engine supporting standard and ultra-steep FRESH slopes
def butter_filter_sos(cutoff_low, cutoff_high, fs, filter_type='band', order=8):
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

# Clinical RMS Calibration Engine
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

def process_audio_buffer(file_source, lowcut=None, highcut=None, filter_type='band', order=8, trim_seconds=0.0):
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f:
            file_bytes = f.read()
        is_mp3 = file_source.lower().endswith('.mp3')
    else:
        file_bytes = file_source.read()
        is_mp3 = file_source.name.lower().endswith('.mp3')
        file_source.seek(0)
    
    if is_mp3:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
        audio = audio.set_frame_rate(44100).set_channels(1)
        fs = audio.frame_rate
        data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        data = data / (2**15)
    else:
        data, fs = sf.read(io.BytesIO(file_bytes))
        
    # --- PHYSICAL SIGNAL GATE (TRIM INTRO) ---
    if trim_seconds > 0:
        start_sample = int(trim_seconds * fs)
        if start_sample < len(data):
            if len(data.shape) > 1:
                data = data[start_sample:, :]
            else:
                data = data[start_sample:]
        
    if filter_type == 'raw':
        filtered_data = data
    else:
        sos = butter_filter_sos(lowcut, highcut, fs, filter_type=filter_type, order=order)
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]):
                audio_band = sosfilt(sos, data[:, channel])
                filtered_data[:, channel] = audio_band
        else:
            filtered_data = sosfilt(sos, data)

    if len(filtered_data.shape) > 1:
        for channel in range(filtered_data.shape[1]):
            filtered_data[:, channel] = rms_normalize(filtered_data[:, channel], target_db=-20.0)
    else:
        filtered_data = rms_normalize(filtered_data, target_db=-20.0)
        
    virtual_file = io.BytesIO()
    sf.write(virtual_file, filtered_data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

# Helper to inject HTML5 audio cards with dynamic playhead alignment settings
def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset):
    import base64
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    audio_src = f"data:audio/wav;base64,{audio_base64}"
    
    html_code = f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        <div style="font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold; margin-bottom: 6px; letter-spacing: 0.5px;">{label}</div>
        
        <audio id="audio_{element_key}" src="{audio_src}" controls style="width:100%;"></audio>
        
        <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button onclick="document.getElementById('audio_{element_key}').play()" style="flex: 1; background-color: #10b981; color: white; border: none; padding: 6px; border-radius: 4px; font-family: monospace; font-size: 0.75rem; cursor: pointer; font-weight: bold;">▶ PLAY</button>
            <button onclick="document.getElementById('audio_{element_key}').pause()" style="flex: 1; background-color: #ef4444; color: white; border: none; padding: 6px; border-radius: 4px; font-family: monospace; font-size: 0.75rem; cursor: pointer; font-weight: bold;">⏸ PAUSE</button>
        </div>
        
        <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button onclick="var clickTime = document.getElementById('audio_{element_key}').currentTime; window.parent.sharedVraLoopPoint = Math.max(0, clickTime - {preroll_offset}); this.innerHTML='⚙️ SET -' + {preroll_offset} + 's'; setTimeout(()=>{{this.innerHTML='🔴 MARK POINT'}}, 1500);" style="flex: 1; background-color: #f59e0b; color: #0f172a; border: none; padding: 6px; border-radius: 4px; font-family: monospace; font-size: 0.75rem; cursor: pointer; font-weight: bold;">🔴 MARK POINT</button>
            <button onclick="if(window.parent.sharedVraLoopPoint !== undefined) {{ document.getElementById('audio_{element_key}').currentTime = window.parent.sharedVraLoopPoint; document.getElementById('audio_{element_key}').play(); }}" style="flex: 1; background-color: #38bdf8; color: #0f172a; border: none; padding: 6px; border-radius: 4px; font-family: monospace; font-size: 0.75rem; cursor: pointer; font-weight: bold;">↩️ JUMP BACK</button>
        </div>
    </div>
    """
    st.components.v1.html(html_code, height=165)

# Main structural container mimicking the physical control board chassis
with st.container(border=True):
    
    # Functional System Mode Toggle Strip
    st.markdown("<div style='font-family: monospace; font-size: 0.8rem; color: #64748b; margin-bottom: 2px;'>[CONSOLE FUNCTION CONFIGURATION]</div>", unsafe_allow_html=True)
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
        st.markdown("<div style='font-family: monospace; font-size: 0.75rem; color: #fbbf24; margin-bottom: 4px;'>⭐ [FAVORITES SPEED-DIAL DECK] Pinned Stimuli Links</div>", unsafe_allow_html=True)
        fav_cols = st.columns(max(len(st.session_state.favorites), 1))
        for f_idx, fav_name in enumerate(st.session_state.favorites):
            with fav_cols[f_idx]:
                if st.button(f"🎵 {fav_name[:25]}...", use_container_width=True, key=f"fav_btn_{fav_name}"):
                    st.session_state.selected_track_override = fav_name
                    st.rerun()
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

    top_col1, top_col2 = st.columns([1, 1])
    
    with top_col1:
        st.markdown("<div style='font-family: monospace; font-size: 0.8rem; color: #94a3b8; margin-bottom: 5px;'>[AUDIO STIMULI BANK] SELECT ACTIVE TRACK</div>", unsafe_allow_html=True)
        
        # Handle cross-state override changes safely from favorites selection events
        default_index = 0
        if "selected_track_override" in st.session_state and st.session_state.selected_track_override in all_tracks_list:
            default_index = all_tracks_list.index(st.session_state.selected_track_override) + 1
            del st.session_state.selected_track_override
            
        dropdown_options = ["-- Select Track from Bank --"] + all_tracks_list
        selected_track_name = st.selectbox("", options=dropdown_options, index=default_index, label_visibility="collapsed", key="master_bank_dropdown")
        
    with top_col2:
        st.markdown("<div style='font-family: monospace; font-size: 0.8rem; color: #94a3b8; margin-bottom: 5px;'>[PORT PORTAL] BATCH IMPORT NEW FILES TO BANK</div>", unsafe_allow_html=True)
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
        # Pin/Unpin Toggle Button Directly Under Selector Layout Frame
        is_fav = selected_track_name in st.session_state.favorites
        fav_label = "⭐ REMOVE FROM FAVOURITES" if is_fav else "☆ ADD TO FAVOURITES"
        if st.button(fav_label, key="toggle_fav_action"):
            if is_fav:
                st.session_state.favorites.remove(selected_track_name)
            else:
                st.session_state.favorites.append(selected_track_name)
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
                    temp_audio = AudioSegment.from_file(active_target, format="mp3")
                    total_duration = len(temp_audio) / 1000.0
                else:
                    temp_data, temp_fs = sf.read(active_target)
                    total_duration = len(temp_data) / float(temp_fs)
            else:
                active_target.seek(0)
                file_bytes = active_target.read()
                if active_target.name.lower().endswith('.mp3'):
                    temp_audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
                    total_duration = len(temp_audio) / 1000.0
                else:
                    temp_data, temp_fs = sf.read(io.BytesIO(file_bytes))
                    total_duration = len(temp_data) / float(temp_fs)
                active_target.seek(0)
        except:
            total_duration = 60.0
            if not isinstance(active_target, str): active_target.seek(0)

        # Dual Control Dial Block: Signal Gate + Pre-Roll Alignment Dial
        slider_col1, slider_col2 = st.columns(2)
        with slider_col1:
            st.markdown("<div style='font-family: monospace; font-size: 0.8rem; color: #38bdf8; margin-top: 10px; margin-bottom: -5px;'>[SIGNAL TRIMMING GATE] CUT START POSITION (SECONDS)</div>", unsafe_allow_html=True)
            trim_seconds = st.slider("", min_value=0.0, max_value=min(total_duration - 2.0, 30.0), value=0.0, step=0.5, label_visibility="collapsed", key="trim_slider")
        with slider_col2:
            st.markdown("<div style='font-family: monospace; font-size: 0.8rem; color: #fbbf24; margin-top: 10px; margin-bottom: -5px;'>[ALIGNMENT LATENCY DIAL] JUMP BACK PRE-ROLL (SECONDS)</div>", unsafe_allow_html=True)
            preroll_offset = st.slider("", min_value=0.0, max_value=5.0, value=2.0, step=0.1, label_visibility="collapsed", key="latency_dial")
        
        stimuli_manifest = [
            {"label": "Full-Range", "low": None, "high": None, "type": "raw", "suffix": "Full-Range", "order": 8},
            {"label": "Low-Pass (≤1000 Hz)", "low": None, "high": 1000, "type": "low", "suffix": "LowPass_1kHz", "order": 8},
            {"label": "High-Pass (>1000 Hz)", "low": 1000, "high": None, "type": "high", "suffix": "HighPass_1kHz", "order": 8},
            {"label": "500Hz Original NBN", "low": 420, "high": 595, "type": "band", "suffix": "500Hz_NBN", "order": 8},
            {"label": "1000Hz Original NBN", "low": 841, "high": 1189, "type": "band", "suffix": "1000Hz_NBN", "order": 8},
            {"label": "2000Hz Original NBN", "low": 1682, "high": 2378, "type": "band", "suffix": "2000Hz_NBN", "order": 8},
            {"label": "4000Hz Original NBN", "low": 3364, "high": 4757, "type": "band", "suffix": "4000Hz_NBN", "order": 8},
            {"label": "500Hz FRESH", "low": 450, "high": 550, "type": "band", "suffix": "500Hz_FRESH", "order": 20},
            {"label": "1000Hz FRESH", "low": 900, "high": 1100, "type": "band", "suffix": "1000Hz_FRESH", "order": 20},
            {"label": "2000Hz FRESH", "low": 1800, "high": 2200, "type": "band", "suffix": "2000Hz_FRESH", "order": 20},
            {"label": "4000Hz FRESH", "low": 3600, "high": 4400, "type": "band", "suffix": "4000Hz_FRESH", "order": 20}
        ]
        
        # --- VIEW MODE 1: LIVE PRESENTATION MODE DESK ---
        if "LIVE LINE-IN" in ui_mode:
            left_col, center_col, right_col = st.columns(3)
            
            with left_col:
                st.markdown("<div style='background-color: #1e293b; padding: 6px 10px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold;'>[PANEL A] MASTER ROUTING</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown("<div style='font-family: monospace; font-size: 0.7rem; color: #e2e8f0; margin-bottom: 2px;'>🎚️ AUDIOMETER VU CALIBRATION (1kHz Tone)</div>", unsafe_allow_html=True)
                    fs_cal = 44100
                    t_cal = np.linspace(0, 5.0, int(fs_cal * 5.0), endpoint=False)
                    tone_cal = np.sin(2 * np.pi * 1000 * t_cal) * (10 ** (-20.0 / 20.0) * np.sqrt(2))
                    cal_buffer = io.BytesIO()
                    sf.write(cal_buffer, tone_cal, fs_cal, format='WAV')
                    cal_buffer.seek(0)
                    st.audio(cal_buffer, format="audio/wav")
                    
                    st.markdown("<hr style='margin: 8px 0; border-color: #334155;' />", unsafe_allow_html=True)

                    processed_buffer = process_audio_buffer(active_target, None, None, 'raw', 8, trim_seconds)
                    if not isinstance(active_target, str): active_target.seek(0)
                    render_audiometer_channel("🎛️ FULL-RANGE FLAT", processed_buffer, "full", preroll_offset)
                    
                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    processed_buffer = process_audio_buffer(active_target, None, 1000, 'low', 8, trim_seconds)
                    if not isinstance(active_target, str): active_target.seek(0)
                    render_audiometer_channel("🎚️ LOW-PASS (≤1000 Hz)", processed_buffer, "lp", preroll_offset)
                    
                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    processed_buffer = process_audio_buffer(active_target, 1000, None, 'high', 8, trim_seconds)
                    if not isinstance(active_target, str): active_target.seek(0)
                    render_audiometer_channel("🎚️ HIGH-PASS (>1000 Hz)", processed_buffer, "hp", preroll_offset)

            with center_col:
                st.markdown("<div style='background-color: #1e293b; padding: 6px 10px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold;'>[CHANNEL 1] STANDARD NBN BANK</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    nbn_items = [item for item in stimuli_manifest if "NBN" in item["suffix"]]
                    for idx, item in enumerate(nbn_items):
                        if idx > 0:
                            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                        freq_lbl = item["suffix"].split('_')[0]
                        processed_buffer = process_audio_buffer(active_target, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                        if not isinstance(active_target, str): active_target.seek(0)
                        render_audiometer_channel(f"🔊 FREQ {freq_lbl.upper()} // NBN", processed_buffer, f"nbn_{freq_lbl}", preroll_offset)

            with right_col:
                # We simply don't render the FRESH bank columns if we are in LIVE mode.
                st.markdown("<div style='background-color: #1e293b; padding: 6px 10px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold;'>[CHANNEL 2] FRESH STEEP BANK // HIDDEN</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown("<div style='font-family: monospace; font-size: 0.75rem; color: #64748b;'>FRESH Stimuli Bank restricted to Bulk Export mode.</div>", unsafe_allow_html=True)

        # --- VIEW MODE 2: EXPORT & DOWNLOAD ARCHIVE SECTION ---
        else:
            st.markdown("<h4 style='margin: 0 0 10px 0; font-family: monospace; color: #f8fafc;'>📦 DISK COMPILATION EXPEDITIONS</h4>", unsafe_allow_html=True)
            
            dl_col1, dl_col2 = st.columns([1, 2])
            
            with dl_col1:
                st.markdown("<div style='background-color: #1e293b; padding: 6px 10px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold;'>[BATCH UTILITY] PACK COMPLETE MATRICES</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.caption("Bundles all 11 processed, gated, and RMS-normalized variants into one zip file.")
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
                st.markdown("<div style='background-color: #1e293b; padding: 6px 10px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold;'>[INDIVIDUAL SELECTION] EXTRACT SPECIFIC MANIFEST ARTIFACTS</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    for item in stimuli_manifest:
                        processed_buffer = process_audio_buffer(active_target, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                        if not isinstance(active_target, str): active_target.seek(0)
                        
                        btn_col1, btn_col2 = st.columns([3, 1])
                        with btn_col1:
                            st.markdown(f"<div style='font-family: monospace; font-size: 0.85rem; color: #e2e8f0; padding-top: 6px;'>→ {base_name}_{item['suffix']}.wav</div>", unsafe_allow_html=True)
                        with btn_col2:
                            st.download_button("📥 SAVE WAV", data=processed_buffer, file_name=f"{base_name}_{item['suffix']}.wav", mime="audio/wav", use_container_width=True, key=f"archive_dl_{item['suffix']}")

# Tiny layout buffer line at the bottom
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
