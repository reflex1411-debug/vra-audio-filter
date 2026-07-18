import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile

# Set wide layout to establish a comprehensive dual-channel audiometer faceplate
st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

def process_audio_buffer(uploaded_file, lowcut=None, highcut=None, filter_type='band', order=8, trim_seconds=0.0):
    file_bytes = uploaded_file.read()
    
    if uploaded_file.name.lower().endswith('.mp3'):
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

# Main structural container mimicking the physical control board chassis
with st.container(border=True):
    st.markdown("<div style='font-family: monospace; font-size: 0.8rem; color: #94a3b8; margin-bottom: 5px;'>[INPUT ROUTING] SELECT AUDIO SOURCE</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["mp3", "wav"], label_visibility="collapsed")

    if uploaded_file is not None:
        base_name = uploaded_file.name.rsplit('.', 1)[0]
        
        # Determine track length to scale the slider limits dynamically
        try:
            temp_bytes = uploaded_file.read()
            if uploaded_file.name.lower().endswith('.mp3'):
                temp_audio = AudioSegment.from_file(io.BytesIO(temp_bytes), format="mp3")
                total_duration = len(temp_audio) / 1000.0
            else:
                temp_data, temp_fs = sf.read(io.BytesIO(temp_bytes))
                total_duration = len(temp_data) / float(temp_fs)
            uploaded_file.seek(0)
        except:
            total_duration = 60.0 # Standard fallback
            uploaded_file.seek(0)

        # Hardware Attenuation Style Gate Slider
        st.markdown("<div style='font-family: monospace; font-size: 0.8rem; color: #38bdf8; margin-top: 10px; margin-bottom: -5px;'>[SIGNAL TRIMMING GATE] CUT START POSITION (SECONDS)</div>", unsafe_allow_html=True)
        trim_seconds = st.slider("", min_value=0.0, max_value=min(total_duration - 2.0, 30.0), value=0.0, step=0.5, label_visibility="collapsed")
        
        # 11-variant digital manifest mapping
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

        # Thin electronic VU/Soundwave signal indicator strip
        st.markdown("""
            <div style="background: #020617; border-radius: 4px; padding: 6px 12px; margin: 10px 0px 20px 0px; display: flex; align-items: center; justify-content: space-between; border: 1px solid #1e293b;">
                <span style="color: #22c55e; font-weight: bold; font-size: 0.75rem; font-family: monospace; letter-spacing: 0.5px;">✓ TRACK ANALYSED // CH1 CALIBRATION LOCKED</span>
                <div style="display: flex; align-items: flex-end; height: 14px; gap: 2px;">
                    <div style="width: 3px; height: 4px; background: #22c55e; animation: pulse 0.4s infinite alternate;"></div>
                    <div style="width: 3px; height: 12px; background: #22c55e; animation: pulse 0.2s infinite alternate 0.1s;"></div>
                    <div style="width: 3px; height: 8px; background: #22c55e; animation: pulse 0.3s infinite alternate 0.2s;"></div>
                    <div style="width: 3px; height: 14px; background: #22c55e; animation: pulse 0.1s infinite alternate 0.3s;"></div>
                </div>
            </div>
            <style>@keyframes pulse { 0% { height: 3px; } 100% { height: 14px; } }</style>
        """, unsafe_allow_html=True)
        
        # 3-Column Instrument Cluster Grid
        left_col, center_col, right_col = st.columns(3)
        
        # --- COLUMN 1: INTERFACE COMMAND & BROAD ATTENUATION ---
        with left_col:
            st.markdown("<div style='background-color: #1e293b; padding: 6px 10px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold;'>[PANEL A] MASTER ROUTING</div>", unsafe_allow_html=True)
            with st.container(border=True):
                # Hardcoded 1kHz Continuous Calibration Tone for Line-In VU Matching
                st.markdown("<div style='font-family: monospace; font-size: 0.7rem; color: #e2e8f0; margin-bottom: 2px;'>🎚️ AUDIOMETER VU CALIBRATION (1kHz Tone)</div>", unsafe_allow_html=True)
                fs_cal = 44100
                t_cal = np.linspace(0, 5.0, int(fs_cal * 5.0), endpoint=False)
                tone_cal = np.sin(2 * np.pi * 1000 * t_cal) * (10 ** (-20.0 / 20.0) * np.sqrt(2))
                cal_buffer = io.BytesIO()
                sf.write(cal_buffer, tone_cal, fs_cal, format='WAV')
                cal_buffer.seek(0)
                st.audio(cal_buffer, format="audio/wav")
                
                st.markdown("<hr style='margin: 8px 0; border-color: #334155;' />", unsafe_allow_html=True)

                # Full Range
                processed_buffer = process_audio_buffer(uploaded_file, None, None, 'raw', 8, trim_seconds)
                uploaded_file.seek(0)
                st.audio(processed_buffer, format="audio/wav")
                st.download_button("🎛️ FULL-RANGE FLAT", data=processed_buffer, file_name=f"{base_name}_Full-Range.wav", mime="audio/wav", use_container_width=True)
                
                # Low Pass
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                processed_buffer = process_audio_buffer(uploaded_file, None, 1000, 'low', 8, trim_seconds)
                uploaded_file.seek(0)
                st.audio(processed_buffer, format="audio/wav")
                st.download_button("🎚️ LOW-PASS (≤1000 Hz)", data=processed_buffer, file_name=f"{base_name}_LowPass_1kHz.wav", mime="audio/wav", use_container_width=True)
                
                # High Pass
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                processed_buffer = process_audio_buffer(uploaded_file, 1000, None, 'high', 8, trim_seconds)
                uploaded_file.seek(0)
                st.audio(processed_buffer, format="audio/wav")
                st.download_button("🎚️ HIGH-PASS (>1000 Hz)", data=processed_buffer, file_name=f"{base_name}_HighPass_1kHz.wav", mime="audio/wav", use_container_width=True)

            st.markdown("<div style='background-color: #1e293b; padding: 6px 10px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold; margin-top: 12px;'>[PANEL B] MASTER BATCH</div>", unsafe_allow_html=True)
            with st.container(border=True):
                with st.spinner("Compiling..."):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                        for item in stimuli_manifest:
                            track_data = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                            uploaded_file.seek(0)
                            zip_file.writestr(f"{base_name}_{item['suffix']}.wav", track_data.getvalue())
                    zip_buffer.seek(0)
                    
                    st.download_button("📦 EXPORT COMPLETE SET (.ZIP)", data=zip_buffer, file_name=f"{base_name}_VRA_Complete_Set.zip", mime="application/zip", use_container_width=True, type="primary")

        # --- COLUMN 2: CHANNEL 1 — STANDARD FILTER BANK (NBN) ---
        with center_col:
            st.markdown("<div style='background-color: #1e293b; padding: 6px 10px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold;'>[CHANNEL 1] STANDARD NBN BANK</div>", unsafe_allow_html=True)
            with st.container(border=True):
                nbn_items = [item for item in stimuli_manifest if "NBN" in item["suffix"]]
                for idx, item in enumerate(nbn_items):
                    if idx > 0:
                        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    freq_lbl = item["suffix"].split('_')[0]
                    processed_buffer = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                    uploaded_file.seek(0)
                    st.audio(processed_buffer, format="audio/wav")
                    st.download_button(f"🔊 FREQ {freq_lbl.upper()} // NBN", data=processed_buffer, file_name=f"{base_name}_{item['suffix']}.wav", mime="audio/wav", use_container_width=True, key=f"nbn_{freq_lbl}")

        # --- COLUMN 3: CHANNEL 2 — HIGH-SPECIFICITY BANK (FRESH) ---
        with right_col:
            st.markdown("<div style='background-color: #1e293b; padding: 6px 10px; border-radius: 4px 4px 0 0; border: 1px solid #334155; font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold;'>[CHANNEL 2] FRESH STEEP BANK</div>", unsafe_allow_html=True)
            with st.container(border=True):
                fresh_items = [item for item in stimuli_manifest if "FRESH" in item["suffix"]]
                for idx, item in enumerate(fresh_items):
                    if idx > 0:
                        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    freq_lbl = item["suffix"].split('_')[0]
                    processed_buffer = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                    uploaded_file.seek(0)
                    st.audio(processed_buffer, format="audio/wav")
                    st.download_button(f"⚡ FREQ {freq_lbl.upper()} // FRESH", data=processed_buffer, file_name=f"{base_name}_{item['suffix']}.wav", mime="audio/wav", use_container_width=True, key=f"fresh_{freq_lbl}")

# Tiny layout buffer line at the bottom
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
