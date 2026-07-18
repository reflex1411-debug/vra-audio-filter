import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile

# Set up a wide layout to maximize screen real estate and reduce scrolling
st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-Compact Design Header with Embedded Hummingbird Logo
st.markdown("""
    <div style="background-color: #f0f2f6; padding: 12px 25px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #475569; display: flex; align-items: center; justify-content: center; gap: 20px;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="40" height="40">
            <path d="M70,30 C60,25 45,35 40,45 C38,42 32,30 20,25 C25,38 35,45 38,48 C30,52 15,55 5,52 C18,58 32,58 39,53 C41,60 45,75 42,90 C48,75 50,60 48,52 C55,48 85,32 95,30 C85,32 75,32 70,30 Z" fill="#475569"/>
        </svg>
        <div style="text-align: left;">
            <h1 style="color: #1e293b; margin: 0; font-family: sans-serif; font-size: 1.5rem; display: inline-block;">Neilio's VRA Audio Toolkit</h1>
            <span style="color: #64748b; font-size: 0.9rem; margin-left: 15px;">Calibrated, RMS-normalized clinical stimuli.</span>
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

def process_audio_buffer(uploaded_file, lowcut=None, highcut=None, filter_type='band', order=8):
    file_bytes = uploaded_file.read()
    
    if uploaded_file.name.lower().endswith('.mp3'):
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
        audio = audio.set_frame_rate(44100).set_channels(1)
        fs = audio.frame_rate
        data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        data = data / (2**15)
    else:
        data, fs = sf.read(io.BytesIO(file_bytes))
        
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

# Clean UI Box for File Uploading
uploaded_file = st.file_uploader("📂 Select an audio track from your computer", type=["mp3", "wav"])

if uploaded_file is not None:
    base_name = uploaded_file.name.rsplit('.', 1)[0]
    
    # Master manifest detailing all 11 possible outputs
    stimuli_manifest = [
        {"label": "Full-Range", "low": None, "high": None, "type": "raw", "suffix": "Full-Range", "order": 8},
        {"label": "Low-Pass (≤1000 Hz)", "low": None, "high": 1000, "type": "low", "suffix": "LowPass_1kHz", "order": 8},
        {"label": "High-Pass (>1000 Hz)", "low": 1000, "high": None, "type": "high", "suffix": "HighPass_1kHz", "order": 8},
        # Original NBN Profile (Order 8, 1/3 Octave Bounds)
        {"label": "500Hz Original NBN", "low": 420, "high": 595, "type": "band", "suffix": "500Hz_NBN", "order": 8},
        {"label": "1000Hz Original NBN", "low": 841, "high": 1189, "type": "band", "suffix": "1000Hz_NBN", "order": 8},
        {"label": "2000Hz Original NBN", "low": 1682, "high": 2378, "type": "band", "suffix": "2000Hz_NBN", "order": 8},
        {"label": "4000Hz Original NBN", "low": 3364, "high": 4757, "type": "band", "suffix": "4000Hz_NBN", "order": 8},
        # FRESH Profile (Order 20, Narrower Bounds)
        {"label": "500Hz FRESH", "low": 450, "high": 550, "type": "band", "suffix": "500Hz_FRESH", "order": 20},
        {"label": "1000Hz FRESH", "low": 900, "high": 1100, "type": "band", "suffix": "1000Hz_FRESH", "order": 20},
        {"label": "2000Hz FRESH", "low": 1800, "high": 2200, "type": "band", "suffix": "2000Hz_FRESH", "order": 20},
        {"label": "4000Hz FRESH", "low": 3600, "high": 4400, "type": "band", "suffix": "4000Hz_FRESH", "order": 20}
    ]

    # Thin, highly compact audio processing status strip with mini-soundwave
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 6px; padding: 6px 15px; margin-bottom: 15px; display: flex; align-items: center; justify-content: space-between; border: 1px solid #334155;">
            <span style="color: #38bdf8; font-weight: 600; font-size: 0.85rem; font-family: sans-serif;">✨ Audio Variants Generated Successfully</span>
            <div style="display: flex; align-items: flex-end; height: 16px; gap: 2px;">
                <div style="width: 2px; height: 6px; background: #38bdf8; animation: pulse 0.5s infinite alternate;"></div>
                <div style="width: 2px; height: 14px; background: #38bdf8; animation: pulse 0.3s infinite alternate 0.1s;"></div>
                <div style="width: 2px; height: 10px; background: #38bdf8; animation: pulse 0.4s infinite alternate 0.2s;"></div>
                <div style="width: 2px; height: 16px; background: #38bdf8; animation: pulse 0.2s infinite alternate 0.3s;"></div>
            </div>
        </div>
        <style>@keyframes pulse { 0% { height: 4px; } 100% { height: 16px; } }</style>
    """, unsafe_allow_html=True)
    
    # 3-Column Master Grid Layout
    left_col, center_col, right_col = st.columns(3)
    
    # --- COLUMN 1: BASELINE & BROAD FILTERS ---
    with left_col:
        st.markdown("<h4 style='margin: 0 0 5px 0; font-size: 1.05rem;'>📋 Base & Broad Splits</h4>", unsafe_allow_html=True)
        
        with st.container(border=True):
            processed_buffer = process_audio_buffer(uploaded_file, None, None, 'raw', 8)
            uploaded_file.seek(0)
            st.download_button("📥 Full-Range Original", data=processed_buffer, file_name=f"{base_name}_Full-Range.wav", mime="audio/wav", use_container_width=True)
            
            processed_buffer = process_audio_buffer(uploaded_file, None, 1000, 'low', 8)
            uploaded_file.seek(0)
            st.download_button("📥 Low-Pass (≤1kHz)", data=processed_buffer, file_name=f"{base_name}_LowPass_1kHz.wav", mime="audio/wav", use_container_width=True)
            
            processed_buffer = process_audio_buffer(uploaded_file, 1000, None, 'high', 8)
            uploaded_file.seek(0)
            st.download_button("📥 High-Pass (>1kHz)", data=processed_buffer, file_name=f"{base_name}_HighPass_1kHz.wav", mime="audio/wav", use_container_width=True)

        # Bulk Actions integrated inside the first column to prevent bottom-scrolling
        st.markdown("<h4 style='margin: 10px 0 5px 0; font-size: 1.05rem;'>📦 Batch Download</h4>", unsafe_allow_html=True)
        with st.container(border=True):
            with st.spinner("Zipping..."):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for item in stimuli_manifest:
                        track_data = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"], item["order"])
                        uploaded_file.seek(0)
                        zip_file.writestr(f"{base_name}_{item['suffix']}.wav", track_data.getvalue())
                zip_buffer.seek(0)
                
                st.download_button("📦 All 11 Tracks (.ZIP)", data=zip_buffer, file_name=f"{base_name}_VRA_Complete_Set.zip", mime="application/zip", use_container_width=True, type="primary")

    # --- COLUMN 2: ORIGINAL NARROWBAND (NBN) ---
    with center_col:
        st.markdown("<h4 style='margin: 0 0 5px 0; font-size: 1.05rem;'>📊 Original Narrowband</h4>", unsafe_allow_html=True)
        with st.container(border=True):
            nbn_items = [item for item in stimuli_manifest if "NBN" in item["suffix"]]
            for item in nbn_items:
                freq_lbl = item["suffix"].split('_')[0]
                processed_buffer = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"], item["order"])
                uploaded_file.seek(0)
                st.download_button(f"📥 {freq_lbl} NBN Track", data=processed_buffer, file_name=f"{base_name}_{item['suffix']}.wav", mime="audio/wav", use_container_width=True, key=f"nbn_{freq_lbl}")

    # --- COLUMN 3: FRESH-STYLE FILTERS ---
    with right_col:
        st.markdown("<h4 style='margin: 0 0 5px 0; font-size: 1.05rem;'>⚡ FRESH-Style (Steep)</h4>", unsafe_allow_html=True)
        with st.container(border=True):
            fresh_items = [item for item in stimuli_manifest if "FRESH" in item["suffix"]]
            for item in fresh_items:
                freq_lbl = item["suffix"].split('_')[0]
                processed_buffer = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"], item["order"])
                uploaded_file.seek(0)
                st.download_button(f"📥 {freq_lbl} FRESH Track", data=processed_buffer, file_name=f"{base_name}_{item['suffix']}.wav", mime="audio/wav", use_container_width=True, key=f"fresh_{freq_lbl}")
