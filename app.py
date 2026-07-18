import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile

# Set up a modern, polished page config
st.set_page_config(
    page_title="Neilio's VRA Toolkit", 
    page_icon="🎧", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Design Header
st.markdown("""
    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center; border-left: 5px solid #475569;">
        <h1 style="color: #1e293b; margin: 0; font-family: sans-serif; font-size: 2rem;">🎧 Neilio's VRA Audio Toolkit</h1>
        <p style="color: #64748b; font-size: 1rem; margin-top: 8px; margin-bottom: 0;">Transform standard tracks into calibrated, RMS-normalized clinical stimuli.</p>
    </div>
""", unsafe_allow_html=True)

# Advanced Filtering Engine
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
    """Calculates the Root Mean Square (average energy) of the audio data."""
    return np.sqrt(np.mean(data**2))

def rms_normalize(data, target_db=-20.0, peak_limit=0.95):
    """Normalizes to a precise target RMS decibel level with a safety peak limiter."""
    current_rms = calculate_rms(data)
    if current_rms == 0:
        return data
        
    target_linear = 10 ** (target_db / 20.0)
    gain = target_linear / current_rms
    normalized_data = data * gain
    
    # Peak Limiter Safety Step
    max_peak = np.max(np.abs(normalized_data))
    if max_peak > peak_limit:
        normalized_data = (normalized_data / max_peak) * peak_limit
        
    return normalized_data

def process_audio_buffer(uploaded_file, lowcut=None, highcut=None, filter_type='band'):
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
        sos = butter_filter_sos(lowcut, highcut, fs, filter_type=filter_type, order=8)
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]):
                audio_band = sosfilt(sos, data[:, channel])
                filtered_data[:, channel] = audio_band
        else:
            filtered_data = sosfilt(sos, data)

    # Standardize dynamic levels to a strict clinical target (-20 dBFS RMS)
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
with st.container():
    uploaded_file = st.file_uploader("📂 Select an audio track from your computer", type=["mp3", "wav"])

if uploaded_file is not None:
    st.markdown("---")
    base_name = uploaded_file.name.rsplit('.', 1)[0]
    
    # Configuration manifest for the ZIP package engine
    stimuli_manifest = [
        {"label": "Full-Range", "low": None, "high": None, "type": "raw", "suffix": "Full-Range"},
        {"label": "Low-Pass (≤1000 Hz)", "low": None, "high": 1000, "type": "low", "suffix": "LowPass_1kHz"},
        {"label": "High-Pass (>1000 Hz)", "low": 1000, "high": None, "type": "high", "suffix": "HighPass_1kHz"},
        {"label": "500Hz Narrowband", "low": 420, "high": 595, "type": "band", "suffix": "500Hz_NBN"},
        {"label": "1000Hz Narrowband", "low": 841, "high": 1189, "type": "band", "suffix": "1000Hz_NBN"},
        {"label": "2000Hz Narrowband", "low": 1682, "high": 2378, "type": "band", "suffix": "2000Hz_NBN"},
        {"label": "4000Hz Narrowband", "low": 3364, "high": 4757, "type": "band", "suffix": "4000Hz_NBN"}
    ]

    # --- TOP LEVEL: BULK DOWNLOAD ZIP BUTTON ---
    st.subheader("📦 Bulk Actions")
    with st.container(border=True):
        st.markdown("Compile all 7 calibrated configurations into a single compressed folder.")
        
        with st.spinner("Building ZIP package in memory..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for item in stimuli_manifest:
                    track_data = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"])
                    uploaded_file.seek(0)
                    filename = f"{base_name}_{item['suffix']}.wav"
                    zip_file.writestr(filename, track_data.getvalue())
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📦 Download All Tracks (.ZIP)",
                data=zip_buffer,
                file_name=f"{base_name}_VRA_Calibrated_Set.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎵 Individual Track Downloads")
    st.caption(f"Source track: {uploaded_file.name}")
    
    # 1. Full-Range Baseline Row
    with st.container(border=True):
        st.markdown("**Conditioning & Baseline Track**")
        processed_buffer = process_audio_buffer(uploaded_file, None, None, 'raw')
        uploaded_file.seek(0)
        st.download_button(
            label="📥 Download Full-Range (Calibrated Original)",
            data=processed_buffer,
            file_name=f"{base_name}_Full-Range.wav",
            mime="audio/wav",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Broad Spectrum Splits Row
    st.markdown("**Broad Spectrum Splits**")
    col_lp, col_hp = st.columns(2)
    with col_lp:
        with st.container(border=True):
            st.markdown("### Low-Pass (≤1000 Hz)")
            processed_buffer = process_audio_buffer(uploaded_file, None, 1000, 'low')
            uploaded_file.seek(0)
            st.download_button(
                label="📥 Download Low-Pass",
                data=processed_buffer,
                file_name=f"{base_name}_LowPass_1kHz.wav",
                mime="audio/wav",
                use_container_width=True
            )
    with col_hp:
        with st.container(border=True):
            st.markdown("### High-Pass (>1000 Hz)")
            processed_buffer = process_audio_buffer(uploaded_file, 1000, None, 'high')
            uploaded_file.seek(0)
            st.download_button(
                label="📥 Download High-Pass",
                data=processed_buffer,
                file_name=f"{base_name}_HighPass_1kHz.wav",
                mime="audio/wav",
                use_container_width=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Individual Narrow Bands Grid
    st.markdown("**Narrow Band Frequencies**")
    col1, col2 = st.columns(2)
    
    nbn_items = [item for item in stimuli_manifest if "NBN" in item["suffix"]]
    
    for idx, item in enumerate(nbn_items):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            with st.container(border=True):
                band_title = item["suffix"].split('_')[0]
                st.markdown(f"### {band_title}")
                st.caption(item["label"])
                
                processed_buffer = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"])
                uploaded_file.seek(0)
                
                st.download_button(
                    label=f"📥 Download {band_title}",
                    data=processed_buffer,
                    file_name=f"{base_name}_{item['suffix']}.wav",
                    mime="audio/wav",
                    key=f"btn_{band_title}",
                    use_container_width=True
                )
                
    st.balloons()
