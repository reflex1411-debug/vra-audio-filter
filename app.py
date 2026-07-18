import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io

# Set up a modern, polished page config
st.set_page_config(
    page_title="VRA Audio Toolkit", 
    page_icon="🎧", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Design Header
st.markdown("""
    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center; border-left: 5px solid #475569;">
        <h1 style="color: #1e293b; margin: 0; font-family: sans-serif; font-size: 2rem;">🎧 Paediatric VRA Audio Toolkit</h1>
        <p style="color: #64748b; font-size: 1rem; margin-top: 8px; margin-bottom: 0;">Transform standard tracks into calibrated, dynamically flattened clinical stimuli.</p>
    </div>
""", unsafe_allow_html=True)

# Core Filtering & Compression Functions
def butter_bandpass_sos(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band', output='sos')

def compress_and_flatten(data, threshold=0.05, ratio=10.0):
    abs_data = np.abs(data)
    compressed_data = np.copy(data)
    mask = abs_data > threshold
    if np.any(mask):
        compressed_data[mask] = np.sign(data[mask]) * (threshold + (abs_data[mask] - threshold) / ratio)
    max_val = np.max(np.abs(compressed_data))
    if max_val > 0:
        compressed_data = compressed_data / max_val
    return compressed_data

def process_audio_buffer(uploaded_file, lowcut=None, highcut=None):
    file_bytes = uploaded_file.read()
    
    if uploaded_file.name.lower().endswith('.mp3'):
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
        audio = audio.set_frame_rate(44100).set_channels(1)
        fs = audio.frame_rate
        data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        data = data / (2**15)
    else:
        data, fs = sf.read(io.BytesIO(file_bytes))
        
    if lowcut and highcut:
        sos = butter_bandpass_sos(lowcut, highcut, fs, order=8)
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]):
                audio_band = sosfilt(sos, data[:, channel])
                filtered_data[:, channel] = compress_and_flatten(audio_band)
        else:
            audio_band = sosfilt(sos, data)
            filtered_data = compress_and_flatten(audio_band)
    else:
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]):
                filtered_data[:, channel] = compress_and_flatten(data[:, channel])
        else:
            filtered_data = compress_and_flatten(data)
        
    virtual_file = io.BytesIO()
    sf.write(virtual_file, filtered_data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

# Clean UI Box for File Uploading
with st.container():
    uploaded_file = st.file_uploader("📂 Select an audio track from your computer", type=["mp3", "wav"])

if uploaded_file is not None:
    st.markdown("---")
    st.subheader("🎵 Available Clinical Stimuli")
    st.caption(f"Loaded source track: {uploaded_file.name}")
    
    # 1. Full-Range Conditioned Option gets its own prominent card at the top
    with st.container(border=True):
        st.markdown("**Conditioning & Baseline Track**")
        processed_buffer = process_audio_buffer(uploaded_file, None, None)
        uploaded_file.seek(0)
        
        st.download_button(
            label="📥 Download Full-Range (Flattened Original)",
            data=processed_buffer,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_Full-Range.wav",
            mime="audio/wav",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Narrow Band Frequencies**")

    # 2. Narrow Bands organized into a clean 2x2 grid layout
    bands = {
        "500Hz": ((420, 595), "Bass Tracking"),
        "1000Hz": ((841, 1189), "Core Speech"),
        "2000Hz": ((1682, 2378), "Consonants"),
        "4000Hz": ((3364, 4757), "High Whistle")
    }
    
    base_name = uploaded_file.name.rsplit('.', 1)[0]
    
    # Generate 2 columns to break up the vertical stack
    col1, col2 = st.columns(2)
    
    for idx, (band_name, ((low, high), description)) in enumerate(bands.items()):
        # Alternate items between left (col1) and right (col2) columns
        target_col = col1 if idx % 2 == 0 else col2
        
        with target_col:
            with st.container(border=True):
                st.markdown(f"### {band_name}")
                st.caption(description)
                
                processed_buffer = process_audio_buffer(uploaded_file, low, high)
                uploaded_file.seek(0)
                
                st.download_button(
                    label=f"📥 Download {band_name}",
                    data=processed_buffer,
                    file_name=f"{base_name}_{band_name}_NBN.wav",
                    mime="audio/wav",
                    key=f"btn_{band_name}",
                    use_container_width=True
                )
                
    st.balloons()
