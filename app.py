import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io

# Set up the web page header
st.set_page_config(page_title="Paediatric VRA Audio Filter", page_icon="🎧", layout="centered")
st.title("🎧 Paediatric VRA Audio Filter")
st.markdown("Upload any nursery rhyme or children's song (.mp3 or .wav) to automatically generate frequency-specific or dynamically flattened VRA stimuli.")

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
        
    # If cutoffs are provided, apply the narrow bandpass filter. 
    # If they are None, skip filtering and pass the raw track straight to the compressor.
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

# UI File Uploader
uploaded_file = st.file_uploader("Drag and drop your audio file here", type=["mp3", "wav"])

if uploaded_file is not None:
    st.success(f"Successfully loaded: {uploaded_file.name}")
    st.info("Processing clinical bands... please wait a moment.")
    
    # Define VRA Targets (added Full-Range Flattened option)
    bands = {
        "Full-Range (Flattened Original)": (None, None),
        "500Hz (Bass tracking)": (420, 595),
        "1000Hz (Core speech)": (841, 1189),
        "2000Hz (Consonants)": (1682, 2378),
        "4000Hz (High-frequency whistle)": (3364, 4757)
    }
    
    base_name = uploaded_file.name.rsplit('.', 1)[0]
    
    for label, (low, high) in bands.items():
        band_suffix = label.split()[0]
        
        processed_buffer = process_audio_buffer(uploaded_file, low, high)
        uploaded_file.seek(0)
        
        st.download_button(
            label=f"📥 Download {label} Track",
            data=processed_buffer,
            file_name=f"{base_name}_{band_suffix}.wav",
            mime="audio/wav",
            key=label
        )
    
    st.balloons()if uploaded_file is not None:
    st.success(f"Successfully loaded: {uploaded_file.name}")
    st.info("Processing clinical bands... please wait a moment.")
    
    # Define VRA Targets
    bands = {
        "500Hz (Bass tracking)": (420, 595),
        "1000Hz (Core speech)": (841, 1189),
        "2000Hz (Consonants)": (1682, 2378),
        "4000Hz (High-frequency whistle)": (3364, 4757)
    }
    
    base_name = uploaded_file.name.rsplit('.', 1)[0]
    
    # Process each band and create layout rows for downloading
    for label, (low, high) in bands.items():
        band_suffix = label.split()[0] # e.g., "500Hz"
        
        # Process the audio array
        processed_buffer = process_audio_buffer(uploaded_file, low, high)
        
        # Reset uploaded file read pointer for next loop iteration
        uploaded_file.seek(0)
        
        # Create a visually clean download bar for each track
        st.download_button(
            label=f"📥 Download {label} Track",
            data=processed_buffer,
            file_name=f"{base_name}_{band_suffix}.wav",
            mime="audio/wav",
            key=label
        )
    
    st.balloons()
