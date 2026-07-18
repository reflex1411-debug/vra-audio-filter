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

# Custom Design Header with Embedded Hummingbird Logo
st.markdown("""
    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center; border-left: 5px solid #475569; display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <!-- Clean, modern minimalist SVG Hummingbird Logo -->
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="60" height="60" style="margin-bottom: 10px;">
            <path d="M70,30 C60,25 45,35 40,45 C38,42 32,30 20,25 C25,38 35,45 38,48 C30,52 15,55 5,52 C18,58 32,58 39,53 C41,60 45,75 42,90 C48,75 50,60 48,52 C55,48 85,32 95,30 C85,32 75,32 70,30 Z" fill="#475569"/>
        </svg>
        <h1 style="color: #1e293b; margin: 0; font-family: sans-serif; font-size: 2rem;">Neilio's VRA Audio Toolkit</h1>
        <p style="color: #64748b; font-size: 1rem; margin-top: 8px; margin-bottom: 0;">Transform standard tracks into calibrated, RMS-normalized clinical stimuli.</p>
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
with st.container():
    uploaded_file = st.file_uploader("📂 Select an audio track from your computer", type=["mp3", "wav"])

if uploaded_file is not None:
    st.markdown("---")
    
    # Custom HTML/CSS/JS Soundwave and Music Note Canvas Visualizer
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
            <div style="color: #38bdf8; font-weight: 600; font-size: 1.1rem; margin-bottom: 12px; font-family: sans-serif; display: flex; align-items: center; justify-content: center; gap: 8px;">
                <span>✨ Audio Processing Success</span>
            </div>
            <div style="display: flex; justify-content: center; align-items: flex-end; height: 50px; gap: 3px; margin: 15px auto;">
                <div style="width: 4px; background: #38bdf8; border-radius: 2px; animation: wave 1.2s ease-in-out infinite alternate; animation-delay: 0.1s;"></div>
                <div style="width: 4px; background: #0ea5e9; border-radius: 2px; animation: wave 0.9s ease-in-out infinite alternate; animation-delay: 0.4s;"></div>
                <div style="width: 4px; background: #0284c7; border-radius: 2px; animation: wave 1.4s ease-in-out infinite alternate; animation-delay: 0.2s;"></div>
                <div style="width: 4px; background: #38bdf8; border-radius: 2px; animation: wave 1.0s ease-in-out infinite alternate; animation-delay: 0.6s;"></div>
                <div style="width: 4px; background: #60a5fa; border-radius: 2px; animation: wave 1.1s ease-in-out infinite alternate; animation-delay: 0.3s;"></div>
                <div style="width: 4px; background: #3b82f6; border-radius: 2px; animation: wave 1.3s ease-in-out infinite alternate; animation-delay: 0.5s;"></div>
                <div style="width: 4px; background: #2563eb; border-radius: 2px; animation: wave 0.8s ease-in-out infinite alternate; animation-delay: 0.7s;"></div>
            </div>
            <div style="position: relative; height: 20px; overflow: hidden; font-size: 1.2rem; color: #64748b;">
                <span style="position: absolute; left: 20%; animation: floatNote 3s linear infinite; animation-delay: 0s;">♪</span>
                <span style="position: absolute; left: 40%; animation: floatNote 2.5s linear infinite; animation-delay: 0.5s;">♫</span>
                <span style="position: absolute; left: 60%; animation: floatNote 3.5s linear infinite; animation-delay: 0.2s;">♩</span>
                <span style="position: absolute; left: 80%; animation: floatNote 2.8s linear infinite; animation-delay: 0.8s;">♬</span>
            </div>
        </div>
        
        <style>
            @keyframes wave {
                0% { height: 10px; }
                100% { height: 45px; }
            }
            @keyframes floatNote {
                0% { transform: translateY(20px) scale(0.8); opacity: 0; }
                50% { opacity: 0.8; transform: translateY(-5px) scale(1.1); color: #38bdf8; }
                100% { transform: translateY(-30px) scale(0.8); opacity: 0; }
            }
        </style>
    """, unsafe_allow_html=True)
    
    base_name = uploaded_file.name.rsplit('.', 1)[0]
    
    # Master manifest detailing all 11 possible outputs for the bulk ZIP engine
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎵 Individual Track Downloads")
    st.caption(f"Source track: {uploaded_file.name}")
    
    # 1. Full-Range Baseline Row
    with st.container(border=True):
        st.markdown("**Conditioning & Baseline Track**")
        processed_buffer = process_audio_buffer(uploaded_file, None, None, 'raw', 8)
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
            processed_buffer = process_audio_buffer(uploaded_file, None, 1000, 'low', 8)
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
            processed_buffer = process_audio_buffer(uploaded_file, 1000, None, 'high', 8)
            uploaded_file.seek(0)
            st.download_button(
                label="📥 Download High-Pass",
                data=processed_buffer,
                file_name=f"{base_name}_HighPass_1kHz.wav",
                mime="audio/wav",
                use_container_width=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Dynamic Filter Profiling via TABS
    st.markdown("**Octave Band Selection**")
    tab_nbn, tab_fresh = st.tabs(["📊 Original Narrowband (NBN)", "⚡ FRESH-Style (Ultra-Steep)"])
    
    # Tab 1: Original NBN Layout
    with tab_nbn:
        st.caption("Standard 1/3 octave clinical noise bandwidth filtering profiles (Order 8).")
        col_n1, col_n2 = st.columns(2)
        nbn_items = [item for item in stimuli_manifest if "NBN" in item["suffix"]]
        
        for idx, item in enumerate(nbn_items):
            target_col = col_n1 if idx % 2 == 0 else col_n2
            with target_col:
                with st.container(border=True):
                    band_title = item["suffix"].split('_')[0]
                    st.markdown(f"### {band_title}")
                    st.caption(item["label"])
                    
                    processed_buffer = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"], item["order"])
                    uploaded_file.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download {band_title} NBN",
                        data=processed_buffer,
                        file_name=f"{base_name}_{item['suffix']}.wav",
                        mime="audio/wav",
                        key=f"btn_nbn_{band_title}",
                        use_container_width=True
                    )

    # Tab 2: FRESH-Style Layout
    with tab_fresh:
        st.caption("High-specificity steep filter boundaries (Order 20) to limit off-frequency listening.")
        col_f1, col_f2 = st.columns(2)
        fresh_items = [item for item in stimuli_manifest if "FRESH" in item["suffix"]]
        
        for idx, item in enumerate(fresh_items):
            target_col = col_f1 if idx % 2 == 0 else col_f2
            with target_col:
                with st.container(border=True):
                    band_title = item["suffix"].split('_')[0]
                    st.markdown(f"### {band_title}")
                    st.caption(item["label"])
                    
                    processed_buffer = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"], item["order"])
                    uploaded_file.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download {band_title} FRESH",
                        data=processed_buffer,
                        file_name=f"{base_name}_{item['suffix']}.wav",
                        mime="audio/wav",
                        key=f"btn_fresh_{band_title}",
                        use_container_width=True
                    )

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- BOTTOM LEVEL: BULK DOWNLOAD ZIP BUTTON ---
    st.subheader("📦 Bulk Actions")
    with st.container(border=True):
        st.markdown("Compile all 11 calibrated configurations into a single compressed folder.")
        
        with st.spinner("Building master ZIP package in memory..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for item in stimuli_manifest:
                    track_data = process_audio_buffer(uploaded_file, item["low"], item["high"], item["type"], item["order"])
                    uploaded_file.seek(0)
                    filename = f"{base_name}_{item['suffix']}.wav"
                    zip_file.writestr(filename, track_data.getvalue())
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📦 Download All 11 Tracks (.ZIP)",
                data=zip_buffer,
                file_name=f"{base_name}_VRA_Complete_Set.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )
