import io
import re
import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & APPLE HIG STYLING
# ==============================================================================

st.set_page_config(
    page_title="CYPAC VRA Stimulus Calibrator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif !important;
        }
        .stApp { background-color: #f5f5f7 !important; }
        .main .block-container { padding-top: 1.8rem; max-width: 1280px; }
        
        div[data-testid="metric-container"] {
            background-color: #ffffff !important;
            border-radius: 16px !important;
            padding: 16px 20px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
            border: 1px solid rgba(0, 0, 0, 0.04) !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.6rem !important; font-weight: 700 !important; color: #1d1d1f !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important; font-weight: 600 !important; color: #8e8e93 !important; text-transform: uppercase;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. SCIPY-BASED DSP ENGINE (BROADBAND, LPF, HPF, BPF)
# ==============================================================================


def apply_butterworth_filter(
    data,
    sample_rate,
    filter_type="Broadband",
    center_freq=1000,
    low_cutoff_hz=250,
    high_cutoff_hz=4000,
    q_factor=2.0,
):
    """Applies high-order Butterworth digital filtering using scipy.signal."""
    nyquist = 0.5 * sample_rate

    if filter_type == "Broadband":
        # 20 Hz High-pass for sub-bass rumble
        b, a = signal.butter(
            4, 20.0 / nyquist, btype="highpass", analog=False
        )
        return signal.filtfilt(b, a, data, axis=0)

    elif filter_type == "Low Pass":
        cutoff = min(high_cutoff_hz / nyquist, 0.99)
        b, a = signal.butter(4, cutoff, btype="lowpass", analog=False)
        return signal.filtfilt(b, a, data, axis=0)

    elif filter_type == "High Pass":
        cutoff = max(low_cutoff_hz / nyquist, 0.001)
        b, a = signal.butter(4, cutoff, btype="highpass", analog=False)
        return signal.filtfilt(b, a, data, axis=0)

    elif filter_type == "Band Pass":
        low_f = max(20, center_freq / (2 ** (1 / (2 * q_factor))))
        high_f = min(sample_rate / 2 - 100, center_freq * (2 ** (1 / (2 * q_factor))))
        
        low_norm = max(low_f / nyquist, 0.001)
        high_norm = min(high_f / nyquist, 0.99)

        b, a = signal.butter(
            4, [low_norm, high_norm], btype="bandpass", analog=False
        )
        return signal.filtfilt(b, a, data, axis=0)

    return data


def apply_limiter(data, ceiling_db=-6.0):
    """Soft-clipping limiter to cap post-filter peak spikes."""
    ceiling_linear = 10 ** (ceiling_db / 20.0)
    # Smooth tanh compression curve when signal exceeds threshold
    scaled = data / ceiling_linear
    limited = np.tanh(scaled) * ceiling_linear
    return limited


def process_vra_band(
    audio_data,
    sample_rate,
    filter_type="Broadband",
    center_freq=1000,
    low_cutoff_hz=250,
    high_cutoff_hz=4000,
    q_factor=2.0,
    target_rms_db=-20.0,
    limiter_ceiling_db=-6.0,
):
    # 1. Filter Audio
    filtered = apply_butterworth_filter(
        audio_data,
        sample_rate,
        filter_type=filter_type,
        center_freq=center_freq,
        low_cutoff_hz=low_cutoff_hz,
        high_cutoff_hz=high_cutoff_hz,
        q_factor=q_factor,
    )

    # 2. Target RMS Normalisation (-20.00 dBFS)
    current_rms = np.sqrt(np.mean(filtered**2))

    if current_rms > 0:
        target_rms_linear = 10 ** (target_rms_db / 20.0)
        gain_scale = target_rms_linear / current_rms
        scaled_audio = filtered * gain_scale
    else:
        scaled_audio = filtered

    # 3. Post-Filter Limiter Ceiling
    final_audio = apply_limiter(scaled_audio, ceiling_db=limiter_ceiling_db)

    # 4. Analytics
    final_peak_db = 20 * np.log10(np.max(np.abs(final_audio)) + 1e-9)
    final_rms_db = 20 * np.log10(np.sqrt(np.mean(final_audio**2)) + 1e-9)
    span_db = final_peak_db - final_rms_db

    # 5. Export WAV
    out_buffer = io.BytesIO()
    sf.write(out_buffer, final_audio, int(sample_rate), format="WAV")
    out_buffer.seek(0)

    return out_buffer.getvalue(), {
        "Peak": f"{final_peak_db:.2f} dBFS",
        "RMS": f"{final_rms_db:.2f} dBFS",
        "Span": f"±{span_db:.2f} dB",
    }


# ==============================================================================
# 3. STREAMLIT APPLICATION UI
# ==============================================================================

st.title("🎧 CYPAC VRA Stimulus Calibrator & Filter Bank")
st.caption(
    "Clinical Audiometric Suite: Broadband, Low Pass, High Pass, and 500 Hz – 4"
    " kHz Band-Pass Filters"
)
st.divider()

# Sidebar Settings
st.sidebar.header("🎛️ Audiometric DSP Controls")

target_rms = st.sidebar.number_input(
    "Target RMS Loudness (dBFS)",
    min_value=-30.0,
    max_value=-10.0,
    value=-20.0,
    step=0.5,
    help="Default audiometer line level calibration standard.",
)

limiter_ceiling = st.sidebar.slider(
    "Post-Filter Limiter Ceiling (dBFS)",
    min_value=-12.0,
    max_value=-1.0,
    value=-6.0,
    step=0.5,
    help="Clamps post-filter peak spikes to keep dynamic span tightly constrained.",
)

bpf_q = st.sidebar.slider(
    "Band-Pass Filter Q-Factor",
    min_value=0.5,
    max_value=4.0,
    value=2.0,
    step=0.1,
    help="Filter selectivity for 500 Hz, 1 kHz, 2 kHz, and 4 kHz bands.",
)

lp_cutoff = st.sidebar.number_input(
    "Low Pass Cutoff (Hz)", value=1000, step=100
)
hp_cutoff = st.sidebar.number_input(
    "High Pass Cutoff (Hz)", value=1000, step=100
)

# Upload Section
uploaded_file = st.file_uploader(
    "Upload VRA Master Music / Stimulus Clip (.wav, .flac, .ogg):",
    type=["wav", "flac", "ogg"],
)

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()

    # Read audio into numpy array via soundfile
    master_audio, sr = sf.read(io.BytesIO(raw_bytes))

    st.subheader("🔊 Master Input Audio")
    st.audio(raw_bytes)
    st.divider()

    st.subheader("🎚️ Calibrated Filter Outputs")

    FILTER_CONFIGS = [
        {"name": "Broadband (Unfiltered)", "type": "Broadband", "freq": None},
        {
            "name": f"Low Pass ({lp_cutoff} Hz)",
            "type": "Low Pass",
            "freq": None,
        },
        {
            "name": f"High Pass ({hp_cutoff} Hz)",
            "type": "High Pass",
            "freq": None,
        },
        {"name": "500 Hz BPF", "type": "Band Pass", "freq": 500},
        {"name": "1000 Hz (1 kHz) BPF", "type": "Band Pass", "freq": 1000},
        {"name": "2000 Hz (2 kHz) BPF", "type": "Band Pass", "freq": 2000},
        {"name": "4000 Hz (4 kHz) BPF", "type": "Band Pass", "freq": 4000},
    ]

    for cfg in FILTER_CONFIGS:
        with st.expander(f"🎵 {cfg['name']}", expanded=True):
            wav_data, stats = process_vra_band(
                master_audio,
                sr,
                filter_type=cfg["type"],
                center_freq=cfg["freq"] if cfg["freq"] else 1000,
                low_cutoff_hz=hp_cutoff,
                high_cutoff_hz=lp_cutoff,
                q_factor=bpf_q,
                target_rms_db=target_rms,
                limiter_ceiling_db=limiter_ceiling,
            )

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.audio(wav_data, format="audio/wav")

            with col2:
                st.metric("RMS Level", stats["RMS"])

            with col3:
                st.metric("Peak Level", stats["Peak"])

            with col4:
                st.metric("Span", stats["Span"])

            file_stem = re.sub(r"\.[^.]+$", "", uploaded_file.name)
            band_tag = (
                cfg["name"]
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            out_filename = f"{file_stem}_{band_tag}_RMS-20dB.wav"

            st.download_button(
                label=f"📥 Download {cfg['name']} (.wav)",
                data=wav_data,
                file_name=out_filename,
                mime="audio/wav",
                key=f"btn_{cfg['name']}",
            )
