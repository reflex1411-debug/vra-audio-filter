import io
import re
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from pedalboard import HighpassFilter, Limiter, LowpassFilter, Pedalboard
from pedalboard.io import AudioFile

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
# 2. UNIVERSAL DSP ENGINE (BROADBAND, LPF, HPF, BPF)
# ==============================================================================


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
    """Processes an audio buffer through specified filter geometry, applies a post-filter

    Limiter to smash band peak spikes, and normalises strictly to target RMS.
    """
    board_plugins = []

    # 1. Determine Filtering Chain
    if filter_type == "Broadband":
        # Sub-bass rumble protection only
        board_plugins.append(HighpassFilter(cutoff_frequency_hz=20.0))

    elif filter_type == "Low Pass":
        board_plugins.append(LowpassFilter(cutoff_frequency_hz=high_cutoff_hz))

    elif filter_type == "High Pass":
        board_plugins.append(HighpassFilter(cutoff_frequency_hz=low_cutoff_hz))

    elif filter_type == "Band Pass":
        low_f = max(20, center_freq / (2 ** (1 / (2 * q_factor))))
        high_f = min(
            sample_rate / 2 - 100, center_freq * (2 ** (1 / (2 * q_factor)))
        )
        board_plugins.append(HighpassFilter(cutoff_frequency_hz=low_f))
        board_plugins.append(LowpassFilter(cutoff_frequency_hz=high_f))

    # 2. Add Post-Filter Limiter to Enforce Tight Dynamic Span
    board_plugins.append(Limiter(threshold_db=limiter_ceiling_db))

    # 3. Execute Pedalboard DSP
    board = Pedalboard(board_plugins)
    filtered_audio = board(audio_data, sample_rate)

    # 4. Target RMS Normalisation (-20.00 dBFS)
    current_rms = np.sqrt(np.mean(filtered_audio**2))

    if current_rms > 0:
        target_rms_linear = 10 ** (target_rms_db / 20.0)
        gain_scale = target_rms_linear / current_rms
        final_audio = filtered_audio * gain_scale
    else:
        final_audio = filtered_audio

    # 5. Calculate Metrics
    final_peak_db = 20 * np.log10(np.max(np.abs(final_audio)) + 1e-9)
    final_rms_db = 20 * np.log10(np.sqrt(np.mean(final_audio**2)) + 1e-9)
    span_db = final_peak_db - final_rms_db

    # 6. Export WAV
    out_buffer = io.BytesIO()
    sf.write(out_buffer, final_audio.T, int(sample_rate), format="WAV")
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

# Sidebar Calibration Settings
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
    help=(
        "Clamps post-filter peak spikes to keep dynamic span tightly"
        " constrained."
    ),
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
    "Upload VRA Master Music / Stimulus Clip (.wav, .mp3, .flac):",
    type=["wav", "mp3", "flac", "ogg"],
)

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()

    # Read audio into memory
    with AudioFile(io.BytesIO(raw_bytes)) as f:
        master_audio = f.read(f.frames)
        sr = f.samplerate

    st.subheader("🔊 Master Input Audio")
    st.audio(raw_bytes)
    st.divider()

    st.subheader("🎚️ Calibrated Filter Outputs")

    # Define the 7 Filter Configurations
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

    # Process and display each band in interactive expanders / grids
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
            band_tag = cfg["name"].replace(" ", "_").replace("(", "").replace(")", "")
            out_filename = f"{file_stem}_{band_tag}_RMS-20dB.wav"

            st.download_button(
                label=f"📥 Download {cfg['name']} (.wav)",
                data=wav_data,
                file_name=out_filename,
                mime="audio/wav",
                key=f"btn_{cfg['name']}",
            )
