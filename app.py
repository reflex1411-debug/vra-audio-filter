import streamlit as st
import numpy as np
import scipy.signal as signal
import io
import base64

st.set_page_config(page_title="Clinical Music Filter Tool", layout="wide")

# ------------------------------------------------------------
# AUDIO FILTERS
# ------------------------------------------------------------

def low_pass_filter(data, sr, cutoff=300):
    sos = signal.butter(6, cutoff, 'low', fs=sr, output='sos')
    return signal.sosfilt(sos, data)

def high_pass_filter(data, sr, cutoff=4000):
    sos = signal.butter(6, cutoff, 'high', fs=sr, output='sos')
    return signal.sosfilt(sos, data)

def nbn_filter(data, sr, low=1000, high=2000):
    sos = signal.butter(6, [low, high], 'bandpass', fs=sr, output='sos')
    return signal.sosfilt(sos, data)

def to_wav_buffer(data, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, data, sr, format='WAV')
    buf.seek(0)
    return buf

# ------------------------------------------------------------
# UI COMPONENT
# ------------------------------------------------------------

def render_audiometer_channel(label, audio_buffer, element_key):
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    audio_src = f"data:audio/wav;base64,{audio_base64}"

    COLOR_MAP = {
        "FRESH": "#f59e0b",
        "LOW-PASS": "#8b5cf6",
        "HIGH-PASS": "#8b5cf6",
        "NBN": "#38bdf8",
    }
    border_color = COLOR_MAP.get(label, "#334155")

    FREQ_RANGES = {
        "FRESH": "Full spectrum",
        "LOW-PASS": "<300 Hz",
        "HIGH-PASS": ">4 kHz",
        "NBN": "1–2 kHz",
    }
    freq_range = FREQ_RANGES.get(label, "—")

    ICONS = {
        "LOW-PASS": f"<svg width='14' height='14'><polyline points='2,2 12,12' stroke='{border_color}' stroke-width='2' fill='none'/></svg>",
        "HIGH-PASS": f"<svg width='14' height='14'><polyline points='2,12 12,2' stroke='{border_color}' stroke-width='2' fill='none'/></svg>",
        "NBN": f"<svg width='14' height='14'><rect x='3' y='4' width='8' height='6' stroke='{border_color}' stroke-width='2' fill='none'/></svg>",
        "FRESH": f"<svg width='14' height='14'><rect x='2' y='6' width='10' height='2' fill='{border_color}'/></svg>",
    }
    icon_svg = ICONS.get(label, "")

    SPECTRUM = {
        "LOW-PASS": f"<svg width='100%' height='40'><polyline points='0,35 20,30 40,25 60,18 80,10 100,5' stroke='{border_color}' stroke-width='3' fill='none'/></svg>",
        "HIGH-PASS": f"<svg width='100%' height='40'><polyline points='0,5 20,10 40,18 60,25 80,30 100,35' stroke='{border_color}' stroke-width='3' fill='none'/></svg>",
        "NBN": f"<svg width='100%' height='40'><polyline points='0,35 20,35 40,10 60,10 80,35 100,35' stroke='{border_color}' stroke-width='3' fill='none'/></svg>",
        "FRESH": f"<svg width='100%' height='40'><rect x='0' y='18' width='100%' height='4' fill='{border_color}'/></svg>",
    }
    spectrum_svg = SPECTRUM.get(label, "")

    freq_badge = f"""
        <span style="
            background:#0f172a;
            padding:2px 6px;
            border-radius:4px;
            color:{border_color};
            font-size:0.7rem;
            margin-left:6px;
            border:1px solid {border_color}55;
        ">
            {freq_range}
        </span>
    """

    html_code = f"""
    <div style="
        background-color:#1e293b;
        border-left:6px solid {border_color};
        border:1px solid #334155;
        border-radius:0 6px 6px 0;
        padding:12px;
        margin-bottom:18px;
        box-shadow:0 0 6px {border_color}55;
    ">
        <div style="
            font-family:monospace;
            font-size:0.85rem;
            color:#f8fafc;
            font-weight:bold;
            margin-bottom:8px;
            display:flex;
            align-items:center;
            gap:6px;
        ">
            {icon_svg}
            <span style="color:{border_color};">●</span>
            <span>{label}</span>
            {freq_badge}
        </div>

        <div style="margin-bottom:8px;">
            {spectrum_svg}
        </div>

        <audio id="audio_{element_key}" src="{audio_src}" controls style="width:100%;"></audio>
    </div>
    """

    st.components.v1.html(html_code, height=220)

# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------

st.title("Clinical Music Filter Tool")
st.write("Upload a WAV file to generate filtered versions for clinical listening.")

uploaded = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded:
    import soundfile as sf
    data, sr = sf.read(uploaded)

    if data.ndim > 1:
        data = data[:, 0]  # convert to mono

    fresh_buf = to_wav_buffer(data, sr)
    lp_buf = to_wav_buffer(low_pass_filter(data, sr), sr)
    hp_buf = to_wav_buffer(high_pass_filter(data, sr), sr)
    nbn_buf = to_wav_buffer(nbn_filter(data, sr), sr)

    st.subheader("Filtered Outputs")

    render_audiometer_channel("FRESH", fresh_buf, "fresh")
    render_audiometer_channel("LOW-PASS", lp_buf, "lp")
    render_audiometer_channel("HIGH-PASS", hp_buf, "hp")
    render_audiometer_channel("NBN", nbn_buf, "nbn")
