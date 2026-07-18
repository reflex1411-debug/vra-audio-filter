import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile
import os

# Set wide layout
st.set_page_config(page_title="Neilio's VRA Toolkit", page_icon="🎧", layout="wide", initial_sidebar_state="collapsed")

# Setup library
LIBRARY_DIR = "library"
if not os.path.exists(LIBRARY_DIR): os.makedirs(LIBRARY_DIR)
if "session_tracks" not in st.session_state: st.session_state.session_tracks = {}
if "favorites" not in st.session_state: st.session_state.favorites = []

# CSS Injection
st.markdown("""
    <style>
        .stApp { background-color: #0f172a !important; }
        .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
        audio { height: 32px !important; margin-bottom: 8px !important; margin-top: 2px !important; width: 100%; }
    </style>
    <div style="background: linear-gradient(180deg, #334155 0%, #1e293b 100%); padding: 15px 25px; border-radius: 8px 8px 0px 0px; border: 2px solid #475569; border-bottom: none; display: flex; align-items: center; justify-content: space-between; box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="text-align: left;">
            <h1 style="color: #f8fafc; margin: 0; font-family: monospace; font-size: 1.4rem; font-weight: 700;">NEILIO'S VRA CLINICAL STIMULI GENERATOR</h1>
            <div style="color: #94a3b8; font-size: 0.75rem; font-family: monospace; margin-top: 2px;">MODEL VRA-11 // RMS-CALIBRATED OUTPUT MATRIX</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Processing Functions
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

def calculate_rms(data): return np.sqrt(np.mean(data**2))

def rms_normalize(data, target_db=-20.0, peak_limit=0.95):
    current_rms = calculate_rms(data)
    if current_rms == 0: return data
    target_linear = 10 ** (target_db / 20.0)
    gain = target_linear / current_rms
    normalized_data = data * gain
    max_peak = np.max(np.abs(normalized_data))
    if max_peak > peak_limit: normalized_data = (normalized_data / max_peak) * peak_limit
    return normalized_data

def process_audio_buffer(file_source, lowcut=None, highcut=None, filter_type='band', order=8, trim_seconds=0.0):
    if isinstance(file_source, str):
        with open(file_source, 'rb') as f: file_bytes = f.read()
        is_mp3 = file_source.lower().endswith('.mp3')
    else:
        file_bytes = file_source.read()
        is_mp3 = file_source.name.lower().endswith('.mp3')
        file_source.seek(0)
    
    if is_mp3:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
        audio = audio.set_frame_rate(44100).set_channels(1)
        fs = audio.frame_rate
        data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    else:
        data, fs = sf.read(io.BytesIO(file_bytes))
        
    if trim_seconds > 0:
        start_sample = int(trim_seconds * fs)
        if start_sample < len(data): data = data[start_sample:, :] if len(data.shape) > 1 else data[start_sample:]
        
    if filter_type == 'raw': filtered_data = data
    else:
        sos = butter_filter_sos(lowcut, highcut, fs, filter_type=filter_type, order=order)
        if len(data.shape) > 1:
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[1]): filtered_data[:, channel] = sosfilt(sos, data[:, channel])
        else: filtered_data = sosfilt(sos, data)

    if len(filtered_data.shape) > 1:
        for channel in range(filtered_data.shape[1]): filtered_data[:, channel] = rms_normalize(filtered_data[:, channel], target_db=-20.0)
    else: filtered_data = rms_normalize(filtered_data, target_db=-20.0)
        
    virtual_file = io.BytesIO()
    sf.write(virtual_file, filtered_data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def get_freq_badge(freq_label):
    colors = {"500": "#f59e0b", "1000": "#38bdf8", "2000": "#a3e635", "4000": "#a855f7"}
    color = next((colors[k] for k in colors if k in freq_label), "#94a3b8")
    return f'<span style="background:{color}33; color:{color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-family: monospace; border: 1px solid {color}80;">{freq_label}</span>'

def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset, freq_tag=""):
    import base64
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
    audio_src = f"data:audio/wav;base64,{audio_base64}"
    badge = get_freq_badge(freq_tag) if freq_tag else ""
    
    html_code = f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
            <div style="font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold;">{label}</div>
            {badge}
        </div>
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

# Main UI
with st.container(border=True):
    ui_mode = st.radio("", ["🎛️ LIVE LINE-IN PRESENTATION DESK", "📦 BULK EXPORT & FILE DOWNLOAD CENTER"], horizontal=True, label_visibility="collapsed")
    
    stored_files = [f for f in os.listdir(LIBRARY_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    all_tracks = stored_files + list(st.session_state.session_tracks.keys())
    
    selected_track_name = st.selectbox("Select Active Track", options=["-- Select --"] + all_tracks)
    
    if selected_track_name != "-- Select --":
        active_target = os.path.join(LIBRARY_DIR, selected_track_name) if selected_track_name in stored_files else io.BytesIO(st.session_state.session_tracks[selected_track_name])
        
        trim_seconds = st.slider("Trim Intro (s)", 0.0, 30.0, 0.0)
        preroll_offset = st.slider("Pre-roll (s)", 0.0, 5.0, 2.0)
        
        stimuli_manifest = [
            {"label": "Full-Range", "low": None, "high": None, "type": "raw", "suffix": "Full-Range", "order": 8},
            {"label": "500Hz NBN", "low": 420, "high": 595, "type": "band", "suffix": "500Hz_NBN", "order": 8},
            {"label": "1000Hz NBN", "low": 841, "high": 1189, "type": "band", "suffix": "1000Hz_NBN", "order": 8},
            {"label": "2000Hz NBN", "low": 1682, "high": 2378, "type": "band", "suffix": "2000Hz_NBN", "order": 8},
            {"label": "4000Hz NBN", "low": 3364, "high": 4757, "type": "band", "suffix": "4000Hz_NBN", "order": 8}
        ]

        if "LIVE LINE-IN" in ui_mode:
            cols = st.columns(3)
            for i, item in enumerate(stimuli_manifest):
                buff = process_audio_buffer(active_target, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                freq_tag = "".join([c for c in item["label"] if c.isdigit()])
                with cols[i % 3]:
                    render_audiometer_channel(item["label"], buff, item["suffix"], preroll_offset, freq_tag=freq_tag)
        else:
            for item in stimuli_manifest:
                buff = process_audio_buffer(active_target, item["low"], item["high"], item["type"], item["order"], trim_seconds)
                st.download_button(f"Save {item['label']}", buff, file_name=f"{item['suffix']}.wav")
