import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import io
import zipfile
import os
import sqlite3
import base64

# --- PERSISTENCE LAYER ---
def get_db():
    conn = sqlite3.connect("vra_toolkit.db", check_same_thread=False)
    conn.execute('CREATE TABLE IF NOT EXISTS library_files (name TEXT PRIMARY KEY, blob BLOB)')
    conn.execute('CREATE TABLE IF NOT EXISTS favorites (name TEXT PRIMARY KEY)')
    return conn

db = get_db()

# --- APP CONFIG & SESSION INITIALIZATION ---
st.set_page_config(page_title="Neilio's VRA Toolkit", page_icon="🎧", layout="wide", initial_sidebar_state="collapsed")

if "session_tracks" not in st.session_state:
    files = db.execute("SELECT name, blob FROM library_files").fetchall()
    st.session_state.session_tracks = {name: blob for name, blob in files}

if "favorites" not in st.session_state:
    favs = db.execute("SELECT name FROM favorites").fetchall()
    st.session_state.favorites = [row[0] for row in favs]

# --- CSS & UI STYLING ---
st.markdown("""
    <style>
        .stApp { background-color: #0f172a !important; }
        audio { height: 32px !important; margin-bottom: 8px !important; margin-top: 2px !important; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- DSP ENGINE ---
def butter_filter_sos(cutoff_low, cutoff_high, fs, filter_type='band', order=8):
    nyq = 0.5 * fs
    if filter_type == 'low': sos = butter(order, cutoff_high / nyq, btype='low', output='sos')
    elif filter_type == 'high': sos = butter(order, cutoff_low / nyq, btype='high', output='sos')
    else: sos = butter(order, [cutoff_low / nyq, cutoff_high / nyq], btype='band', output='sos')
    return sos

def rms_normalize(data, target_db=-20.0, peak_limit=0.95):
    current_rms = np.sqrt(np.mean(data**2))
    if current_rms == 0: return data
    gain = (10 ** (target_db / 20.0)) / current_rms
    normalized_data = data * gain
    max_peak = np.max(np.abs(normalized_data))
    return (normalized_data / max_peak) * peak_limit if max_peak > peak_limit else normalized_data

def process_audio_buffer(file_bytes, lowcut=None, highcut=None, filter_type='band', order=8, trim_seconds=0.0):
    audio = AudioSegment.from_file(io.BytesIO(file_bytes)).set_frame_rate(44100).set_channels(1)
    data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
    fs = 44100
    if trim_seconds > 0: data = data[int(trim_seconds * fs):]
    if filter_type != 'raw':
        sos = butter_filter_sos(lowcut, highcut, fs, filter_type, order)
        data = sosfilt(sos, data)
    data = rms_normalize(data)
    virtual_file = io.BytesIO()
    sf.write(virtual_file, data, fs, format='WAV')
    virtual_file.seek(0)
    return virtual_file

def render_audiometer_channel(label, audio_buffer, element_key, preroll_offset):
    audio_src = f"data:audio/wav;base64,{base64.b64encode(audio_buffer.getvalue()).decode()}"
    st.markdown(f"""
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
        <div style="font-family: monospace; font-size: 0.8rem; color: #f8fafc; font-weight: bold; margin-bottom: 6px;">{label}</div>
        <audio id="audio_{element_key}" src="{audio_src}" controls></audio>
        <div style="display: flex; gap: 6px; margin-top: 6px;">
            <button onclick="document.getElementById('audio_{element_key}').play()" style="flex:1; background:#10b981; border:none; border-radius:4px; color:white;">▶ PLAY</button>
            <button onclick="var p=document.getElementById('audio_{element_key}'); p.currentTime=Math.max(0, p.currentTime-{preroll_offset}); p.play()" style="flex:1; background:#38bdf8; border:none; border-radius:4px;">↩️ JUMP</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN INTERFACE ---
ui_mode = st.radio("", ["🎛️ LIVE LINE-IN", "📦 BULK EXPORT"], horizontal=True)
all_tracks = list(st.session_state.session_tracks.keys())

# Upload Portal
new_uploads = st.file_uploader("Batch Import", type=["mp3", "wav"], accept_multiple_files=True)
if new_uploads:
    for u in new_uploads:
        if u.name not in st.session_state.session_tracks:
            b = u.read()
            st.session_state.session_tracks[u.name] = b
            db.execute("INSERT INTO library_files (name, blob) VALUES (?, ?)", (u.name, b))
            db.commit()
    st.rerun()

selected = st.selectbox("Select Active Track", ["-- Select --"] + all_tracks)

if selected != "-- Select --":
    active_bytes = st.session_state.session_tracks[selected]
    
    # Favorites Toggle
    is_fav = selected in st.session_state.favorites
    if st.button("⭐ Add/Remove Favorite"):
        if is_fav:
            st.session_state.favorites.remove(selected)
            db.execute("DELETE FROM favorites WHERE name = ?", (selected,))
        else:
            st.session_state.favorites.append(selected)
            db.execute("INSERT INTO favorites (name) VALUES (?)", (selected,))
        db.commit()
        st.rerun()

    trim = st.slider("Trim (s)", 0.0, 10.0, 0.0)
    preroll = st.slider("Pre-roll (s)", 0.0, 5.0, 2.0)
    
    if "LIVE" in ui_mode:
        col1, col2 = st.columns(2)
        with col1:
            render_audiometer_channel("FULL-RANGE", process_audio_buffer(active_bytes, trim_seconds=trim), "full", preroll)
        with col2:
            render_audiometer_channel("1000Hz NBN", process_audio_buffer(active_bytes, 841, 1189, 'band', 8, trim), "nbn", preroll)
