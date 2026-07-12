import argparse
import time
from urllib import response
import wave
import os
import datetime
import threading
import msvcrt
from wsgiref import headers

import numpy as np
import subprocess
import requests
import base64
import json

import pyaudiowpatch as pyaudio

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

def setup_kotoba_whisper():
    """Set up the Kotoba Whisper model and return the inference pipeline."""
    model_id = "kotoba-tech/kotoba-whisper-v2.2"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda:0",
    )
    return pipe

try:
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
except Exception:
    load_silero_vad = None
    get_speech_timestamps = None

OLLAMA_URL = "http://localhost:11434/api/chat"

def speach_to_text_by_ollama(wav_file):
    """Send WAV bytes to Ollama API and return the transcribed text."""
    # Convert WAV bytes to base64
    with open(wav_file, "rb") as f:
        wav_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "gemma4:e2b",
        "messages": [
            {
                "role": "user",
                "content": "以下の音声を文字起こししてください。",
                "images": [wav_base64]  # 配列形式でBase64を入れる
            }
        ],
        "options": {"num_ctx": 8192*4},
        "stream": False  # ストリーミングを無効にして一括で受け取る場合
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
    result = response.json()
#   print(result['message']['content'])
    return result['message']['content']


def list_loopback_devices(pa_obj):
    """Print available WASAPI loopback devices."""
    print("Available loopback devices:")
    for device in pa_obj.get_loopback_device_info_generator():
        print(f"- index={device['index']}: {device['name']}")


def prepare_output_directory():
    """Prepare output directory named YYYYMMDD_hhmmss and return its path."""
    session_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Saving output into folder: {session_dir}")
    return session_dir


def esc_watcher(stop_event):
    """Thread function: set stop_event when ESC is pressed."""
    print("Press ESC to stop recording and exit.")
    while not stop_event.is_set():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b"\x1b":
                stop_event.set()
                break
        time.sleep(0.1)

def resolve_loopback_device(pa_obj, requested_index=None):
    """Resolve the loopback device to capture playback audio."""
    if requested_index is not None:
        return pa_obj.get_device_info_by_index(requested_index)

    try:
        return pa_obj.get_default_wasapi_loopback()
    except (OSError, LookupError):
        for device in pa_obj.get_loopback_device_info_generator():
            return device
        raise RuntimeError("No WASAPI loopback device was found.")


def resample_mono_bytes(data_bytes, src_rate, dst_rate, target_frames):
    """Resample mono PCM bytes from src_rate to dst_rate for a target frame count."""
    if target_frames <= 0:
        return b""

    if src_rate == dst_rate:
        arr = np.frombuffer(data_bytes, dtype=np.int16)
        if arr.size == target_frames:
            return data_bytes
        if arr.size == 0:
            return b""
        if arr.size < target_frames:
            padded = np.pad(arr, (0, target_frames - arr.size), mode="edge")
            return padded.astype(np.int16).tobytes()
        return arr[:target_frames].astype(np.int16).tobytes()

    arr = np.frombuffer(data_bytes, dtype=np.int16)
    if arr.size == 0:
        return b""

    src_duration = arr.size / float(src_rate)
    if src_duration <= 0:
        return b""

    source_times = np.arange(arr.size, dtype=np.float32) / float(src_rate)
    target_times = np.linspace(0.0, src_duration, num=target_frames, endpoint=False)
    resampled = np.interp(target_times, source_times, arr.astype(np.float32)).astype(np.int16)
    return resampled.tobytes()


def downmix_to_mono_bytes(data_bytes, src_channels):
    """Downmix interleaved multichannel PCM bytes to mono by averaging channels."""
    if src_channels <= 1:
        return data_bytes

    arr = np.frombuffer(data_bytes, dtype=np.int16)
    if arr.size == 0:
        return b""

    frames = arr.size // src_channels
    if frames <= 0:
        return b""

    interleaved = arr.reshape(frames, src_channels)
    mono = np.mean(interleaved, axis=1, dtype=np.float32).astype(np.int16)
    return mono.tobytes()


def record_split_stereo(output_path, duration_seconds=10, loopback_device_index=None, mic_device_index=None, rate=None):
    """Record mic input on the left channel and loopback output on the right channel."""
    pa_obj = pyaudio.PyAudio()
    stop_event = threading.Event()

    try:
        loopback_info = resolve_loopback_device(pa_obj, requested_index=loopback_device_index)
        if mic_device_index is not None:
            mic_info = pa_obj.get_device_info_by_index(mic_device_index)
        else:
            mic_info = pa_obj.get_default_input_device_info()

        print(f"Using loopback device: {loopback_info['name']}")
        print(f"Using mic device: {mic_info['name']}")

        mic_rate = int(mic_info.get("defaultSampleRate") or 44100)
        loopback_rate = int(loopback_info.get("defaultSampleRate") or 44100)
        common_rate = int(rate) if rate is not None else loopback_rate
        mic_channels = max(1, int(mic_info.get("maxInputChannels") or 1))
        loopback_channels = max(1, int(loopback_info.get("maxInputChannels") or 2))
        print(f"Detected sample rates: mic={mic_rate} Hz, loopback={loopback_rate} Hz; using {common_rate} Hz for both streams")
        print(f"Detected channel counts: mic={mic_channels}, loopback={loopback_channels}")
        buffer_duration_seconds = 0.02
        frames_per_buffer = max(1, int(common_rate * buffer_duration_seconds))
        target_frames = frames_per_buffer

        mic_stream = pa_obj.open(
            format=pyaudio.paInt16,
            channels=mic_channels,
            rate=common_rate,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=frames_per_buffer,
        )
        loopback_stream = pa_obj.open(
            format=pyaudio.paInt16,
            channels=loopback_channels,
            rate=common_rate,
            input=True,
            input_device_index=loopback_info["index"],
            frames_per_buffer=frames_per_buffer,
        )

        session_dir = prepare_output_directory()
        if output_path:
            mp3_path = output_path if os.path.isabs(output_path) else os.path.join(session_dir, output_path)
        else:
            mp3_path = os.path.join(session_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", str(common_rate),
            "-ac", "2",
            "-i", "pipe:0",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            mp3_path,
        ]

        print("Starting ffmpeg process to write split stereo MP3:", " ".join(ffmpeg_cmd))
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        watcher_thread = threading.Thread(target=esc_watcher, args=(stop_event,), daemon=True)
        watcher_thread.start()

        print(f"Recording split stereo to {mp3_path}. Press ESC to stop.")
        start_time = time.time()
        try:
            while not stop_event.is_set():
                if duration_seconds and duration_seconds > 0 and (time.time() - start_time) >= duration_seconds:
                    break

                loopback_data = loopback_stream.read(frames_per_buffer, exception_on_overflow=False)
                mic_data = mic_stream.read(frames_per_buffer, exception_on_overflow=False)
                if not mic_data or not loopback_data:
                    continue

                mic_mono = downmix_to_mono_bytes(mic_data, mic_channels)
                loopback_mono = downmix_to_mono_bytes(loopback_data, loopback_channels)
                left_bytes = resample_mono_bytes(mic_mono, mic_rate, common_rate, target_frames)
                right_bytes = resample_mono_bytes(loopback_mono, loopback_rate, common_rate, target_frames)
                count = min(len(left_bytes) // 2, len(right_bytes) // 2)
                if count <= 0:
                    continue

                left_arr = np.frombuffer(left_bytes, dtype=np.int16)[:count]
                right_arr = np.frombuffer(right_bytes, dtype=np.int16)[:count]
                stereo = np.empty(count * 2, dtype=np.int16)
                stereo[0::2] = left_arr
                stereo[1::2] = right_arr
                proc.stdin.write(stereo.tobytes())
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait()

        mic_stream.stop_stream()
        mic_stream.close()
        loopback_stream.stop_stream()
        loopback_stream.close()
        print("Stopped split stereo recording.")
    finally:
        pa_obj.terminate()


def record_loopback(output_path, duration_seconds=10, device_index=None, channels=None, rate=None):
    """Record raw loopback PCM and stream into ffmpeg to produce a single MP3.

    Recording preserves the input sample rate and channel count. The
    recording runs until ESC is pressed or (optionally) until a finite
    duration elapses when provided.
    """
    pa_obj = pyaudio.PyAudio()
    stop_event = threading.Event()
    pipeline = setup_kotoba_whisper()

    try:
        device_info = resolve_loopback_device(pa_obj, requested_index=device_index)
        print(f"Using device: {device_info['name']}")

        src_channels = int(channels or device_info.get("maxInputChannels") or 2)
        src_rate = int(rate or device_info.get("defaultSampleRate") or 44100)
        frames_per_buffer = 1024

        stream = pa_obj.open(
            format=pyaudio.paInt16,
            channels=src_channels,
            rate=src_rate,
            input=True,
            input_device_index=device_info["index"],
            frames_per_buffer=frames_per_buffer,
        )

        # Prepare output directory and MP3 path
        session_dir = prepare_output_directory()
        start_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mp3_name = f"{start_tag}.mp3"
        mp3_path = os.path.join(session_dir, mp3_name)

        # Build ffmpeg command to accept raw PCM s16le from stdin
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", str(src_rate),
            "-ac", str(src_channels),
            "-i", "pipe:0",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            mp3_path,
        ]

        print("Starting ffmpeg process to write MP3:", " ".join(ffmpeg_cmd))
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        # Start ESC watcher thread
        watcher_thread = threading.Thread(target=esc_watcher, args=(stop_event,), daemon=True)
        watcher_thread.start()

        print(f"Recording and encoding to {mp3_path}. Press ESC to stop.")
        start_time = time.time()
        try:
            while not stop_event.is_set():
                # stop if a finite duration was requested
                if duration_seconds and duration_seconds > 0 and (time.time() - start_time) >= duration_seconds:
                    break

                data = stream.read(frames_per_buffer, exception_on_overflow=False)
                if not data:
                    continue
                try:
                    proc.stdin.write(data)
                except BrokenPipeError:
                    print("ffmpeg process terminated unexpectedly.")
                    break

#                mono_data = downmix_to_mono_bytes(data, src_channels)
#                raw_bytes = resample_mono_bytes(mono_data, src_rate, 16000, frames_per_buffer)
#                pcm = np.frombuffer(raw_bytes, dtype=np.int16)
#                pcm = pcm.astype(np.float32) / 32768.0 
#                result = pipeline({"sampling_rate": 16000, "raw": pcm, "chunk_length_s": 15})
#                print(result["text"])                
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait()

        # Cleanup
        stream.stop_stream()
        stream.close()
        print("Stopped recording and finalized MP3.")

    finally:
        pa_obj.terminate()

def main():
    parser = argparse.ArgumentParser(description="Record Windows playback audio to a MP3 file")
    parser.add_argument("--output", default="output.mp3", help="Output MP3 file path")
    parser.add_argument("--duration", type=float, default=1800.0, help="Recording duration in seconds")
    parser.add_argument("--continuous", action="store_true", help="Run until ESC is pressed (continuous mode)")
    parser.add_argument("--device-index", type=int, default=None, help="WASAPI loopback device index")
    parser.add_argument("--mic-device-index", type=int, default=None, help="Microphone input device index")
    parser.add_argument("--channels", type=int, default=None, help="Number of channels to record")
    parser.add_argument("--rate", type=int, default=None, help="Recording sample rate")
    parser.add_argument("--list-devices", action="store_true", help="List available loopback devices and exit")
    parser.add_argument("--split-stereo", action="store_true", help="Record mic input on the left channel and loopback on the right channel")
    args = parser.parse_args()

    pa_obj = pyaudio.PyAudio()
    try:
        if args.list_devices:
            list_loopback_devices(pa_obj)
            return
    finally:
        pa_obj.terminate()

    duration = 0 if args.continuous else args.duration

    if args.split_stereo:
        record_split_stereo(
            output_path=args.output,
            duration_seconds=duration,
            loopback_device_index=args.device_index,
            mic_device_index=args.mic_device_index,
            rate=args.rate,
        )
    else:
        record_loopback(
            output_path=args.output,
            duration_seconds=duration,
            device_index=args.device_index,
            channels=args.channels,
            rate=args.rate,
        )


if __name__ == "__main__":
    main()

