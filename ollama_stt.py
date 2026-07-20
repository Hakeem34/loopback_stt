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



def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Ollama API.")
    parser.add_argument("wav_file", type=str, help="Path to the WAV file to transcribe.")
    args = parser.parse_args()

    wav_file = args.wav_file
    if not os.path.isfile(wav_file):
        print(f"Error: File '{wav_file}' does not exist.")
        return
    
    speech_text = speach_to_text_by_ollama(wav_file)
    print("Transcribed Text:")
    print(speech_text)

if __name__ == "__main__":
    main()

