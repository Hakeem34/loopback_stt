import subprocess

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

import pprint

# 設定
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


def decode_mp3_to_pcm16k(mp3_path, sampling_rate=16000):
    """Decode MP3 to mono 16-bit PCM at the requested sample rate."""
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", mp3_path,
        "-ac", "1",
        "-ar", str(sampling_rate),
        "-f", "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    pcm16 = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm16.size == 0:
        return np.zeros(0, dtype=np.float32)
    return pcm16.astype(np.float32) / 32768.0


# 推論の実行
#audio = decode_mp3_to_pcm16k("20260712_221834.mp3", sampling_rate=16000)
#result = pipe(audio, sampling_rate=16000, chunk_length_s=15)
#result = pipe("20260712_221834.mp3", chunk_length_s=15)
#result = pipe("20260713_015757.mp3", chunk_length_s=15, return_timestamps="word")
result = pipe("output.mp3", chunk_length_s=15)
pprint.pprint(result)

