import argparse
import struct
import subprocess
import wave
import dataclasses
from pathlib import Path
import torch
import base64
import requests
import json


@dataclasses.dataclass
class AudioSegment:
    start: float
    end: float
    channel: int = 0
    file_path: Path | None = None
    text: str = ""

g_segments: list[AudioSegment] = []

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
except Exception:
    load_silero_vad = None
    get_speech_timestamps = None


def get_audio_channel_count(audio_path: str) -> int:
    """Return the number of audio channels for MP3 or WAV input."""
    input_path = Path(audio_path)
    if input_path.suffix.lower() == ".wav":
        with wave.open(str(input_path), "rb") as wav_file:
            return wav_file.getnchannels()

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "default=nw=1:nk=1",
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip() or "1")


def decode_mp3_to_pcm_bytes(mp3_path: str, sample_rate: int = 16000, channel_count: int = 1) -> bytes:
    """Decode MP3 audio to 16-bit PCM bytes at the requested sample rate."""
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        mp3_path,
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channel_count),
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout


def write_wav_file(output_path: Path, pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> None:
    """Write PCM bytes to a WAV file with 16-bit sample width."""
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def split_stereo_pcm(pcm_bytes: bytes):
    """Split interleaved stereo PCM bytes into left/right mono PCM bytes."""
    if len(pcm_bytes) % 4 != 0:
        raise ValueError("Stereo PCM bytes length is not divisible by 4")

    samples = struct.iter_unpack("<h", pcm_bytes)
    left_samples = []
    right_samples = []
    for index, sample in enumerate(samples):
        if index % 2 == 0:
            left_samples.append(sample[0])
        else:
            right_samples.append(sample[0])

    return struct.pack("<{}h".format(len(left_samples)), *left_samples), struct.pack("<{}h".format(len(right_samples)), *right_samples)


def read_wav_to_pcm_bytes(wav_path: str | Path) -> tuple[bytes, int, int]:
    """Read a WAV file and return PCM bytes plus its sample rate and channel count."""
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())
    return pcm_bytes, sample_rate, channels


def downmix_to_mono_pcm_bytes(pcm_bytes: bytes, channels: int) -> bytes:
    """Downmix interleaved multi-channel PCM bytes to mono PCM bytes."""
    if channels <= 1:
        return pcm_bytes

    frame_count = len(pcm_bytes) // (2 * channels)
    if frame_count <= 0:
        return b""

    fmt = "<" + "h" * channels
    mono_samples = []
    for frame in struct.iter_unpack(fmt, pcm_bytes):
        mono_samples.append(sum(frame) // len(frame))

    return struct.pack("<{}h".format(len(mono_samples)), *mono_samples)


def pcm_bytes_to_float_tensor(pcm_bytes: bytes, sample_rate: int) -> torch.Tensor:
    """Convert 16-bit PCM bytes to a normalized float tensor for Silero VAD."""
    if not pcm_bytes:
        return torch.empty(0, dtype=torch.float32)

    samples = [sample[0] / 32768.0 for sample in struct.iter_unpack("<h", pcm_bytes)]
    return torch.tensor(samples, dtype=torch.float32)


def write_segment_wavs(
    channel: int,
    pcm_bytes: bytes,
    sample_rate: int,
    timestamps: list[dict],
    output_dir: str | Path,
    prefix: str = "segment",
) -> list[Path]:
    """Write speech segments as individual WAV files from PCM bytes."""
    global g_segments

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = [sample[0] for sample in struct.iter_unpack("<h", pcm_bytes)]
    output_paths: list[Path] = []
    for index, timestamp in enumerate(timestamps or [], start=1):
        #print(f"Writing segment {index}: {timestamp}")
        start_sec = float(timestamp.get("start", 0.0))
        end_sec = float(timestamp.get("end", len(samples) / max(sample_rate, 1)))
        start_sample = max(0, int(round(start_sec * sample_rate)))
        end_sample = max(start_sample, int(round(end_sec * sample_rate)))
        segment_samples = samples[start_sample:end_sample]
        if not segment_samples:
            continue

        segment_bytes = struct.pack("<{}h".format(len(segment_samples)), *segment_samples)
        output_path = output_dir / f"{prefix}_{index:03d}.wav"
        write_wav_file(output_path, segment_bytes, sample_rate=sample_rate, channels=1)
        g_segments.append(AudioSegment(start=start_sec, end=end_sec, channel=channel, file_path=output_path))
        output_paths.append(output_path)

    return output_paths


def split_wav_with_silero_vad(
    wav_path: str | Path,
    channel: int = 0,
    output_dir: str | Path | None = None,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 30,
) -> list[Path]:
    """Split a mono WAV file into smaller WAVs using Silero VAD."""
    if torch is None or load_silero_vad is None or get_speech_timestamps is None:
        raise RuntimeError("Silero VAD is not available in this environment")

    input_path = Path(wav_path)
    pcm_bytes, sample_rate, channels = read_wav_to_pcm_bytes(input_path)
    if channels > 1:
        pcm_bytes = downmix_to_mono_pcm_bytes(pcm_bytes, channels)

    audio_tensor = pcm_bytes_to_float_tensor(pcm_bytes, sample_rate=sample_rate)
    if audio_tensor.numel() == 0:
        return []

    model = load_silero_vad()
    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )

    output_folder = Path(output_dir) if output_dir else input_path.parent / f"{input_path.stem}_vad"
    return write_segment_wavs(
        channel=channel,
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        timestamps=timestamps,
        output_dir=output_folder,
        prefix=input_path.stem,
    )


def convert_mp3_to_wav(
    mp3_path: str,
    output_folder: str | None = None,
    sample_rate: int = 16000
) -> list[Path]:
    """Convert an MP3/WAV into one or more mono WAV files and optionally split them with VAD."""
    input_path = Path(mp3_path)

    channel_count = get_audio_channel_count(str(input_path))
    output_paths: list[Path] = []
    if input_path.suffix.lower() == ".wav":
        pcm_bytes, _, _ = read_wav_to_pcm_bytes(input_path)
        output_path = output_folder / f"{input_path.stem}.wav"
        write_wav_file(output_path, pcm_bytes, sample_rate=sample_rate, channels=1)
        output_paths.append(output_path)
    elif channel_count == 2:
        pcm_bytes = decode_mp3_to_pcm_bytes(str(input_path), sample_rate=sample_rate, channel_count=2)
        left_bytes, right_bytes = split_stereo_pcm(pcm_bytes)
        left_path = output_folder / f"{input_path.stem}_L.wav"
        right_path = output_folder / f"{input_path.stem}_R.wav"
        write_wav_file(left_path, left_bytes, sample_rate=sample_rate, channels=1)
        write_wav_file(right_path, right_bytes, sample_rate=sample_rate, channels=1)
        output_paths.extend([left_path, right_path])
    else:
        pcm_bytes = decode_mp3_to_pcm_bytes(str(input_path), sample_rate=sample_rate, channel_count=1)
        output_path = output_folder / f"{input_path.stem}.wav"
        write_wav_file(output_path, pcm_bytes, sample_rate=sample_rate, channels=1)
        output_paths.append(output_path)

    return output_paths


OLLAMA_URL = "http://localhost:11434/api/chat"
def speach_to_text_by_ollama(wav_file):
    """Send WAV bytes to Ollama API and return the transcribed text."""
    # Convert WAV bytes to base64
    with open(wav_file, "rb") as f:
        wav_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "gemma4:e2b",
        #"model": "gemma4:e4b",
        #"model": "gemma4:12b",
        "messages": [
            {
                "role": "user",
                "content": "音声を文字起こししてください。もし有効な音声データを判断できなかった場合は、""---""とだけ回答してください。",
                "images": [wav_base64]  # 配列形式でBase64を入れる
            }
        ],
        "options": {"num_ctx": 8192*4},
        "stream": False  # ストリーミングを無効にして一括で受け取る場合
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
    result = response.json()
    print(result['message']['content'])
    return result['message']['content']

def setup_kotoba_whisper():
    """Set up the Kotoba Whisper model."""
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline
    )

    # 設定
    global model_id, torch_dtype, device, model_kwargs
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MP3/WAV audio to 16-bit WAV files and optionally split by Silero VAD")
    parser.add_argument("input_mp3", help="Path to the input MP3 or WAV file")
    parser.add_argument("--output-dir", help="Directory to save the output WAV file(s)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate in Hz")
    parser.add_argument("--split-vad", action="store_true", help="Split each WAV into smaller speech segments with Silero VAD")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD threshold")
    parser.add_argument("--min-speech-duration-ms", type=int, default=250, help="Minimum speech duration in milliseconds")
    parser.add_argument("--min-silence-duration-ms", type=int, default=300, help="Minimum silence duration in milliseconds")
    parser.add_argument("--speech-pad-ms", type=int, default=30, help="Padding around detected speech in milliseconds")
    parser.add_argument("--tts-kotoba", action="store_true", help="Use Kotoba Whisper for TTS instead of Silero VAD")
    parser.add_argument("--tts-ollama", action="store_true", help="Use Ollama(Gemma4:e2b) for TTS instead of Silero VAD")
    args = parser.parse_args()

    output_folder = Path(args.output_dir) if args.output_dir else Path(args.input_mp3).parent / f"{Path(args.input_mp3).stem}_wav"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_paths = convert_mp3_to_wav(
        args.input_mp3,
        output_folder=output_folder,
        sample_rate=args.sample_rate
    )

    segmented_paths: list[Path] = []
    if args.split_vad:
        channel = 0
        for output_path in output_paths:
            segmented_paths.extend(
                split_wav_with_silero_vad(
                    output_path,
                    channel,
                    output_dir=output_folder / output_path.stem,
                    threshold=args.vad_threshold,
                    min_speech_duration_ms=args.min_speech_duration_ms,
                    min_silence_duration_ms=args.min_silence_duration_ms,
                    speech_pad_ms=args.speech_pad_ms,
                )
            )
            channel += 1

        sorted_segments = sorted(g_segments, key=lambda s: (s.start, s.end, s.channel))
        if args.tts_ollama:
            output_text_path = args.input_mp3.replace(".mp3", "_transcription_ollama.txt").replace(".wav", "_transcription.txt")
            with open(output_text_path, "w", encoding="utf-8") as output_text:
                for segment in sorted_segments:
                    text = speach_to_text_by_ollama(segment.file_path)
                    segment.text = text
                    second = int(segment.start)
                    mirisecond = int((segment.start - second) * 10)
                    output_text.write(f"[{segment.channel}][{second:04d}.{mirisecond}] : {segment.text}\n")
                    output_text.flush()  # Flush after each write to ensure immediate writing to disk

        if args.tts_kotoba:
            pipe = setup_kotoba_whisper()
            output_text_path = args.input_mp3.replace(".mp3", "_transcription_kotoba.txt").replace(".wav", "_transcription.txt")
            with open(output_text_path, "w", encoding="utf-8") as output_text:
                for segment in sorted_segments:
                    text = pipe(str(segment.file_path), chunk_length_s=15, ignore_warning=True)
                    segment.text = text["text"]
                    #output_text.write(f"Segment: {segment.start:.2f}s - {segment.end:.2f}s, Channel: {segment.channel}, Text: {segment.text}\n")
                    second = int(segment.start)
                    mirisecond = int((segment.start - second) * 10)
                    output_text.write(f"[{segment.channel}][{second:04d}.{mirisecond}] : {segment.text}\n")
                    output_text.flush()  # Flush after each write to ensure immediate writing to disk

    for output_path in output_paths:
        print(output_path)



if __name__ == "__main__":
    main()
