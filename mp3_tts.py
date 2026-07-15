import argparse
import struct
import subprocess
import wave
from pathlib import Path


def get_audio_channel_count(mp3_path: str) -> int:
    """Return the number of audio channels in the MP3."""
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
        mp3_path,
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


def convert_mp3_to_wav(mp3_path: str, output_dir: str | None = None, sample_rate: int = 16000) -> list[Path]:
    """Convert an MP3 into one or more mono WAV files at 16-bit/16kHz."""
    input_path = Path(mp3_path)
    output_folder = Path(output_dir) if output_dir else input_path.parent / f"{input_path.stem}_wav"
    output_folder.mkdir(parents=True, exist_ok=True)

    channel_count = get_audio_channel_count(str(input_path))
    if channel_count == 2:
        pcm_bytes = decode_mp3_to_pcm_bytes(str(input_path), sample_rate=sample_rate, channel_count=2)
        left_bytes, right_bytes = split_stereo_pcm(pcm_bytes)
        left_path = output_folder / f"{input_path.stem}_L.wav"
        right_path = output_folder / f"{input_path.stem}_R.wav"
        write_wav_file(left_path, left_bytes, sample_rate=sample_rate, channels=1)
        write_wav_file(right_path, right_bytes, sample_rate=sample_rate, channels=1)
        return [left_path, right_path]

    pcm_bytes = decode_mp3_to_pcm_bytes(str(input_path), sample_rate=sample_rate, channel_count=1)
    output_path = output_folder / f"{input_path.stem}.wav"
    write_wav_file(output_path, pcm_bytes, sample_rate=sample_rate, channels=1)
    return [output_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MP3 audio to 16-bit 16kHz WAV files")
    parser.add_argument("input_mp3", help="Path to the input MP3 file")
    parser.add_argument("--output-dir", help="Directory to save the output WAV file(s)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate in Hz")
    args = parser.parse_args()

    output_paths = convert_mp3_to_wav(args.input_mp3, output_dir=args.output_dir, sample_rate=args.sample_rate)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
