import argparse
import os

import numpy as np
import torch
import pprint

def setup_environment():
    """Set up the environment for the script."""
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


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Ollama API.")
    parser.add_argument("file", type=str, help="Path to the audio file to transcribe.")
    args = parser.parse_args()

    audio_file = args.file
    if not os.path.isfile(audio_file):
        print(f"Error: File '{audio_file}' does not exist.")
        return
    
    pipe = setup_environment()
    result = pipe(audio_file, chunk_length_s=15)
    print("Transcribed Text:")
    pprint.pprint(result)

    output_file = os.path.splitext(audio_file)[0] + "_transcription.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])

if __name__ == "__main__":
    main()

