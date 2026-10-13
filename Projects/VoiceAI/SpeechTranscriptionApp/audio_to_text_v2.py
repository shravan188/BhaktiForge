import os
import whisper
import torch

from pydub import AudioSegment

torch.set_num_threads(8)

# Combine it with your filename
INPUT_FILE = r'C:\Users\dell\Documents\Bhakti Projects\audio\wJ8WJJsSaYw.mp3'
OUTPUT_SRT = r"C:\Users\dell\Documents\Bhakti Projects\wJ8WJJsSaYw.srt"

CHUNK_MS = 60 * 1000      # 60 seconds
TEMP_DIR = "temp_chunks"

os.makedirs(TEMP_DIR, exist_ok=True)


model = whisper.load_model("base.en")
print("Model loaded!")

# Load audio
audio = AudioSegment.from_file(INPUT_FILE)

subtitle_index = 1

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"


with open(OUTPUT_SRT, "w", encoding="utf-8") as srt:
    for chunk_start in range(0, len(audio), CHUNK_MS):
        chunk_end = min(chunk_start + CHUNK_MS, len(audio))
        chunk = audio[chunk_start: chunk_end]

        chunk_path = os.path.join(
            TEMP_DIR,
            f"chunk_{chunk_start//1000}.wav"
        )

        chunk.export(chunk_path, format='wav')

        print(f"Transcribing {chunk_start/1000:.0f}s - {chunk_end/1000:.0f}s")

        result = model.transcribe(chunk_path, language="en", fp16=False)

        offset = chunk_start / 1000 # This is because pydub uses milliseconds, while Whisper uses seconds.

        for seg in result['segments']:
            start = seg['start'] + offset
            end = seg['end'] + offset
            text = seg['text'].strip()

            srt.write(f"{subtitle_index}")
            srt.write(
                f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
            )
            srt.write(text + "\n\n")

            subtitle_index += 1


