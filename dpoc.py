import whisperx
import gc
from moviepy.editor import *
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--input", help="first number")
parser.add_argument("--searchword", help="second number")

args = parser.parse_args()

location = args.input
name = args.searchword



device = "cuda"
audio_file = location
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
# print(result)
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print(result["segments"]) # after alignment

# name="Elon"
from moviepy.editor import *
clip = VideoFileClip(location)

for i in result["segments"]:
  text_output=i['text']
  # print(text_output)
  lower_input_string = text_output.lower()
  lower_search_word = name.lower()

  if lower_search_word in lower_input_string:
    start_time=i['start']
    end_time=i['end']
    print(start_time,end_time)
    clip1 = clip.subclip(start_time,end_time)
    clip1.write_videofile('elon'+str(start_time)+'.mp4',codec='libx264')
  else:
    pass