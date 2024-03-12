import time, sys
import numpy as np

# https://huggingface.co/docs/transformers/main/model_doc/whisper
# https://github.com/collabora/WhisperSpeech



# # import wave
# # with wave.open('output.wav', 'wb') as wf:
# #     wf.writeframes(stream.read(CHUNK))


# import pyaudio as pa

# RECORD_SECONDS = 5
# CHUNK = 1024
# RATE = 44100
# FORMAT = pa.paInt16
# CHANNELS = 1 if sys.platform == 'darwin' else 2

# audio = pa.PyAudio()
# # test = audio.get_format_from_width(2)
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

# print('* recording')
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     audio_in = stream.read(CHUNK) # bytes
#     stream.write(audio_in)
# print('* done')

# stream.close()
# audio.terminate()




import numpy as np
import re
import threading, queue
from whisperspeech.pipeline import Pipeline
import sounddevice as sd

# uncomment the model you want to use
# model_ref = 'collabora/whisperspeech:s2a-q4-small-en+pl.model'
model_ref = 'collabora/whisperspeech:s2a-q4-tiny-en+pl.model'
# model_ref = 'collabora/whisperspeech:s2a-q4-base-en+pl.model'

pipe = Pipeline(s2a_ref=model_ref)

input_text = """
This. script. processes. a. body. of. text. one. sentence. at. a. time. and. plays. them. consecutively.  This enables the audio playback to begin sooner instead of waiting for the entire body of text to be processed.  The script uses the threading and queue modules that are part of the standard Python library.  It also uses the sound device library, which is fairly reliable across different platforms.  I hope you enjoy, and feel free to modify or distribute at your pleasure.
"""

sentences = re.split(r'[.!?;]+\s*', input_text)

audio_queue = queue.Queue()

def process_text_to_audio(sentences, pipe):
    for sentence in sentences:  # Iterate through each sentence
        if sentence:  # Ensure the sentence is not empty
            audio_tensor = pipe.generate(sentence)  # Generate audio tensor for the sentence
            audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)  # Convert tensor to numpy array, scale, and cast to int16
            if len(audio_np.shape) == 1:  # Check if the numpy array is 1D
                audio_np = np.expand_dims(audio_np, axis=0)  # Add a new dimension to make it 2D
            else:
                audio_np = audio_np.T  # Transpose the numpy array if it's not 1D
            audio_queue.put(audio_np)  # Put the audio numpy array into the queue
    audio_queue.put(None)  # Signal that processing is complete

def play_audio_from_queue(audio_queue):
    while True:  # Loop indefinitely to process audio data
        audio_np = audio_queue.get()  # Retrieve the next audio numpy array from the queue
        if audio_np is None:  # Check if the queue is signaling that processing is complete
            break  # Exit the loop if signal received
        try:
            sd.play(audio_np, samplerate=24000)  # Play the audio numpy array using sounddevice
            sd.wait()  # Wait for the playback to finish before proceeding
        except Exception as e:
            print(f"Error playing audio: {e}")  # Print any errors encountered during playback

processing_thread = threading.Thread(target=process_text_to_audio, args=(sentences, pipe))
playback_thread = threading.Thread(target=play_audio_from_queue, args=(audio_queue,))

processing_thread.start()
playback_thread.start()

processing_thread.join()
playback_thread.join()
