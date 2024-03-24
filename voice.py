import time, sys
import threading, queue
import numpy as np

import sounddevice as sd

# https://huggingface.co/docs/transformers/main/model_doc/whisper
# https://github.com/collabora/WhisperSpeech
# https://vb-audio.com/Cable/


RATE = 44100 # 44100 16000




# Set the desired audio parameters
duration = 2.0  # Duration of the recording in seconds
sample_rate = 16000  # Sample rate of the audio (44100,16000)
channels = 1  # Number of audio channels (e.g., 1 for mono, 2 for stereo)

# def audio_callback(indata, frames, time, status):
#     if status:
#         print(status)
#     # Do something with the audio data (e.g., save it, process it, etc.)
#     print("Audio data received:", indata.shape)

# # Start the audio recording
# with sd.InputStream(callback=audio_callback, channels=channels, samplerate=sample_rate):
#     print("Recording audio for", duration, "seconds...")
#     sd.sleep(int(duration * 1000))
# print("Audio recording finished.")

# Start the audio recording
with sd.InputStream(channels=channels, samplerate=sample_rate) as stream:
    print("Recording audio for", duration, "seconds...")
    # Read the audio data from the stream
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    # Wait for the recording to finish
    sd.wait()
    # Process the audio data
    print("Audio data shape:", audio_data.shape)
    # Do something with the audio data (e.g., save it, process it, etc.)
print("Audio recording finished.")

audio_data = np.squeeze(audio_data, -1)

# # Set the virtual microphone as the default output device
# sd.default.device = 'CABLE Input (VB-Audio Virtual Cable)'

# # Generate some audio data (replace with your own audio data)
# duration = 5  # duration of the audio in seconds
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# audio_data = np.sin(2 * np.pi * 400 * t)  # generate a 440 Hz sine wave

# # Play the audio data through the virtual microphone
# sd.play(audio_data, sample_rate)
# sd.wait()


from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Use the model and processor to transcribe the audio:
input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features

# Generate token ids
predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription[0])


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




# import re
# from whisperspeech.pipeline import Pipeline

# # uncomment the model you want to use
# # model_ref = 'collabora/whisperspeech:s2a-q4-small-en+pl.model'
# model_ref = 'collabora/whisperspeech:s2a-q4-tiny-en+pl.model'
# # model_ref = 'collabora/whisperspeech:s2a-q4-base-en+pl.model'

# pipe = Pipeline(s2a_ref=model_ref)

# input_text = """
# This. script. processes. a. body. of. text. one. sentence. at. a. time. and. plays. them. consecutively.  This enables the audio playback to begin sooner instead of waiting for the entire body of text to be processed.  The script uses the threading and queue modules that are part of the standard Python library.  It also uses the sound device library, which is fairly reliable across different platforms.  I hope you enjoy, and feel free to modify or distribute at your pleasure.
# """

# sentences = re.split(r'[.!?;]+\s*', input_text)

# audio_queue = queue.Queue()

# def process_text_to_audio(sentences, pipe):
#     for sentence in sentences:  # Iterate through each sentence
#         if sentence:  # Ensure the sentence is not empty
#             audio_tensor = pipe.generate(sentence)  # Generate audio tensor for the sentence
#             audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)  # Convert tensor to numpy array, scale, and cast to int16
#             if len(audio_np.shape) == 1:  # Check if the numpy array is 1D
#                 audio_np = np.expand_dims(audio_np, axis=0)  # Add a new dimension to make it 2D
#             else:
#                 audio_np = audio_np.T  # Transpose the numpy array if it's not 1D
#             audio_queue.put(audio_np)  # Put the audio numpy array into the queue
#     audio_queue.put(None)  # Signal that processing is complete

# def play_audio_from_queue(audio_queue):
#     while True:  # Loop indefinitely to process audio data
#         audio_np = audio_queue.get()  # Retrieve the next audio numpy array from the queue
#         if audio_np is None:  # Check if the queue is signaling that processing is complete
#             break  # Exit the loop if signal received
#         try:
#             sd.play(audio_np, samplerate=24000)  # Play the audio numpy array using sounddevice
#             sd.wait()  # Wait for the playback to finish before proceeding
#         except Exception as e:
#             print(f"Error playing audio: {e}")  # Print any errors encountered during playback

# processing_thread = threading.Thread(target=process_text_to_audio, args=(sentences, pipe))
# playback_thread = threading.Thread(target=play_audio_from_queue, args=(audio_queue,))

# processing_thread.start()
# playback_thread.start()

# processing_thread.join()
# playback_thread.join()
