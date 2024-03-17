import time, sys
import threading, queue
import numpy as np

import pynput.keyboard as kb

# https://pynput.readthedocs.io/en/latest/keyboard.html
# https://huggingface.co/docs/transformers/main/model_doc/llava
# https://huggingface.co/llava-hf
# https://huggingface.co/visheratin/MC-LLaVA-3b


kbc = kb.Controller()
q = queue.SimpleQueue()



import torch
import requests
from PIL import Image
from transformers import AutoModel, AutoProcessor

# model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", torch_dtype=torch.float16, trust_remote_code=True)
# model = AutoModel.from_pretrained("air-foundry/MC-LLaVA-3b-live", torch_dtype=torch.float16, trust_remote_code=True)
model = AutoModel.from_pretrained("C:/Users/Wil/GitHub/MC-LLaVA-3b-live", torch_dtype=torch.float16, trust_remote_code=True)
model = model.to("cuda")
# processor = AutoProcessor.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)
# processor = AutoProcessor.from_pretrained("air-foundry/MC-LLaVA-3b-live", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("C:/Users/Wil/GitHub/MC-LLaVA-3b-live", trust_remote_code=True)

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# prompt = "What's the content of the image?"
# prompt = f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant"
prompt = "I'm looking for a car "

with torch.inference_mode():
    # inputs = processor(prompt, [image], model, max_crops=100, num_tokens=728)
    inputs = processor(prompt, None, model, max_crops=100, num_tokens=728)
    output = model.generate(**inputs, max_new_tokens=200, use_cache=True, do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.eos_token_id)

result = processor.tokenizer.decode(output[0]).replace(prompt, "").replace("<|im_end|>", "")
print(result)




# import requests
# from PIL import Image
# from transformers import AutoProcessor, LlavaForConditionalGeneration

# model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
# # model = model.to("cuda")
# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
# inputs = processor(text=prompt, images=image, return_tensors="pt")

# # Generate
# generate_ids = model.generate(**inputs, max_length=30)
# result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(result)




# def on_press(key):
#     if hasattr(key,'char') and ord(key.char) > 32 and ord(key.char) < 127:
#         q.put(key.char)
#     if key == kb.Key.space:
#         q.put(' ')

# listener = kb.Listener(on_press=on_press)
# listener.start()

# w = ''
# while True:
#     c = q.get()
#     if c in " .!?;":
#         time.sleep(2)
#         kbc.type(w+' ')
#         w = ''
#     else:
#         w += c




# time.sleep(5)
# # Press and release space
# kbc.press(kb.Key.space)
# kbc.release(kb.Key.space)

# # Type a lower case A; this will work even if no key on the
# # physical keyboard is labelled 'A'
# kbc.press('a')
# kbc.release('a')

# # Type two upper case As
# kbc.press('A')
# kbc.release('A')
# with kbc.pressed(kb.Key.shift):
#     kbc.press('a')
#     kbc.release('a')

# # Type 'Hello World' using the shortcut type method
# kbc.type('Hello World')



# def on_press(key):
#     try:
#         print('alphanumeric key {0} pressed'.format(key.char))
#     except AttributeError:
#         print('special key {0} pressed'.format(key))

# def on_release(key):
#     print('{0} released'.format(key))
#     if hasattr(key,'char'): kbc.press(key.char)
#     if key == kb.Key.space: kbc.press(key)
#     if key == kb.Key.esc:
#         # Stop listener
#         return False

# # Collect events until released
# with kb.Listener(on_press=on_press, on_release=on_release) as listener:
#     listener.join()

# # # ...or, in a non-blocking fashion:
# # listener = kb.Listener(on_press=on_press, on_release=on_release)
# # listener.start()



# q = queue.Queue()

# def worker():
#     while True:
#         item = q.get()
#         print(f'Working on {item}')
#         time.sleep(2)
#         print(f'Finished {item}')
#         q.task_done()

# # Turn-on the worker thread.
# threading.Thread(target=worker, daemon=True).start()

# # Send thirty task requests to the worker.
# for item in range(5):
#     q.put(item)

# # Block until all tasks are done.
# q.join()
# print('All work completed')
