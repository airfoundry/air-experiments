import time, sys
import numpy as np
import torch

import cv2 as cv

from diffusers import AutoPipelineForImage2Image


pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True, torch_dtype=torch.float16, variant="fp16", add_watermarker=False)
pipeline.set_progress_bar_config(disable=True)
pipeline = pipeline.to("cuda")

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
prompt2 = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"



def captureFrame(cap):
    ret, frame = cap.read()
    if not ret: print("Cannot read frame"); return None
    frame = cv.resize(frame, (256,256), interpolation=cv.INTER_LINEAR)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return frame

cap = cv.VideoCapture(0)
if cap is None or not cap.isOpened(): print("Cannot open camera"); quit(0)
print("width {}   height {}".format(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)))


while True:
    frame = captureFrame(cap)
    if frame is None: break

    frame = np.asarray(frame)/255
    frame = pipeline(prompt=prompt, guidance_scale=1.0, num_inference_steps=4, strength=0.3, image=frame, output_type="np").images[0]
    frame = np.asarray(frame*256, dtype=np.uint8)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'): break

cap.release()
cv.destroyAllWindows()





# # https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl_turbo

# # from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
# from diffusers import AutoPipelineForText2Image
# from diffusers import AutoPipelineForImage2Image
# from diffusers.utils import load_image
# from PIL import Image

# # pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True, torch_dtype=torch.float16, variant="fp16", add_watermarker=False)
# pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True, torch_dtype=torch.float16, variant="fp16", add_watermarker=False)
# # pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
# pipeline.set_progress_bar_config(disable=True)
# pipeline = pipeline.to("cuda")
# # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

# init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
# init_image = init_image.resize((256, 256))
# init_image = np.asarray(init_image)/255
# # test = pipeline.image_processor.preprocess(init_image)

# prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
# prompt2 = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
# t1_start = time.perf_counter_ns()
# # image = pipeline(prompt=prompt, prompt_2=prompt2, guidance_scale=0.0, num_inference_steps=1, output_type="np").images[0]
# image = pipeline(prompt=prompt, guidance_scale=0.0, num_inference_steps=4, strength=0.5, image=init_image, output_type="np").images[0]
# total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds
# print(total_time)
# image = np.asarray(image*256,dtype=np.uint8)
# image = Image.fromarray(image, 'RGB')
# image.show()





# def getPSNR(I1, I2):
#     s1 = cv.absdiff(I1, I2) #|I1 - I2|
#     s1 = np.float32(s1)     # cannot make a square on 8 bits
#     s1 = s1 * s1            # |I1 - I2|^2
#     sse = s1.sum()          # sum elements per channel
#     if sse <= 1e-10:        # sum channels
#         return 0            # for small values return zero
#     else:
#         shape = I1.shape
#         mse = 1.0 * sse / np.prod(shape)
#         psnr = 10.0 * np.log10((255 * 255) / mse)
#         return psnr

# def getMSSISM(i1, i2):
#     C1 = 6.5025
#     C2 = 58.5225
#     # INITS
#     I1 = np.float32(i1) # cannot calculate on one byte large values
#     I2 = np.float32(i2)
#     I2_2 = I2 * I2 # I2^2
#     I1_2 = I1 * I1 # I1^2
#     I1_I2 = I1 * I2 # I1 * I2
#     # END INITS
#     # PRELIMINARY COMPUTING
#     mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
#     mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
#     mu1_2 = mu1 * mu1
#     mu2_2 = mu2 * mu2
#     mu1_mu2 = mu1 * mu2
#     sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
#     sigma1_2 -= mu1_2
#     sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
#     sigma2_2 -= mu2_2
#     sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
#     sigma12 -= mu1_mu2
#     t1 = 2 * mu1_mu2 + C1
#     t2 = 2 * sigma12 + C2
#     t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
#     t1 = mu1_2 + mu2_2 + C1
#     t2 = sigma1_2 + sigma2_2 + C2
#     t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
#     ssim_map = cv.divide(t3, t1)    # ssim_map =  t3./t1;
#     mssim = cv.mean(ssim_map)       # mssim = average of ssim map
#     return mssim

# def captureFrame(cap):
#     ret, frame = cap.read()
#     if not ret: print("Cannot read frame"); return None
#     frame = cv.resize(frame, (256,256), interpolation=cv.INTER_LINEAR)
#     # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     return frame

# cap = cv.VideoCapture(0)
# if cap is None or not cap.isOpened(): print("Cannot open camera"); quit(0)
# print("width {}   height {}".format(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

# frame_old = captureFrame(cap) # frame = np.uint8 (480, 640, 3)
# if frame_old is None: quit(0)

# while True:
#     frame = captureFrame(cap)
#     if frame is None: break

#     movement = getPSNR(frame_old, frame)
#     if movement > 0 and movement < 20.0: cv.imshow('frame', frame)
#     # movement = getMSSISM(frame_old, frame); movement = movement[0]
#     # if movement < 1.0 and movement < 0.86: cv.imshow('frame', frame)
#     frame_old = frame
#     # print(movement)

#     if cv.waitKey(1) == ord('q'): break
#     # time.sleep(0.001)
# cap.release()
# cv.destroyAllWindows()
