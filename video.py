import time, sys
import numpy as np
import torch

import cv2
import pyvirtualcam


from diffusers import AutoPipelineForImage2Image

pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True, torch_dtype=torch.float16, variant="fp16", add_watermarker=False)
pipeline.set_progress_bar_config(disable=True)
pipeline = pipeline.to("cuda")

# prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
prompt = "person, virtual reality headset. Bioluminescent, glitch, pixelation, vibrant, AI"
prompt2 = "person, virtual reality headset. Bioluminescent, glitch, pixelation, vibrant, AI"

WIDTH, HEIGHT = 640, 480
fps = 2.0



def captureFrame(cap):
    ret, frame = cap.read()
    if not ret: print("Cannot read frame"); return None
    frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

cap = cv2.VideoCapture(0)
if cap is None or not cap.isOpened(): print("Cannot open camera"); quit(0)
print("width {}   height {}  fps {}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)))
# fps = cap.get(cv2.CAP_PROP_FPS)

# codec = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output/movie.mp4', codec, fps, (WIDTH, HEIGHT))


with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    print(f'Using virtual camera: {cam.device}')

    while True:
        frame = captureFrame(cap)
        if frame is None: break

        frame = np.asarray(frame)/255
        frame = pipeline(prompt=prompt, guidance_scale=2.0, num_inference_steps=16, strength=0.3, image=frame, output_type="np").images[0]
        frame = np.asarray(frame*256, dtype=np.uint8)

        # out.write(frame)
        # cv2.imshow('frame', frame)
        cam.send(frame)
        if cv2.waitKey(1) == ord('q'): break

cap.release()
# out.release()
cv2.destroyAllWindows()




# import colorsys
# import pyvirtualcam

# with pyvirtualcam.Camera(width=640, height=480, fps=2) as cam:
#     print(f'Using virtual camera: {cam.device}')
#     frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
#     while True:
#         h, s, v = (cam.frames_sent % 100) / 100, 1.0, 1.0
#         r, g, b = colorsys.hsv_to_rgb(h, s, v)
#         frame[:] = (r * 255, g * 255, b * 255)
#         cam.send(frame)
#         cam.sleep_until_next_frame()




# import ffmpeg

# # Set the RTMP stream key and URL
# stream_key = 'live_1051297811_NN07Y70eFSW4yM5XBEIUbHixwtzruR'
# rtmp_url = f'rtmp://live.twitch.tv/app/{stream_key}' # ?bandwidthtest=true

# # Open the default camera (usually 0)
# cap = cv2.VideoCapture(0)

# # Get the video properties
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Create a video writer object
# process = (
#     ffmpeg
#     .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
#     .output(rtmp_url, format='flv', vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast', tune='zerolatency',
#             r=fps, g=60, b='2M', maxrate='2M', bufsize='2M', crf=23)
#     .run_async(pipe_stdin=True)
# )

# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Write the frame to the video writer
#     process.stdin.write(frame.tobytes())

#     if cv2.waitKey(1) == ord('q'): break

# # Release the camera and close the video writer
# cap.release()
# process.stdin.close()
# process.wait()




# print(cv2.__version__)
# info = cv2.getBuildInformation()
# video, parallel = info.index('Video'), info.index('Parallel')
# print(info[video:parallel])


# def captureFrame(cap):
#     ret, frame = cap.read()
#     if not ret: print("Cannot read frame"); return None
#     frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return frame

# cap = cv2.VideoCapture(0)
# if cap is None or not cap.isOpened(): print("Cannot open camera"); quit(0)
# print("width {}   height {}  fps {}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)))
# # fps = cap.get(cv2.CAP_PROP_FPS)


# # codec = cv2.VideoWriter_fourcc(*'mp4v')
# # out = cv2.VideoWriter('output/movie.mp4', codec, fps, (WIDTH, HEIGHT))


# # Set the RTMP stream key and URL
# stream_key = 'live_1051297811_NN07Y70eFSW4yM5XBEIUbHixwtzruR'
# rtmp_url = f'rtmp://slc.contribute.live-video.net/app/{stream_key}?bandwidthtest=true'

# codec = cv2.VideoWriter_fourcc(*'flv1')
# out = cv2.VideoWriter(rtmp_url, codec, fps, (WIDTH, HEIGHT))


# while True:
#     frame = captureFrame(cap)
#     if frame is None: break

#     # frame = np.asarray(frame)/255
#     # frame = pipeline(prompt=prompt, guidance_scale=1.0, num_inference_steps=4, strength=0.3, image=frame, output_type="np").images[0]
#     # frame = np.asarray(frame*256, dtype=np.uint8)

#     out.write(frame)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) == ord('q'): break

# cap.release()
# out.release()
# cv2.destroyAllWindows()





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
#     s1 = cv2.absdiff(I1, I2) #|I1 - I2|
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
#     mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
#     mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
#     mu1_2 = mu1 * mu1
#     mu2_2 = mu2 * mu2
#     mu1_mu2 = mu1 * mu2
#     sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
#     sigma1_2 -= mu1_2
#     sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
#     sigma2_2 -= mu2_2
#     sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
#     sigma12 -= mu1_mu2
#     t1 = 2 * mu1_mu2 + C1
#     t2 = 2 * sigma12 + C2
#     t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
#     t1 = mu1_2 + mu2_2 + C1
#     t2 = sigma1_2 + sigma2_2 + C2
#     t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
#     ssim_map = cv2.divide(t3, t1)    # ssim_map =  t3./t1;
#     mssim = cv2.mean(ssim_map)       # mssim = average of ssim map
#     return mssim

# def captureFrame(cap):
#     ret, frame = cap.read()
#     if not ret: print("Cannot read frame"); return None
#     frame = cv2.resize(frame, (256,256), interpolation=cv2.INTER_LINEAR)
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return frame

# cap = cv2.VideoCapture(0)
# if cap is None or not cap.isOpened(): print("Cannot open camera"); quit(0)
# print("width {}   height {}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# frame_old = captureFrame(cap) # frame = np.uint8 (480, 640, 3)
# if frame_old is None: quit(0)

# while True:
#     frame = captureFrame(cap)
#     if frame is None: break

#     movement = getPSNR(frame_old, frame)
#     if movement > 0 and movement < 20.0: cv2.imshow('frame', frame)
#     # movement = getMSSISM(frame_old, frame); movement = movement[0]
#     # if movement < 1.0 and movement < 0.86: cv2.imshow('frame', frame)
#     frame_old = frame
#     # print(movement)

#     if cv2.waitKey(1) == ord('q'): break
#     # time.sleep(0.001)
# cap.release()
# cv2.destroyAllWindows()
