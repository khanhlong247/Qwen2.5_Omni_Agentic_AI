from __future__ import annotations
import base64
import copy
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"Aspect ratio too large: {max(height, width) / min(height, width)}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    return pil_image.convert("RGB")

def fetch_image(ele: dict, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    image = ele.get("image", ele.get("image_url"))
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith(("http://", "https://")):
        resp = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(resp.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        _, base64_data = image.split("base64,", 1)
        image_obj = Image.open(BytesIO(base64.b64decode(base64_data)))
    else:
        image_obj = Image.open(image)
    
    image = to_rgb(image_obj)
    width, height = image.size
    resized_h, resized_w = smart_resize(height, width, factor=size_factor)
    return image.resize((resized_w, resized_h))

def smart_nframes(ele: dict, total_frames: int, video_fps: float) -> int:
    fps = ele.get("fps", FPS)
    nframes = total_frames / video_fps * fps
    nframes = min(min(max(nframes, FPS_MIN_FRAMES), FPS_MAX_FRAMES), total_frames)
    return floor_by_factor(int(nframes), FRAME_FACTOR)

def _read_video_torchvision(ele: dict) -> tuple[torch.Tensor, float]:
    video_path = ele["video"].replace("file://", "")
    video, _, info = io.read_video(video_path, pts_unit="sec", output_format="TCHW")
    total_frames, video_fps = video.size(0), info["video_fps"]
    nframes = smart_nframes(ele, total_frames, video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    return video[idx], (nframes / max(total_frames, 1e-6) * video_fps)

def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False):
    video, sample_fps = _read_video_torchvision(ele)
    nframes, _, height, width = video.shape
    resized_h, resized_w = smart_resize(height, width, factor=image_factor, min_pixels=VIDEO_MIN_PIXELS, max_pixels=VIDEO_MAX_PIXELS)
    video = transforms.functional.resize(video, [resized_h, resized_w], interpolation=InterpolationMode.BICUBIC, antialias=True).float()
    return (video, sample_fps) if return_video_sample_fps else video

def extract_vision_info(conversations: list) -> list[dict]:
    vision_infos = []
    for msg in (conversations if isinstance(conversations[0], list) else [conversations]):
        for m in msg:
            if isinstance(m.get("content"), list):
                for ele in m["content"]:
                    if any(k in ele for k in ["image", "image_url", "video"]):
                        vision_infos.append(ele)
    return vision_infos

def process_vision_info(conversations: list, return_video_kwargs: bool = False):
    vision_infos = extract_vision_info(conversations)
    image_inputs, video_inputs, fps_list = [], [], []
    for info in vision_infos:
        if "image" in info or "image_url" in info:
            image_inputs.append(fetch_image(info))
        elif "video" in info:
            v_input, v_fps = fetch_video(info, return_video_sample_fps=True)
            video_inputs.append(v_input)
            fps_list.append(v_fps)
    
    res_images = image_inputs if image_inputs else None
    res_videos = video_inputs if video_inputs else None
    if return_video_kwargs:
        return res_images, res_videos, {'fps': fps_list}
    return res_images, res_videos