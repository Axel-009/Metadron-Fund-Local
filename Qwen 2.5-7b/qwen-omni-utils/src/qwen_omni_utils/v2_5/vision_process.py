from __future__ import annotations
import base64, copy, logging, math, os, sys, time, warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional
import requests, torch, torchvision
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

def round_by_factor(number, factor): return round(number / factor) * factor
def ceil_by_factor(number, factor): return math.ceil(number / factor) * factor
def floor_by_factor(number, factor): return math.floor(number / factor) * factor

def smart_resize(height, width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}")
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

def to_rgb(pil_image):
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    return pil_image.convert("RGB")

def fetch_image(ele, size_factor=IMAGE_FACTOR):
    image = ele.get("image", ele.get("image_url"))
    image_obj = None
    if isinstance(image, Image.Image): image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio: image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"): image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            with BytesIO(base64.b64decode(base64_data)) as bio: image_obj = copy.deepcopy(Image.open(bio))
    else: image_obj = Image.open(image)
    if image_obj is None: raise ValueError(f"Unrecognized image input: {image}")
    image = to_rgb(image_obj)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(ele["resized_height"], ele["resized_width"], factor=size_factor)
    else:
        width, height = image.size
        resized_height, resized_width = smart_resize(height, width, factor=size_factor, min_pixels=ele.get("min_pixels", MIN_PIXELS), max_pixels=ele.get("max_pixels", MAX_PIXELS))
    return image.resize((resized_width, resized_height))

def smart_nframes(ele, total_frames, video_fps):
    assert not ("fps" in ele and "nframes" in ele)
    if "nframes" in ele: nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should in [{FRAME_FACTOR}, {total_frames}], got {nframes}")
    return nframes

def _read_video_torchvision(ele):
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path")
        if "file://" in video_path: video_path = video_path[7:]
    video, audio, info = io.read_video(video_path, start_pts=ele.get("video_start", 0.0), end_pts=ele.get("video_end", None), pts_unit="sec", output_format="TCHW")
    total_frames, video_fps = video.size(0), info["video_fps"]
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    return video[idx], nframes / max(total_frames, 1e-6) * video_fps

def is_decord_available():
    import importlib.util
    return importlib.util.find_spec("decord") is not None

def calculate_video_frame_range(ele, total_frames, video_fps):
    if video_fps <= 0: raise ValueError("video_fps must be positive")
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None: return 0, total_frames - 1, total_frames
    max_duration = total_frames / video_fps
    start_frame = math.ceil(max(0.0, min(video_start, max_duration)) * video_fps) if video_start is not None else 0
    end_frame = min(math.floor(max(0.0, min(video_end, max_duration)) * video_fps), total_frames - 1) if video_end is not None else total_frames - 1
    if start_frame >= end_frame: raise ValueError(f"Invalid time range: start {start_frame} >= end {end_frame}")
    return start_frame, end_frame, end_frame - start_frame + 1

def _read_video_decord(ele):
    import decord
    vr = decord.VideoReader(ele["video"])
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(ele, total_frames, video_fps)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = torch.tensor(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
    return video, nframes / max(total_frames, 1e-6) * video_fps

def is_torchcodec_available():
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec") is None: return False
        from torchcodec.decoders import VideoDecoder
        return True
    except: return False

def _read_video_torchcodec(ele):
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(ele["video"], num_ffmpeg_threads=int(os.environ.get("TORCHCODEC_NUM_THREADS", 8)))
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(ele, total_frames, video_fps)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    return decoder.get_frames_at(indices=idx).data, nframes / max(total_frames, 1e-6) * video_fps

VIDEO_READER_BACKENDS = {"decord": _read_video_decord, "torchvision": _read_video_torchvision, "torchcodec": _read_video_torchcodec}
FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)

@lru_cache(maxsize=1)
def get_video_reader_backend():
    if FORCE_QWENVL_VIDEO_READER is not None: return FORCE_QWENVL_VIDEO_READER
    if is_torchcodec_available(): return "torchcodec"
    if is_decord_available(): return "decord"
    return "torchvision"

def fetch_video(ele, image_factor=IMAGE_FACTOR, return_video_sample_fps=False):
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try: video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"{video_reader_backend} error, using torchvision: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = min(ele.get("max_pixels", max_pixels), max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(ele["resized_height"], ele["resized_width"], factor=image_factor)
        else:
            resized_height, resized_width = smart_resize(height, width, factor=image_factor, min_pixels=min_pixels, max_pixels=max_pixels)
        video = transforms.functional.resize(video, [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True).float()
        return (video, sample_fps) if return_video_sample_fps else video
    else:
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [fetch_image({"image": v, **process_info}, size_factor=image_factor) for v in ele["video"]]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes: images.extend([images[-1]] * (nframes - len(images)))
        return (images, process_info.pop("fps", 2.0)) if return_video_sample_fps else images

def extract_vision_info(conversations):
    vision_infos = []
    if isinstance(conversations[0], dict): conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if "image" in ele or "image_url" in ele or "video" in ele or ele.get("type", "") in ("image", "image_url", "video"):
                        vision_infos.append(ele)
    return vision_infos

def process_vision_info(conversations, return_video_kwargs=False):
    vision_infos = extract_vision_info(conversations)
    image_inputs, video_inputs, video_sample_fps_list = [], [], []
    for vi in vision_infos:
        if "image" in vi or "image_url" in vi: image_inputs.append(fetch_image(vi))
        elif "video" in vi:
            video_input, video_sample_fps = fetch_video(vi, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else: raise ValueError("image, image_url or video should in content.")
    if not image_inputs: image_inputs = None
    if not video_inputs: video_inputs = None
    if return_video_kwargs: return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs
