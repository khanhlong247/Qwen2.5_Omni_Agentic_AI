import base64
from io import BytesIO
import audioread
import av
import librosa
import numpy as np

SAMPLE_RATE = 24000

def _check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        return len(audio_streams) > 0
    except:
        return False

def process_audio_info(conversations, use_audio_in_video=False):
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
        
    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
                
            for ele in content:
                data = None
                audio_start = ele.get("audio_start", ele.get("video_start", 0.0))
                audio_end = ele.get("audio_end", ele.get("video_end", None))
                
                if ele["type"] == "audio":
                    path = ele.get("audio", ele.get("audio_url"))
                    if isinstance(path, np.ndarray):
                        if path.ndim > 1: raise ValueError("Support only mono audio")
                        start_idx = int(SAMPLE_RATE * audio_start)
                        end_idx = None if audio_end is None else int(SAMPLE_RATE * audio_end)
                        audios.append(path[start_idx:end_idx])
                        continue
                    elif path.startswith("data:audio"):
                        _, base64_data = path.split("base64,", 1)
                        data = BytesIO(base64.b64decode(base64_data))
                    elif path.startswith(("http://", "https://")):
                        data = audioread.ffdec.FFmpegAudioFile(path)
                    else:
                        data = path.replace("file://", "")
                
                elif use_audio_in_video and ele["type"] == "video":
                    path = ele.get("video", ele.get("video_url")).replace("file://", "")
                    if _check_if_video_has_audio(path):
                        data = path
                    else:
                        continue
                
                else:
                    continue

                if data:
                    duration = (audio_end - audio_start) if audio_end is not None else None
                    y, _ = librosa.load(data, sr=SAMPLE_RATE, offset=audio_start, duration=duration)
                    audios.append(y)
                    
    return audios if len(audios) > 0 else None