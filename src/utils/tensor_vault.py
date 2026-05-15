import torch
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class AudioTensorVault:
    """
    Quản lý lưu trữ Short-term memory cho Audio Tensors (Feature Maps).
    Tuân thủ Temporal Window (5 phút).
    """
    def __init__(self, window_minutes: int = 5):
        self.window_seconds = window_minutes * 60
        # Cấu trúc: { session_id: [ {"timestamp": float, "tensor": torch.Tensor} ] }
        self._vault: Dict[str, List[dict]] = {}

    def add_feature_map(self, session_id: str, tensor: torch.Tensor):
        """Lưu trữ feature map mới vào vault, đưa về CPU để tiết kiệm VRAM."""
        if session_id not in self._vault:
            self._vault[session_id] = []
        
        entry = {
            "timestamp": time.time(),
            "tensor": tensor.detach().to("cpu")
        }
        self._vault[session_id].append(entry)
        self._cleanup(session_id)

    def _cleanup(self, session_id: str):
        """Xóa các tensor đã cũ hơn window_seconds."""
        now = time.time()
        self._vault[session_id] = [
            item for i, item in enumerate(self._vault[session_id])
            if (now - item["timestamp"]) <= self.window_seconds
        ]

    def get_combined_context(self, session_id: str, device: str = "cpu") -> Optional[torch.Tensor]:
        """Nối tất cả các tensor trong session hiện tại thành một context duy nhất."""
        if session_id not in self._vault or not self._vault[session_id]:
            return None
        
        tensors = [item["tensor"] for item in self._vault[session_id]]
        # Nối theo chiều Sequence Length (thường là dim 1 trong Qwen)
        combined = torch.cat(tensors, dim=1) 
        return combined.to(device)

    def clear_session(self, session_id: str):
        if session_id in self._vault:
            del self._vault[session_id]