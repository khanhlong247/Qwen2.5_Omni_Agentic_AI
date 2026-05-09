import torch
import os
from transformers import AutoProcessor
from src.utils.qwen_omni_utils.v2_5 import process_mm_info
from src.schemas.tools_schema import InsuranceToolSchema

class QwenOmniAudioProcessor:
    def __init__(self, model_path: str, audio_base_path: str):
        self.model_path = model_path
        self.audio_base_path = audio_base_path
        
        print(f"Loading Processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.schema_manager = InsuranceToolSchema()

    def _get_system_instruction(self) -> str:
        tools_prompt = self.schema_manager.get_qwen_xml_prompt()
        return (
            f"You are a professional Insurance AI Agent. {tools_prompt}\n\n"
            "### CORE OPERATING RULES:\n"
            "1. ANALYZE: Listen to all audio inputs to identify the user's intent.\n"
            "2. MATCH: Select the MOST appropriate tool from the <tools> list.\n"
            "3. EXECUTE: Output ONLY the tool call using format: <tool_call>{'name': '...', 'arguments': {...}}</tool_call>\n"
            "4. EXTRACTION: Extract required parameters directly from the audio.\n"
            "5. CONSTRAINT: Do not explain. Do not engage in small talk."
        )

    def process_conversation(self, audio_filenames: list, save_pt: str = "preprocessed_inputs.pt"):
        """
        Thực hiện toàn bộ quy trình tiền xử lý đa phương thức.
        """
        user_content = [
            {"type": "audio", "audio": os.path.join(self.audio_base_path, f)} 
            for f in audio_filenames
        ]
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._get_system_instruction()}]
            },
            {
                "role": "user", 
                "content": user_content
            }
        ]

        print("--- PHASE 1: TEXT TEMPLATE ---")
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        print("--- PHASE 2: AUDIO EXTRACTION ---")
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        print("--- PHASE 3: TENSOR CREATION ---")
        inputs = self.processor(
            text=text_prompt, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt"
        )

        if save_pt:
            torch.save(inputs, save_pt)
            print(f"Tensors saved successfully to {save_pt}")

        return inputs