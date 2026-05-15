import torch
import json
import re
import os
import gc
import gradio as gr
import peft.utils.save_and_load
from transformers import BitsAndBytesConfig

from src.core.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from src.services.audio_processor import QwenOmniAudioProcessor
from src.repositories.insurance_manager import InsuranceManager
from src.services.memory_service import MultimodalMemoryService

torch.set_num_threads(os.cpu_count())

MODEL_PATH = "./models" 
AUDIO_BASE = "./data"

class QwenOmniAgent:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_manager = InsuranceManager()
        self.omni_processor = QwenOmniAudioProcessor(model_path, AUDIO_BASE)
        
        gc.collect()
        self.model = self._load_model(model_path)
        
        self.memory_service = MultimodalMemoryService(
            session_id="session_gradio_demo",
            user_id="user_khanh_phenikaa",
            model_instance=self.model,
            processor_instance=self.omni_processor.processor
        )

    def _load_model(self, model_path):
        print(f"Loading Base Model Qwen2.5-Omni (Safe Load Mode)...")
        
        load_dtype = torch.bfloat16 if self.device == "cpu" else torch.bfloat16
        
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            device_map={"": "cpu"} if self.device == "cpu" else "auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
        
        if self.device == "cpu":
            model = model.to(torch.float32)

        print("Base model is ready!")
        return model

    def _universal_parser(self, response_text):
        tool_call = None
        match = re.search(r'<tool_call>(.*?)</tool_call>', response_text, re.DOTALL)
        raw_content = match.group(1).strip() if match else response_text
        potential_jsons = re.findall(r'\{.*\}', raw_content, re.DOTALL)
        if potential_jsons:
            try:
                clean_json = potential_jsons[-1].replace("'", '"')
                data = json.loads(clean_json)
                if "name" not in data:
                    if "policy_id" in data: data = {"name": "verify_policy", "arguments": data}
                    elif "hospital_name" in data: data = {"name": "process_medical_claim", "arguments": data}
                tool_call = data
            except: pass
        return tool_call

    def handle_gradio_turn(self, audio_files, chat_history):
        if not audio_files:
            return chat_history, None
        
        if not isinstance(audio_files, list):
            audio_files = [audio_files]

        last_audio_path = audio_files[-1]
        for audio_path in audio_files:
            inputs = self.omni_processor.process_conversation([audio_path])
            with torch.no_grad():
                input_features = inputs['input_features'].to(self.device)
                feature_mask = inputs['feature_attention_mask'].to(self.device)
                audio_outputs = self.model.thinker.get_audio_features(input_features, feature_mask)
                self.memory_service.update_audio_memory(audio_outputs.last_hidden_state)
            del inputs, audio_outputs
            gc.collect()

        history_msgs = []
        try:
            history_msgs = self.memory_service.logic_agent.get_chat_history()
        except:
            history_msgs = self.memory_service.logic_agent.memory.messages if hasattr(self.memory_service.logic_agent, 'memory') else []

        context_text = ""
        for m in history_msgs[-4:]:
            role = "User" if m.role == "user" else "Assistant"
            content = m.content if isinstance(m.content, str) else "Audio interaction"
            context_text += f"{role}: {content}\n"

        print("\n[Thinking] CPU Inference...")
        
        system_with_context = f"History:\n{context_text}\nLatest intent:"
        last_audio_inputs = self.omni_processor.process_conversation([last_audio_path])
        final_inputs = self.omni_processor.processor(
            text=system_with_context, 
            audio=last_audio_inputs['input_features'],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **final_inputs,
                generation_mode="text",
                max_new_tokens=64,
                use_cache=True
            )
        
        resp = self.omni_processor.processor.batch_decode(
            output_ids[:, final_inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0].strip()
        
        tool_to_run = self._universal_parser(resp)
        observation = "None"
        if tool_to_run:
            func_name, args = tool_to_run.get("name"), tool_to_run.get("arguments", {})
            if hasattr(self.db_manager, func_name):
                result = getattr(self.db_manager, func_name)(**args)
                observation = json.dumps(result, ensure_ascii=False)

        final_text = self.memory_service.run_logic_turn(f"Intent: {resp}. Result: {observation}")

        chat_history.append({"role": "user", "content": f"Đã gửi {len(audio_files)} file."})
        chat_history.append({"role": "assistant", "content": final_text})
        
        gc.collect()
        return chat_history, None

    def cleanup(self):
        self.db_manager.close()

def launch_gradio():
    agent = QwenOmniAgent(MODEL_PATH)
    with gr.Blocks(title="Qwen-Omni Agent") as demo:
        gr.Markdown("# Qwen2.5-Omni CPU Agent")
        chatbot = gr.Chatbot(label="Conversation", height=500)
        with gr.Row():
            audio_input = gr.File(label="Upload Audio", file_count="multiple", file_types=["audio"])
            clear_btn = gr.Button("Clear Chat")

        audio_input.change(fn=agent.handle_gradio_turn, inputs=[audio_input, chatbot], outputs=[chatbot, audio_input])
        clear_btn.click(fn=lambda: ([], None), outputs=[chatbot, audio_input])

    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())

if __name__ == "__main__":
    launch_gradio()