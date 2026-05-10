import torch
import json
import re
import os
import peft.utils.save_and_load
from transformers import BitsAndBytesConfig

from src.core.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from src.services.audio_processor import QwenOmniAudioProcessor
from src.repositories.insurance_manager import InsuranceManager

# --- CONFIGURATION ---
MODEL_PATH = "./models" 
AUDIO_BASE = "./data"
OUTPUT_BASE = "./output"

class QwenOmniAgent:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.db_manager = InsuranceManager()
        self.omni_processor = QwenOmniAudioProcessor(model_path, AUDIO_BASE)
        
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        print(f"Loading Base Model Qwen2.5-Omni...")
        
        bnb_config = None
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        print("Base model is ready!")
        return model

    def _patch_peft(self):
        if not hasattr(peft.utils.save_and_load, "_maybe_shard_state_dict_for_tp"):
            peft.utils.save_and_load._maybe_shard_state_dict_for_tp = lambda state_dict, *args, **kwargs: state_dict
            print("PEFT Monkey Patch applied.")

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

    def run_workflow(self, audio_files):
        inputs = self.omni_processor.process_conversation(audio_files)
        inputs = {k: v.to(self.device).to(torch.bfloat16) if v.dtype == torch.float32 else v.to(self.device) 
                  for k, v in inputs.items()}

        # Suy luận lượt 1 (Gọi Tool)
        print("\n[Thinking] Analyzing audio for intent...")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_mode="text",
                thinker_max_new_tokens=512,
                do_sample=True,
                temperature=0.1
            )
        
        input_len = inputs['input_ids'].shape[1]
        full_response = self.omni_processor.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        
        # Trích xuất và Thực thi Tool
        tool_to_run = self._universal_parser(full_response)
        observation = None

        if tool_to_run:
            func_name = tool_to_run.get("name")
            args = tool_to_run.get("arguments", {})
            print(f"CALLING TOOL: {func_name} with {args}")
            
            if hasattr(self.db_manager, func_name):
                method = getattr(self.db_manager, func_name)
                result = method(**args)
                observation = json.dumps(result, ensure_ascii=False)
                print(f"DB RESULT: {observation}")

        # Suy luận lượt 2 (Phản hồi tự nhiên)
        if observation:
            print("\n[Talking] Starting generation...")
            messages = [
                {"role": "assistant", "content": full_response},
                {"role": "user", "content": f"Kết quả từ hệ thống: {observation}"}
            ]
            
            final_prompt = self.omni_processor.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            final_inputs = self.omni_processor.processor(text=final_prompt, return_tensors="pt").to(self.device)
            final_inputs = {k: v.to(torch.float32) if v.dtype == torch.float32 else v for k, v in final_inputs.items()}

            with torch.no_grad():
                print("AI is 'thinking' of the words...")
                output_text_ids = self.model.generate(**final_inputs, generation_mode="text", max_new_tokens=128)
                input_len = final_inputs['input_ids'].shape[1]
                final_text = self.omni_processor.processor.batch_decode(output_text_ids[:, input_len:], skip_special_tokens=True)[0].strip()
                
                print("\n" + "═"*50)
                print(f"AGENT TEXT: {final_text}")
                print("═"*50)

                outputs = self.model.generate(
                    **final_inputs, 
                    generation_mode="audio", 
                    max_new_tokens=15,
                    token2wav_num_steps=5, 
                    do_sample=False        
                )
            
            print("\n--- TUPLE INSPECTION START ---")
            print(f"Output type: {type(outputs)}")
            
            if isinstance(outputs, tuple):
                print(f"Tuple length: {len(outputs)}")
                for i, element in enumerate(outputs):
                    print(f"  - Element [{i}]:")
                    print(f"    Type: {type(element)}")
                    if torch.is_tensor(element):
                        print(f"    Shape: {element.shape}")
                        print(f"    Dtype: {element.dtype}")
                        print(f"    Device: {element.device}")
                    elif isinstance(element, (list, tuple)):
                        print(f"    Length: {len(element)}")
                        if len(element) > 0:
                            print(f"    Type of first item: {type(element[0])}")
                    else:
                        print(f"    Value: {element}")
            else:
                print("Output is NOT a tuple. It might be a ModelOutput object.")
                for attr in ['sequences', 'audio_wav', 'sampling_rate']:
                    if hasattr(outputs, attr):
                        val = getattr(outputs, attr)
                        print(f"  - Attribute '{attr}': type={type(val)}")
            print("--- TUPLE INSPECTION END ---\n")

            audio_wav = None
            sampling_rate = 24000 

            if isinstance(outputs, tuple):
                final_output_ids = outputs[0]
                if len(outputs) > 1:
                    audio_wav = outputs[1]
                if len(outputs) > 2:
                    sampling_rate = outputs[2]
            else:
                final_output_ids = getattr(outputs, "sequences", outputs)
                audio_wav = getattr(outputs, "audio_wav", None)
                sampling_rate = getattr(outputs, "sampling_rate", 24000)

            input_len = final_inputs['input_ids'].shape[1]
            text_ids = final_output_ids[:, input_len:]
            final_text = self.omni_processor.processor.batch_decode(text_ids, skip_special_tokens=True)[0].strip()

            if audio_wav is not None:
                try:
                    if torch.is_tensor(audio_wav) and audio_wav.numel() > 0:
                        audio_data_np = audio_wav[0].cpu().float().numpy() if audio_wav.ndim > 1 else audio_wav.cpu().float().numpy()
                    elif isinstance(audio_wav, (list, tuple)) and len(audio_wav) > 0:
                        print(f"audio_wav is a list with {len(audio_wav)} items")
                        item = audio_wav[0]
                        if torch.is_tensor(item) and item.numel() > 0:
                            audio_data_np = item.cpu().float().numpy().flatten()
                        else:
                            print("Item inside list is empty or NOT a valid tensor.")
                            audio_wav = None
                    
                    if audio_wav is not None:
                        output_path = os.path.join(OUTPUT_BASE, "agent_reply.wav")
                        import scipy.io.wavfile as wavfile
                        wavfile.write(output_path, sampling_rate, audio_data_np)
                        print(f"Audio feedback has been saved at: {output_path}")
                except Exception as e:
                    print(f"Audio Extraction Error: {e}")

    def cleanup(self):
        self.db_manager.close()

if __name__ == "__main__":
    agent = QwenOmniAgent(MODEL_PATH)
    
    try:
        if os.path.exists(AUDIO_BASE):
            valid_extensions = ('.wav', '.mp3', '.m4a', '.flac')
            audio_files = [
                f for f in os.listdir(AUDIO_BASE) 
                if f.lower().endswith(valid_extensions)
            ]
            
            audio_files.sort()

            if audio_files:
                print(f"Audio file {len(audio_files)} has been found in {AUDIO_BASE}:")
                for i, f in enumerate(audio_files):
                    print(f"   [{i+1}] {f}")
                
                agent.run_workflow(audio_files)
            else:
                print(f"Warning: The '{AUDIO_BASE}' folder is empty; there are no audio files to process.")
        else:
            print(f"Error: Folder '{AUDIO_BASE}' not found. Please create this folder and insert the audio file into it.")
            
    except Exception as e:
        print(f"An error occurred during execution: {e}")
    finally:
        agent.cleanup()