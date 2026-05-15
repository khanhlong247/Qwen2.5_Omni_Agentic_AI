import os
import torch
from typing import List, Any, Iterator, AsyncIterator
from agno.agent import Agent
from agno.models.base import Model, ModelResponse 
from agno.models.message import Message
from agno.db.sqlite import SqliteDb
from agno.memory.manager import MemoryManager
from src.utils.tensor_vault import AudioTensorVault

class LocalQwenWrapper(Model):
    def __init__(self, model, processor):
        super().__init__(id="qwen-2.5-omni")
        self.model = model
        self.processor = processor

    def invoke(self, messages: List[Message], **kwargs) -> ModelResponse:
        return self._generate_response(messages)

    async def ainvoke(self, messages: List[Message], **kwargs) -> ModelResponse:
        return self._generate_response(messages)

    def invoke_stream(self, messages: List[Message], **kwargs) -> Iterator[ModelResponse]:
        raise NotImplementedError("Streaming is not supported in LocalQwenWrapper yet.")

    async def ainvoke_stream(self, messages: List[Message], **kwargs) -> AsyncIterator[ModelResponse]:
        raise NotImplementedError("Async Streaming is not supported in LocalQwenWrapper yet.")

    def _parse_provider_response(self, response: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response_delta: Any) -> ModelResponse:
        return response_delta

    def _generate_response(self, messages: List[Message]) -> ModelResponse:
        formatted_msgs = []
        for m in messages:
            content_str = ""
            if isinstance(m.content, list):
                content_str = " ".join([block.get("text", "") for block in m.content if block.get("type") == "text"])
            else:
                content_str = str(m.content)
            formatted_msgs.append({"role": m.role, "content": content_str})

        prompt = self.processor.apply_chat_template(formatted_msgs, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        
        if self.model.device.type == "cuda":
            inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=256)
        
        input_len = inputs['input_ids'].shape[1]
        content = self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        
        return ModelResponse(content=content)

class MultimodalMemoryService:
    def __init__(self, session_id: str, user_id: str, model_instance, processor_instance):
        self.session_id = session_id
        self.user_id = user_id
        self.audio_vault = AudioTensorVault(window_minutes=5)
        
        local_qwen = LocalQwenWrapper(
            model=model_instance,
            processor=processor_instance,
        )
        
        storage = SqliteDb(session_table="agent_sessions", db_file="tmp/agno_storage.db")
        memory_db = SqliteDb(memory_table="user_memories", db_file="tmp/agno_memory.db")
        
        self.logic_agent = Agent(
            session_id=self.session_id,
            user_id=self.user_id,
            model=local_qwen,
            db=storage,
            memory_manager=MemoryManager(
                db=memory_db, 
                model=local_qwen,
                add_memories=False, 
                update_memories=False
            ),
            add_history_to_context=True,
            enable_agentic_memory=True,
            description="Professional insurance agents use Qwen-Omni to handle conversational logic."
        )

    def update_audio_memory(self, feature_map: torch.Tensor):
        self.audio_vault.add_feature_map(self.session_id, feature_map)

    def get_audio_context(self, device: str):
        return self.audio_vault.get_combined_context(self.session_id, device)

    def run_logic_turn(self, query_text: str):
        response = self.logic_agent.run(query_text)
        return response.content