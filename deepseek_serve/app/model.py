from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import List, Iterator
import torch
from schemas import Message
from threading import Thread

class DeepSeekModel:
    def __init__(self):
        self.model_name = "deepseek-ai/deepseek-coder-1.3b-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, messages: List[Message], max_length: int = 512) -> str:
        # Convert messages to list of dicts for the chat template
        messages_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Create the prompt using chat template
        inputs = self.tokenizer.apply_chat_template(
            messages_dicts,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids=inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    def generate_stream(self, messages: List[Message], max_length: int = 512) -> Iterator[str]:
        messages_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        inputs = self.tokenizer.apply_chat_template(
            messages_dicts,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # input_length = len(inputs[0])
        
        # Create a streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        # Create generation kwargs
        generation_kwargs = dict(
            input_ids=inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield from the streamer
        for new_text in streamer:
            if new_text.strip():
                yield new_text