from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import List

logger = logging.getLogger(__name__)

class ActionManager:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

    def retrieval_answer(self, query: str, context: List[str]) -> str:
        try:
            # Truncate and combine context
            truncated_context = [c[:200] + "..." if len(c) > 200 else c for c in context]
            context_text = " ".join(truncated_context)
            
            # Create prompt
            prompt = (
                f"Based on the following context, answer the question concisely:\n"
                f"Context: {context_text}\n"
                f"Question: {query}\n"
                f"Answer:"
            )

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=128,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            answer = response[len(prompt):].strip()
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ""