import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Any, Optional, Dict
from langchain_core.runnables.base import Runnable, RunnableConfig

class TinyLLM(Runnable):
    def __init__(self):
        self.model_name = "microsoft/Phi-3.5-mini-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="mps"
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.95
        }

    def _format_prompt(self, input_dict: Dict[str, Any]) -> str:
        """Format the prompt template with the current page content."""
        # For debugging - print what we received
        print("Received input:", input_dict)
        
        if not isinstance(input_dict, dict) or "current_page" not in input_dict:
            return str(input_dict)
            
        # The ChatPromptTemplate has already formatted the template
        # We just need to extract the content
        if isinstance(input_dict["current_page"], str):
            return input_dict["current_page"]
        
        # If it's a message object, get its content
        if hasattr(input_dict["current_page"], "content"):
            return input_dict["current_page"].content
            
        return str(input_dict["current_page"])

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        """Process input and generate response."""
        # Format the prompt with the input content
        formatted_prompt = self._format_prompt(input)
        
        # For debugging - print what we're sending to the model
        print("\nSending to model:", formatted_prompt)
        
        # Create messages format
        messages = [
        {"role": "system", "content": "You are a helpful AI assistant skilled at analyzing documents and extracting information."},
        {"role": "user", "content": formatted_prompt}
    ]
        
        # Generate response
        output = self.pipe(messages, **self.generation_args)
        
        # For debugging - print what we got back
        print("\nModel output:", output[0]['generated_text'])
        
        return output[0]['generated_text']

llm = TinyLLM()