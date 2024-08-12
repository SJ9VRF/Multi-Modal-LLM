
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

def load_model(model_id="llava-hf/llava-1.5-7b-hf"):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    return model, processor

def prepare_inputs(processor, images, prompts):
    inputs = processor(prompts, images=images, padding=True, return_tensors="pt").to("cuda")
    return inputs

def generate_text(model, inputs, max_new_tokens=20):
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return output
