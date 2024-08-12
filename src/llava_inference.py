import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

class LLaVaInference:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )

    def prepare_inputs(self, images, prompts):
        inputs = self.processor(prompts, images=images, padding=True, return_tensors="pt").to(self.device)
        return inputs

    def generate_text(self, inputs, max_new_tokens=20):
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(output, skip_special_tokens=True)
        return generated_text
