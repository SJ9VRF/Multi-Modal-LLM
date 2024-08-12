from src.llava_inference.py import LLaVaInference
from src.utils import load_image, display_image

def main():
    # Load images
    image1_url = "https://llava-vl.github.io/static/images/view.jpg"
    image2_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    image1 = load_image(image1_url)
    image2 = load_image(image2_url)
    
    # Display images (optional, if running in an environment with display capabilities)
    display_image(image1)
    display_image(image2)
    
    # Define prompts
    prompts = [
        "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
        "USER: <image>\nPlease describe this image\nASSISTANT:"
    ]
    
    # Initialize LLaVa Inference
    llava_inference = LLaVaInference()
    
    # Prepare inputs
    inputs = llava_inference.prepare_inputs(images=[image1, image2], prompts=prompts)
    
    # Generate text
    generated_texts = llava_inference.generate_text(inputs, max_new_tokens=20)
    
    # Output results
    for text in generated_texts:
        print(text.split("ASSISTANT:")[-1])

if __name__ == "__main__":
    main()
