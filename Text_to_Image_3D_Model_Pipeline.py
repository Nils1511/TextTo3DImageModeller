import os
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import numpy as np
import subprocess
import argparse

class TextTo3DPipeline:
    def __init__(self, 
                 sd_model_id="runwayml/stable-diffusion-v1-5", 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 huggingface_token=None):
        """
        Initialize the Text-to-3D pipeline.
        
        Args:
            sd_model_id: The Stable Diffusion model ID to use
            device: The device to run inference on
            huggingface_token: HuggingFace API token for accessing models
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize text-to-image model
        self.text_to_image = StableDiffusionPipeline.from_pretrained(
            sd_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_auth_token=huggingface_token
        )
        self.text_to_image.to(self.device)
        
        # Safety checker can be disabled for faster inference
        # self.text_to_image.safety_checker = None
        
        print("Text-to-image model loaded successfully")
        
        # Image-to-3D integration will be handled via API calls
        # The path where generated images and 3D models will be saved
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_image(self, prompt, num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: The text prompt to generate an image from
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            height: Height of the generated image
            width: Width of the generated image
            
        Returns:
            PIL Image object
        """
        # Add the bonus tip "Text-to-Illustration" to the prompt for better results
        enhanced_prompt = "Text-to-Illustration " + prompt
        
        print(f"Generating image with prompt: {enhanced_prompt}")
        with torch.no_grad():
            image = self.text_to_image(
                enhanced_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            ).images[0]
        
        # Save the generated image
        image_path = os.path.join(self.output_dir, "generated_image.png")
        image.save(image_path)
        print(f"Image saved to {image_path}")
        
        return image, image_path
    
    def convert_image_to_3d_trellis(self, image_path, api_url="https://api-inference.huggingface.co/models/JeffreyXiang/TRELLIS"):
        """
        Convert an image to a 3D model using the TRELLIS API.
        
        Args:
            image_path: Path to the image to convert
            api_url: URL of the TRELLIS API
            
        Returns:
            Path to the 3D model file
        """
        print(f"Converting image to 3D model using TRELLIS...")
        
        # Implementation for direct API call
        # Note: In a production environment, you should implement proper error handling and rate limiting
        headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_TOKEN', '')}"}
        
        with open(image_path, "rb") as f:
            data = f.read()
        
        response = requests.post(api_url, headers=headers, data=data)
        
        # Save the 3D model
        model_path = os.path.join(self.output_dir, "generated_model.obj")
        with open(model_path, "wb") as f:
            f.write(response.content)
        
        print(f"3D model saved to {model_path}")
        return model_path
    
    def convert_image_to_3d_hunyuan(self, image_path, api_url="https://api-inference.huggingface.co/models/tencent/Hunyuan3D-2"):
        """
        Convert an image to a 3D model using the Hunyuan3D-2 API.
        
        Args:
            image_path: Path to the image to convert
            api_url: URL of the Hunyuan3D-2 API
            
        Returns:
            Path to the 3D model file
        """
        print(f"Converting image to 3D model using Hunyuan3D-2...")
        
        headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_TOKEN', '')}"}
        
        with open(image_path, "rb") as f:
            data = f.read()
        
        response = requests.post(api_url, headers=headers, data=data)
        
        # Save the 3D model
        model_path = os.path.join(self.output_dir, "generated_model_hunyuan.obj")
        with open(model_path, "wb") as f:
            f.write(response.content)
        
        print(f"3D model saved to {model_path}")
        return model_path
    
    def text_to_3d(self, text_prompt, model="trellis"):
        """
        Full pipeline from text to 3D model.
        
        Args:
            text_prompt: The text prompt to generate a 3D model from
            model: The model to use for image-to-3D conversion ("trellis" or "hunyuan")
            
        Returns:
            Path to the 3D model file
        """
        # Step 1: Generate image from text
        image, image_path = self.generate_image(text_prompt)
        
        # Step 2: Convert image to 3D model
        if model.lower() == "trellis":
            model_path = self.convert_image_to_3d_trellis(image_path)
        elif model.lower() == "hunyuan":
            model_path = self.convert_image_to_3d_hunyuan(image_path)
        else:
            raise ValueError(f"Unknown model: {model}. Choose 'trellis' or 'hunyuan'.")
        
        return model_path

# Command line interface
def main():
    parser = argparse.ArgumentParser(description="Text-to-3D Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for 3D model generation")
    parser.add_argument("--model", type=str, default="trellis", choices=["trellis", "hunyuan"], 
                        help="Model to use for image-to-3D conversion")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5", 
                        help="Stable Diffusion model ID")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    
    args = parser.parse_args()
    
    # Check for HUGGINGFACE_TOKEN environment variable
    if "HUGGINGFACE_TOKEN" not in os.environ:
        print("Warning: HUGGINGFACE_TOKEN environment variable not set.")
        print("Set it with: export HUGGINGFACE_TOKEN=your_token")
    
    # Initialize pipeline
    pipeline = TextTo3DPipeline(sd_model_id=args.sd_model)
    
    # Generate 3D model
    model_path = pipeline.text_to_3d(args.prompt, model=args.model)
    print(f"3D model generated at: {model_path}")

if __name__ == "__main__":
    main()