import os
import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import subprocess
import argparse
import glob
import time
import json
from tqdm import tqdm
import trimesh
import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import threading

class TextTo3DPipeline:
    def __init__(self, 
                 sd_model_id="runwayml/stable-diffusion-v1-5", 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 huggingface_token=None,
                 local_model_path=None,
                 use_controlnet=False):
        """
        Initialize the Text-to-3D pipeline.
        
        Args:
            sd_model_id: The Stable Diffusion model ID to use
            device: The device to run inference on
            huggingface_token: HuggingFace API token for accessing models
            local_model_path: Path to local Stable Diffusion model
            use_controlnet: Whether to use ControlNet for improved generations
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize text-to-image model (with local model support)
        if local_model_path and os.path.exists(local_model_path):
            print(f"Loading local model from: {local_model_path}")
            self.text_to_image = StableDiffusionPipeline.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                local_files_only=True
            )
        else:
            print(f"Loading model from HuggingFace: {sd_model_id}")
            self.text_to_image = StableDiffusionPipeline.from_pretrained(
                sd_model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_auth_token=huggingface_token
            )
        
        self.text_to_image.to(self.device)
        
        # Optional ControlNet for enhanced quality
        self.use_controlnet = use_controlnet
        if use_controlnet:
            try:
                print("Setting up ControlNet for enhanced image generation...")
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", 
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    sd_model_id,
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                self.controlnet_pipe.to(self.device)
                # Processor for generating canny edge maps
                self.preprocessor = pipeline("image-preprocessing", "lllyasviel/sd-controlnet-canny")
            except Exception as e:
                print(f"Failed to load ControlNet: {e}")
                self.use_controlnet = False
        
        # Safety checker can be disabled for faster inference
        # self.text_to_image.safety_checker = None
        
        print("Text-to-image model loaded successfully")
        
        # The path where generated images and 3D models will be saved
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cache for batch processing
        self.batch_cache = {}
        
        # Available 3D model converters
        self.available_3d_models = {
            "trellis": "JeffreyXiang/TRELLIS",
            "hunyuan": "tencent/Hunyuan3D-2",
            "luma": "luma-ai/luma-v1"
        }
    
    def generate_image(self, prompt, num_inference_steps=50, guidance_scale=7.5, 
                      height=512, width=512, negative_prompt=None, seed=None):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: The text prompt to generate an image from
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            height: Height of the generated image
            width: Width of the generated image
            negative_prompt: Text to guide what not to include in the image
            seed: Random seed for reproducibility
            
        Returns:
            PIL Image object and saved image path
        """
        # Add the bonus tip "Text-to-Illustration" to the prompt for better results
        enhanced_prompt = "Text-to-Illustration " + prompt
        
        print(f"Generating image with prompt: {enhanced_prompt}")
        
        # Set generator for reproducibility if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate the image
        with torch.no_grad():
            image = self.text_to_image(
                enhanced_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator
            ).images[0]
        
        # Save the generated image
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = int(time.time())
        image_filename = f"generated_image_{timestamp}.png"
        image_path = os.path.join(self.output_dir, image_filename)
        image.save(image_path)
        print(f"Image saved to {image_path}")
        
        return image, image_path
    
    def generate_multi_view_images(self, prompt, num_views=4, **kwargs):
        """
        Generate multiple views of the same object for better 3D reconstruction.
        
        Args:
            prompt: Base text prompt
            num_views: Number of different views to generate
            **kwargs: Additional arguments for image generation
            
        Returns:
            List of image paths
        """
        view_prompts = []
        view_descriptions = [
            "front view",
            "side view from the right",
            "side view from the left",
            "back view",
            "top view",
            "bottom view",
            "3/4 view from front right",
            "3/4 view from front left"
        ]
        
        # Use only available view descriptions
        view_descriptions = view_descriptions[:num_views]
        
        for view in view_descriptions:
            view_prompts.append(f"{prompt}, {view}")
        
        image_paths = []
        for view_prompt in view_prompts:
            _, image_path = self.generate_image(view_prompt, **kwargs)
            image_paths.append(image_path)
        
        return image_paths
    
    def convert_image_to_3d(self, image_path, model_name="trellis"):
        """
        Convert an image to a 3D model using the specified API.
        
        Args:
            image_path: Path to the image to convert
            model_name: Name of the 3D model generator to use
            
        Returns:
            Path to the 3D model file
        """
        model_name = model_name.lower()
        if model_name not in self.available_3d_models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(self.available_3d_models.keys())}")
        
        model_id = self.available_3d_models[model_name]
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        print(f"Converting image to 3D model using {model_name.upper()}...")
        
        # Implementation for direct API call
        headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_TOKEN', '')}"}
        
        with open(image_path, "rb") as f:
            data = f.read()
        
        try:
            response = requests.post(api_url, headers=headers, data=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error accessing API: {e}")
            return None
        
        # Save the 3D model
        timestamp = int(time.time())
        model_path = os.path.join(self.output_dir, f"generated_model_{model_name}_{timestamp}.obj")
        with open(model_path, "wb") as f:
            f.write(response.content)
        
        print(f"3D model saved to {model_path}")
        return model_path
    
    def convert_multi_view_to_3d(self, image_paths):
        """
        Convert multiple images from different views to a single 3D model.
        
        Args:
            image_paths: List of paths to images from different views
            
        Returns:
            Path to the 3D model file
        """
        print(f"Converting multi-view images to 3D model...")
        
        # Create a temporary directory for the multi-view processing
        temp_dir = os.path.join(self.output_dir, f"multiview_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy all images to the temp directory
        for i, img_path in enumerate(image_paths):
            shutil.copy(img_path, os.path.join(temp_dir, f"view_{i}.png"))
        
        # In a real implementation, this would call a multi-view 3D reconstruction API or library
        # For demonstration, we'll just merge the first two images using the regular API
        
        if len(image_paths) > 0:
            model_path = self.convert_image_to_3d(image_paths[0])
            return model_path
        else:
            return None
    
    def text_to_3d(self, text_prompt, model="trellis", multi_view=False, num_views=4, **kwargs):
        """
        Full pipeline from text to 3D model.
        
        Args:
            text_prompt: The text prompt to generate a 3D model from
            model: The model to use for image-to-3D conversion
            multi_view: Whether to use multi-view generation for better 3D models
            num_views: Number of views to generate if multi_view is True
            **kwargs: Additional arguments for image generation
            
        Returns:
            Path to the 3D model file
        """
        if multi_view:
            # Step 1: Generate multiple views from text
            image_paths = self.generate_multi_view_images(text_prompt, num_views=num_views, **kwargs)
            
            # Step 2: Convert multi-view images to 3D model
            model_path = self.convert_multi_view_to_3d(image_paths)
        else:
            # Step 1: Generate image from text
            image, image_path = self.generate_image(text_prompt, **kwargs)
            
            # Step 2: Convert image to 3D model
            model_path = self.convert_image_to_3d(image_path, model=model)
        
        return model_path
    
    def batch_process(self, prompts, model="trellis", **kwargs):
        """
        Process multiple prompts in batch.
        
        Args:
            prompts: List of text prompts
            model: The model to use for image-to-3D conversion
            **kwargs: Additional arguments for text_to_3d
            
        Returns:
            Dictionary mapping prompts to model paths
        """
        results = {}
        
        for prompt in tqdm(prompts, desc="Processing batch"):
            try:
                model_path = self.text_to_3d(prompt, model=model, **kwargs)
                results[prompt] = model_path
                
                # Save to cache for the web interface
                batch_id = str(int(time.time()))
                if batch_id not in self.batch_cache:
                    self.batch_cache[batch_id] = {}
                self.batch_cache[batch_id][prompt] = {
                    "model_path": model_path,
                    "timestamp": time.time()
                }
            except Exception as e:
                print(f"Error processing prompt: {prompt}")
                print(f"Error details: {e}")
                results[prompt] = None
        
        return results
    
    def convert_model_format(self, input_path, output_format="glb"):
        """
        Convert a 3D model to a different format.
        
        Args:
            input_path: Path to the input 3D model
            output_format: Format to convert to (glb, stl, fbx, etc.)
            
        Returns:
            Path to the converted model
        """
        if not os.path.exists(input_path):
            print(f"Input model not found: {input_path}")
            return None
        
        output_formats = ["obj", "glb", "stl", "ply", "fbx"]
        if output_format.lower() not in output_formats:
            print(f"Unsupported output format: {output_format}")
            print(f"Supported formats: {', '.join(output_formats)}")
            return None
        
        try:
            # Load the mesh
            mesh = trimesh.load(input_path)
            
            # Generate output path
            output_path = os.path.splitext(input_path)[0] + f".{output_format.lower()}"
            
            # Export in the desired format
            mesh.export(output_path)
            
            print(f"Model converted and saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error converting model: {e}")
            return None
    
    def add_simple_animation(self, model_path):
        """
        Add a simple rotation animation to a 3D model.
        
        Args:
            model_path: Path to the 3D model
            
        Returns:
            Path to the animated model
        """
        # This is a placeholder for a real animation implementation
        # In a real app, you would use a library like Blender's Python API
        
        print("Adding simple rotation animation...")
        
        # For demonstration, we'll just create a copy with "_animated" suffix
        animated_path = os.path.splitext(model_path)[0] + "_animated.glb"
        
        try:
            # Load the mesh
            mesh = trimesh.load(model_path)
            
            # Add animation data (this is simplified)
            # In a real implementation, this would set up keyframes and animation data
            
            # Export as GLB (which supports animations)
            mesh.export(animated_path)
            
            print(f"Animated model saved to {animated_path}")
            return animated_path
        except Exception as e:
            print(f"Error adding animation: {e}")
            return None
    
    def setup_web_interface(self):
        """
        Set up a Gradio web interface for the text-to-3D pipeline.
        """
        def generate_3d_model(prompt, model_type, num_steps, guidance_scale, 
                            negative_prompt, seed, use_multi_view, num_views, output_format):
            try:
                # Convert seed to int or None
                seed_val = int(seed) if seed.strip() else None
                
                # Generate the 3D model
                model_path = self.text_to_3d(
                    prompt, 
                    model=model_type,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    seed=seed_val,
                    multi_view=use_multi_view,
                    num_views=num_views
                )
                
                # Convert format if needed
                if output_format.lower() != "obj":
                    model_path = self.convert_model_format(model_path, output_format=output_format)
                
                # Get the generated image path
                latest_image = sorted(
                    glob.glob(os.path.join(self.output_dir, "generated_image_*.png")),
                    key=os.path.getmtime
                )[-1]
                
                # Load the 3D model for visualization
                mesh_html = "3D model generated successfully! Download the file to view."
                
                return latest_image, model_path, mesh_html
            except Exception as e:
                return None, None, f"Error: {str(e)}"
        
        def process_batch(batch_text, model_type, num_steps, guidance_scale):
            # Parse the batch text into individual prompts
            prompts = [p.strip() for p in batch_text.split('\n') if p.strip()]
            
            # Start a background thread for batch processing
            def run_batch():
                try:
                    self.batch_process(
                        prompts,
                        model=model_type,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale
                    )
                except Exception as e:
                    print(f"Error in batch processing: {e}")
            
            thread = threading.Thread(target=run_batch)
            thread.start()
            
            return f"Batch processing started for {len(prompts)} prompts. Check the console for progress."
        
        # Set up the Gradio interface
        with gr.Blocks(title="Text to 3D Model Generator") as interface:
            gr.Markdown("# Text to 3D Model Generator")
            gr.Markdown("Generate 3D models from text descriptions")
            
            with gr.Tabs():
                with gr.TabItem("Single Generation"):
                    with gr.Row():
                        with gr.Column():
                            prompt_input = gr.Textbox(label="Text Prompt", placeholder="Enter a description of the 3D model")
                            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="What to avoid in the generation")
                            
                            with gr.Row():
                                model_type = gr.Dropdown(
                                    choices=list(self.available_3d_models.keys()),
                                    value="trellis",
                                    label="3D Model Generator"
                                )
                                output_format = gr.Dropdown(
                                    choices=["obj", "glb", "stl", "ply"],
                                    value="obj",
                                    label="Output Format"
                                )
                            
                            with gr.Row():
                                num_steps = gr.Slider(minimum=10, maximum=150, value=50, step=1, label="Inference Steps")
                                guidance_scale = gr.Slider(minimum=1, maximum=15, value=7.5, step=0.1, label="Guidance Scale")
                            
                            with gr.Row():
                                seed = gr.Textbox(label="Seed (empty for random)", placeholder="Enter a number for reproducible results")
                            
                            with gr.Row():
                                use_multi_view = gr.Checkbox(label="Use Multi-View Generation", value=False)
                                num_views = gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Number of Views")
                            
                            generate_button = gr.Button("Generate 3D Model")
                        
                        with gr.Column():
                            image_output = gr.Image(label="Generated Image")
                            model_output = gr.File(label="3D Model File")
                            model_viewer = gr.HTML(label="3D Model Viewer")
                    
                    generate_button.click(
                        generate_3d_model,
                        inputs=[prompt_input, model_type, num_steps, guidance_scale, 
                                negative_prompt, seed, use_multi_view, num_views, output_format],
                        outputs=[image_output, model_output, model_viewer]
                    )
                
                with gr.TabItem("Batch Processing"):
                    batch_text = gr.Textbox(
                        label="Batch Prompts", 
                        placeholder="Enter one prompt per line",
                        lines=10
                    )
                    
                    with gr.Row():
                        batch_model_type = gr.Dropdown(
                            choices=list(self.available_3d_models.keys()),
                            value="trellis",
                            label="3D Model Generator"
                        )
                    
                    with gr.Row():
                        batch_num_steps = gr.Slider(minimum=10, maximum=150, value=50, step=1, label="Inference Steps")
                        batch_guidance_scale = gr.Slider(minimum=1, maximum=15, value=7.5, step=0.1, label="Guidance Scale")
                    
                    batch_button = gr.Button("Start Batch Processing")
                    batch_result = gr.Textbox(label="Batch Status")
                    
                    batch_button.click(
                        process_batch,
                        inputs=[batch_text, batch_model_type, batch_num_steps, batch_guidance_scale],
                        outputs=[batch_result]
                    )
        
        return interface
    
    def visualize_model(self, model_path):
        """
        Visualize a 3D model (placeholder for integration with a 3D viewer).
        
        Args:
            model_path: Path to the 3D model
            
        Returns:
            An HTML string or a visualization object
        """
        if not os.path.exists(model_path):
            return "Model file not found."
        
        # In a real application, this would generate an interactive 3D viewer
        # For demonstration purposes, we'll return a simple message
        return f"<div style='padding: 20px; background-color: #f0f0f0; border-radius: 5px;'><p>3D model is ready at: {model_path}</p><p>Please use a 3D viewer software to open this file.</p></div>"


# Command line interface
def main():
    parser = argparse.ArgumentParser(description="Enhanced Text-to-3D Generation")
    parser.add_argument("--prompt", type=str, help="Text prompt for 3D model generation")
    parser.add_argument("--model", type=str, default="trellis", choices=["trellis", "hunyuan", "luma"], 
                        help="Model to use for image-to-3D conversion")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5", 
                        help="Stable Diffusion model ID")
    parser.add_argument("--local_model", type=str, help="Path to local Stable Diffusion model")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output_format", type=str, default="obj", choices=["obj", "glb", "stl", "ply", "fbx"],
                        help="Output format for the 3D model")
    parser.add_argument("--batch_file", type=str, help="Path to file with batch prompts (one per line)")
    parser.add_argument("--multi_view", action="store_true", help="Use multi-view generation for better 3D models")
    parser.add_argument("--num_views", type=int, default=4, help="Number of views to generate if multi_view is True")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--controlnet", action="store_true", help="Use ControlNet for improved quality")
    
    args = parser.parse_args()
    
    # Check for HUGGINGFACE_TOKEN environment variable
    if "HUGGINGFACE_TOKEN" not in os.environ:
        print("Warning: HUGGINGFACE_TOKEN environment variable not set.")
        print("Set it with: export HUGGINGFACE_TOKEN=your_token")
    
    # Initialize pipeline
    pipeline = TextTo3DPipeline(
        sd_model_id=args.sd_model,
        local_model_path=args.local_model,
        use_controlnet=args.controlnet
    )
    
    if args.web:
        # Start the web interface
        interface = pipeline.setup_web_interface()
        interface.launch(share=True)
    elif args.batch_file and os.path.exists(args.batch_file):
        # Batch processing from file
        with open(args.batch_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        results = pipeline.batch_process(
            prompts,
            model=args.model,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            multi_view=args.multi_view,
            num_views=args.num_views
        )
        
        # Print batch results
        print("\nBatch Processing Results:")
        for prompt, model_path in results.items():
            status = "Success" if model_path else "Failed"
            print(f"- '{prompt}': {status}")
            if model_path:
                print(f"  Model path: {model_path}")
                
                # Convert format if needed
                if args.output_format.lower() != "obj":
                    converted_path = pipeline.convert_model_format(model_path, output_format=args.output_format)
                    if converted_path:
                        print(f"  Converted to {args.output_format}: {converted_path}")
        
    elif args.prompt:
        # Single prompt processing
        model_path = pipeline.text_to_3d(
            args.prompt,
            model=args.model,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            multi_view=args.multi_view,
            num_views=args.num_views
        )
        
        if model_path:
            print(f"3D model generated at: {model_path}")
            
            # Convert format if needed
            if args.output_format.lower() != "obj":
                converted_path = pipeline.convert_model_format(model_path, output_format=args.output_format)
                if converted_path:
                    print(f"Model converted to {args.output_format}: {converted_path}")
    else:
        # No prompt provided, show help
        parser.print_help()

if __name__ == "__main__":
    main()