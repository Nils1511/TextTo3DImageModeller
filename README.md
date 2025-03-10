# Text-to-3D Pipeline

A comprehensive Python tool for converting text descriptions into 3D models using diffusion models and 3D generation APIs.

## Features

- **Text-to-Image-to-3D Pipeline**: Generate 3D models from text descriptions
- **Local Model Support**: Use local Stable Diffusion models for offline generation
- **Batch Processing**: Efficiently process multiple prompts in batch 
- **Web Interface**: User-friendly Gradio web UI for easy interaction
- **Advanced Quality Controls**: Fine-tune generation with negative prompts, seeds, and ControlNet
- **Multi-View Generation**: Create better 3D models using multiple perspectives
- **Format Conversion**: Convert between common 3D model formats (OBJ, GLB, STL, PLY, FBX)
- **Animation Support**: Add simple animations to generated models

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/text-to-3d-pipeline.git
cd text-to-3d-pipeline
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your HuggingFace token:
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

## Usage

### Command Line Interface

**Basic usage:**
```bash
python text_to_3d_pipeline.py --prompt "A futuristic car with smooth curves"
```

**With local model:**
```bash
python text_to_3d_pipeline.py --prompt "A dragon statue" --local_model "/path/to/local/model"
```

**Batch processing from file:**
```bash
python text_to_3d_pipeline.py --batch_file "prompts.txt" --output_format "glb"
```

**Launch web interface:**
```bash
python text_to_3d_pipeline.py --web
```

**Multi-view generation:**
```bash
python text_to_3d_pipeline.py --prompt "A coffee mug" --multi_view --num_views 6
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--prompt` | Text prompt for 3D model generation |
| `--model` | Model to use for image-to-3D conversion (trellis, hunyuan, luma) |
| `--sd_model` | Stable Diffusion model ID |
| `--local_model` | Path to local Stable Diffusion model |
| `--steps` | Number of inference steps |
| `--guidance` | Guidance scale |
| `--negative_prompt` | Negative prompt |
| `--seed` | Random seed for reproducibility |
| `--output_format` | Output format for the 3D model (obj, glb, stl, ply, fbx) |
| `--batch_file` | Path to file with batch prompts (one per line) |
| `--multi_view` | Use multi-view generation for better 3D models |
| `--num_views` | Number of views to generate if multi_view is True |
| `--web` | Launch web interface |
| `--controlnet` | Use ControlNet for improved quality |

### Web Interface

The web interface provides an intuitive way to interact with the pipeline:

1. Launch the web interface:
```bash
python text_to_3d_pipeline.py --web
```

2. Open the provided URL in your browser
3. Use the interface to:
   - Enter text prompts
   - Configure generation parameters
   - Process single or batch prompts
   - Preview and download resulting 3D models

## Examples

### Single 3D Model Generation

```python
from text_to_3d_pipeline import TextTo3DPipeline

# Initialize the pipeline
pipeline = TextTo3DPipeline()

# Generate a 3D model from text
model_path = pipeline.text_to_3d(
    "A modern office chair with ergonomic design",
    model="trellis",
    num_inference_steps=50,
    guidance_scale=7.5
)

print(f"3D model generated at: {model_path}")
```

### Batch Processing

```python
# Process multiple prompts
prompts = [
    "A modern desk lamp with LED light",
    "A minimalist coffee table",
    "An ergonomic office chair",
    "A decorative vase with floral pattern"
]

results = pipeline.batch_process(prompts, model="hunyuan")

for prompt, model_path in results.items():
    print(f"'{prompt}': {model_path}")
```

### Format Conversion

```python
# Convert from OBJ to GLB format
obj_path = "outputs/generated_model_trellis_1709123456.obj"
glb_path = pipeline.convert_model_format(obj_path, output_format="glb")

print(f"Converted model: {glb_path}")
```

## API References

The pipeline uses the following APIs:

1. **Stable Diffusion** (via HuggingFace) for text-to-image generation
2. **TRELLIS, Hunyuan3D, and Luma** (via HuggingFace) for image-to-3D conversion

## Dependencies

- torch
- diffusers
- transformers
- trimesh
- gradio
- PIL
- numpy
- requests
- tqdm

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace for their model hosting and inference APIs
- Stability AI for Stable Diffusion
- The creators of TRELLIS, Hunyuan3D, and other 3D generation models
