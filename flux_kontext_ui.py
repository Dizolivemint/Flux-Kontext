import os
# Set environment variables before importing torch
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"

import modal
import gradio as gr
from io import BytesIO
from PIL import Image
import torch
from fastapi import FastAPI

# Try to import FluxKontextPipeline
try:
    from diffusers import FluxKontextPipeline
except ImportError:
    try:
        from diffusers.pipelines.flux import FluxKontextPipeline
    except ImportError:
        raise ImportError(
            "FluxKontextPipeline not found. Please ensure you have the latest diffusers installed "
            "from git: pip install git+https://github.com/huggingface/diffusers.git"
        )

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

# === MODAL SETUP ===
flux_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git@main",
        "transformers>=4.44.0",
        "accelerate>=0.26.0",
        "safetensors>=0.4.2",
        "huggingface_hub>=0.23.0",
        "sentencepiece>=0.2.0",
        "protobuf>=3.20.0",
        "peft>=0.8.0",
        "gradio[mcp]>=4.0.0",
        "Pillow>=10.0.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(name="flux-kontext-sota", image=flux_image)

@app.cls(
    gpu="H100",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("mcp-api-key"),
    ],
    volumes={"/cache": modal.Volume.from_name("hf-cache", create_if_missing=True)},
    scaledown_window=300,
)
class FluxModel:
    @modal.enter()
    def load_model(self):
        import diffusers
        print(f"Diffusers version: {diffusers.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        
        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=os.environ["HF_TOKEN"],
            cache_dir="/cache",
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        # Enable optimizations
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing(slice_size="max")
        
        torch.cuda.empty_cache()
        print("Model loaded successfully")

    @modal.method()
    def generate(self, prompt: str, image_bytes: bytes = None,
                 guidance_scale: float = 3.5, num_inference_steps: int = 50,
                 preserve_aspect_ratio: bool = True,
                 max_dimension: int = 1536) -> bytes:
        try:
            # Parse the input image if provided
            if not image_bytes:
                raise ValueError("FluxKontextPipeline requires an input image")
            
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            original_width, original_height = image.size
            print(f"Original image size: {original_width}x{original_height}")
            
            # Calculate dimensions
            if preserve_aspect_ratio:
                # Calculate scaling to fit within max_dimension while preserving aspect ratio
                scale = min(max_dimension / original_width, max_dimension / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                
                # Round to nearest 64 pixels (Flux works better with multiples of 64)
                new_width = (new_width // 64) * 64
                new_height = (new_height // 64) * 64
                
                # Ensure minimum size of 512
                new_width = max(new_width, 512)
                new_height = max(new_height, 512)
            else:
                # Use square dimensions
                new_width = new_height = max_dimension
            
            print(f"Processing at: {new_width}x{new_height}")
            
            # Resize if needed
            if image.size != (new_width, new_height):
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Set deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Debug: Check pipeline configuration
            print(f"Guidance scale: {guidance_scale}")
            print(f"Inference steps: {num_inference_steps}")
            print(f"Prompt: {prompt}")
            
            # Generate with dynamic dimensions
            with torch.inference_mode():
                # Don't pass height/width if they might not be respected
                output = self.pipe(
                    prompt=prompt,
                    image=image,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator("cuda").manual_seed(42),
                    output_type="pil"
                ).images[0]
            
            # Check if output is different from input
            try:
                import numpy as np
                
                print(f"Input image size: {image.size}")
                print(f"Output image size: {output.size}")
                
                # Only compare if sizes match, otherwise just note the size change
                if image.size == output.size:
                    input_array = np.array(image)
                    output_array = np.array(output)
                    difference = np.abs(input_array.astype(float) - output_array.astype(float)).mean()
                    print(f"Average pixel difference: {difference}")
                    
                    if difference < 1.0:
                        print("WARNING: Output is very similar to input. The edit might not have been applied.")
                else:
                    print(f"Output dimensions changed from {image.size} to {output.size}")
                    # Do comparison on resized versions
                    comparison_size = (512, 512)
                    input_resized = image.resize(comparison_size, Image.Resampling.LANCZOS)
                    output_resized = output.resize(comparison_size, Image.Resampling.LANCZOS)
                    
                    input_array = np.array(input_resized)
                    output_array = np.array(output_resized)
                    difference = np.abs(input_array.astype(float) - output_array.astype(float)).mean()
                    print(f"Average pixel difference (after resizing): {difference}")
                
            except Exception as e:
                print(f"Could not calculate pixel difference: {e}")
            
            # Resize output to match input dimensions if preserve_aspect_ratio is True
            if preserve_aspect_ratio and output.size != (original_width, original_height):
                output = output.resize((original_width, original_height), Image.Resampling.LANCZOS)
                print(f"Resized output back to original: {original_width}x{original_height}")
            
            # Convert to bytes
            buf = BytesIO()
            output.save(buf, format="PNG")
            return buf.getvalue()
            
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise

# === GRADIO UI ===
def infer(prompt, input_image, guidance_scale, num_inference_steps, 
          preserve_aspect_ratio, max_dimension):
    """
    Wrapper function for Gradio to call the FluxModel generate method.
    """
    if not input_image:
        return None
    
    # Convert image to bytes
    buf = BytesIO()
    input_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    
    # Look up the FluxModel class from the app
    FluxModelCls = modal.Cls.from_name("flux-kontext-sota", "FluxModel")
    
    # Use the class for remote calls
    output_bytes = FluxModelCls().generate.remote(
        prompt=prompt,
        image_bytes=image_bytes,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        preserve_aspect_ratio=preserve_aspect_ratio,
        max_dimension=max_dimension
    )
    
    # Convert bytes back to PIL Image for Gradio
    return Image.open(BytesIO(output_bytes))

# Create the Gradio interface
with gr.Blocks(title="FLUX.1 Kontext Image Editor") as gr_app:
    gr.Markdown("""
    # FLUX.1 Kontext Image Editor
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image (Required)", 
                type="pil",
                height=400
            )
            prompt = gr.Textbox(
                label="Edit Instruction", 
                placeholder="Describe how you want to modify the image",
                lines=3
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                guidance_scale = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=10.0, 
                    value=6.0, 
                    step=0.5,
                    info="Higher values follow the edit instruction more closely (try 6-8 for stronger edits)"
                )
                num_inference_steps = gr.Slider(
                    label="Inference Steps", 
                    minimum=20, 
                    maximum=100, 
                    value=50, 
                    step=5,
                    info="More steps = better quality but slower"
                )
                preserve_aspect_ratio = gr.Checkbox(
                    label="Preserve Aspect Ratio", 
                    value=True,
                    info="Maintain original image proportions"
                )
                max_dimension = gr.Slider(
                    label="Max Dimension", 
                    minimum=512, 
                    maximum=2048, 
                    value=1024, 
                    step=64,
                    info="Maximum processing dimension (1024 recommended)"
                )
            
            generate_btn = gr.Button("Generate Edit", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Edited Image", 
                type="pil",
                height=400
            )
    
    generate_btn.click(
        fn=infer,
        inputs=[prompt, input_image, guidance_scale, num_inference_steps, 
                preserve_aspect_ratio, max_dimension],
        outputs=output_image,
    )

@app.function(
    image=flux_image,
    min_containers=1,
    max_containers=1,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    fastapi_app = FastAPI()
    # Mount Gradio with MCP enabled
    gr.mount_gradio_app(app=fastapi_app, blocks=gr_app, path="", mcp_server=True)
    return fastapi_app

# === TODOs ===
# TODO: Add batch processing for multiple edits
# TODO: Add style transfer capabilities
# TODO: Implement progressive editing (multiple rounds)
# TODO: Add safety checks for inappropriate content