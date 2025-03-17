from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import base64
import json
import io


# Import the Segment Anything class
from segmentation import Sam2HF
# Import the InPaintingStableDiffusion class
from inpainting import InPaintingStableDiffusion


app = FastAPI()

# Initialize the segmentation class
segmentation_service = Sam2HF()

# Initialize the inpainting class
inpainting_service = InPaintingStableDiffusion()

@app.post("/generate_prompted_mask")
async def generate_mask(
    input_points: str = Form(..., description="List of input points for segmentation"),
    input_point_labels: str = Form(..., description="List of labels for the input points"),
    input_rectangles: str = Form(..., description="List of rectangles for segmentation"),
    init_image: UploadFile = File(..., description="Initial image to be segmented"),
    borders: bool = Form(False, description="Whether to add borders to the mask"),
):
    # Read and process the uploaded image
    init_image_data = await init_image.read()
    init_image = Image.open(io.BytesIO(init_image_data)).convert("RGB")
    image_array = np.array(init_image)

    # Parse the JSON strings into Python lists
    input_points = json.loads(input_points)
    input_point_labels = json.loads(input_point_labels)
    input_rectangles = json.loads(input_rectangles)

    # Convert input prompts to numpy arrays
    input_points = np.array(input_points)
    input_point_labels = np.array(input_point_labels)
    input_rectangles = np.array(input_rectangles)

    # Generate masks using the segmentation service
    masks, _, _ = segmentation_service.generate_masks_from_prompt(
        image_array,
        input_points=input_points,
        input_labels=input_point_labels,
        input_rectangles=input_rectangles
    )

    # Create an RGBA mask
    rgba_mask = Sam2HF.get_rgba_mask(masks=masks, borders=borders, black_and_white=True)
    rgba_array = (rgba_mask * 255).astype(np.uint8)
    pil_rgba_mask = Image.fromarray(rgba_array, 'RGBA')

    # Convert the image to a base64-encoded string
    buffered = io.BytesIO()
    pil_rgba_mask.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return the base64 image data
    return {"mask_image": base64_image}

@app.post("/inpaint")
async def inpaint(
    init_image: UploadFile = File(..., description="Initial image to be inpainted"),
    mask_image: UploadFile = File(..., description="Mask image indicating areas to inpaint"),
    prompt: str = Form(..., description="Text prompt describing the desired inpainting"),
    negative_prompt: str = Form(None, description="Text prompt for undesired elements"),
    use_refiner: bool = Form(True, description="Whether to use the refiner model"),
    seed: int = Form(None, description="Random seed for reproducibility"),
    n_steps: int = Form(50, description="Number of inference steps"),
    high_noise_frac: float = Form(0.7, description="Fraction of noise for refinement")
):
    # Read the uploaded images
    init_image_data = await init_image.read()
    mask_image_data = await mask_image.read()

    # Convert images to PIL format
    init_image_pil = Image.open(io.BytesIO(init_image_data)).convert("RGB")
    mask_image_pil = Image.open(io.BytesIO(mask_image_data)).convert("RGB")

    # Perform inpainting
    result_image = inpainting_service.inpaint(
        init_image=init_image_pil,
        mask_image=mask_image_pil,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_refiner=use_refiner,
        seed=seed,
        n_steps=n_steps,
        high_noise_frac=high_noise_frac
    )

    # Convert the image to a base64-encoded string
    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return the base64 image data
    return {"inpainted_image": base64_image}

# To run the app: `uvicorn your_fastapi_file:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)