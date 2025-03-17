import json
import torch
from diffusers import StableDiffusionXLInpaintPipeline

class InPaintingStableDiffusion:
    def __init__(self) -> None:
        self.base_model_name = None,
        self.refiner_model_name = None,
        self.inpainting_params = None,
        self.base_inpainting_pipe = None,
        self.refiner_inpainting_pipe = None
        self.load_default()
        self.load_models()

    def load_default(self):
        with open("config.json", "r") as f:
            params = json.load(f)
        
        self.base_model_name = params["base_model_id"]
        self.refiner_model_name = params["refiner_model_id"]
        self.inpainting_params = params["inpainting_parameters"]
    
    def load_models(self):
        self.base_inpainting_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

        self.base_inpainting_pipe.enable_model_cpu_offload()

        self.refiner_inpainting_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.refiner_model_name,
            text_encoder=self.base_inpainting_pipe.text_encoder_2,
            vae=self.base_inpainting_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )

        self.refiner_inpainting_pipe.enable_model_cpu_offload()
    
    def inpaint(self,
            init_image, 
            mask_image, 
            prompt, 
            negative_prompt= None, 
            use_refiner=True, 
            seed=None, 
            n_steps= 50, 
            high_noise_frac=None,
            **kwargs
        ):

        if seed:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None
        if not high_noise_frac:
            high_noise_frac = self.inpainting_params["high_noise_frac"]
            
        image = self.base_inpainting_pipe(prompt=prompt,
                    negative_prompt= negative_prompt,
                    image=init_image,
                    mask_image=mask_image, 
                    num_inference_steps=n_steps,
                    strength=self.inpainting_params["strength"],
                    denoising_end= high_noise_frac if use_refiner else 1.0,
                    output_type= "latent" if use_refiner else "pil",
                    generator= generator,
                    **kwargs).images[0]

        if use_refiner:
            image = self.refiner_inpainting_pipe(prompt=prompt,
                            negative_prompt= negative_prompt,
                            mask_image=mask_image,  
                            num_inference_steps= n_steps,
                            denoising_start = high_noise_frac,
                            image=image[None, :], 
                            generator= generator).images[0]

        return image     
