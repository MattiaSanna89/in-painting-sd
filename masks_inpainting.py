import requests
import base64
import json
import cv2
import io
import numpy as np
from PIL import Image
from typing import Dict, List, Union, Optional, Tuple, BinaryIO

class ImageProcessor:
    def __init__(self, base_url: str):
        """
        Initialize the ImageProcessor with base URL for the API endpoints.
        
        Args:
            base_url (str): Base URL for the API (e.g., 'https://0c3b-151-230-234-180.ngrok-free.app')
        """
        self.base_url = base_url.rstrip('/')
        self.inpainting_endpoint = f"{self.base_url}/inpaint"
        self.masking_endpoint = f"{self.base_url}/generate_prompted_mask"

    @staticmethod
    def _convert_to_bytes(image: Union[str, bytes, np.ndarray, Image.Image, BinaryIO]) -> Tuple[bytes, str]:
        """
        Convert various image formats to bytes and determine the file extension.
        
        Args:
            image: Input image in various formats (file path, bytes, numpy array, PIL Image, or file object)
            
        Returns:
            Tuple[bytes, str]: Image bytes and file extension
        
        Raises:
            ValueError: If the image format is not supported or conversion fails
        """
        try:
            # Case 1: File path string
            if isinstance(image, str):
                with open(image, 'rb') as f:
                    return f.read(), image.split('.')[-1].lower()

            # Case 2: Bytes
            elif isinstance(image, bytes):
                return image, 'jpg'  # Default to jpg for raw bytes

            # Case 3: Numpy array
            elif isinstance(image, np.ndarray):
                img_pil = Image.fromarray(np.uint8(image))
                img_byte_arr = io.BytesIO()
                img_pil.save(img_byte_arr, format='PNG')
                return img_byte_arr.getvalue(), 'png'

            # Case 4: PIL Image
            elif isinstance(image, Image.Image):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                return img_byte_arr.getvalue(), 'png'

            # Case 5: File object
            elif hasattr(image, 'read'):
                return image.read(), 'jpg'  # Default to jpg for file objects

            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        except Exception as e:
            raise ValueError(f"Failed to convert image: {str(e)}")
        
    @staticmethod
    def _convert_to_cv2(image: Union[str, bytes, np.ndarray, BinaryIO, Image.Image]) -> np.ndarray:
        """
        Convert various image formats to cv2/numpy array format.
        
        Args:
            image: Input image in various formats
            
        Returns:
            np.ndarray: OpenCV image array
        """
        try:
            # Case 1: Already numpy array
            if isinstance(image, np.ndarray):
                return image

            # Case 2: File path string
            if isinstance(image, str):
                return cv2.imread(image, cv2.IMREAD_UNCHANGED)

            # Case 3: Bytes or file object
            if isinstance(image, bytes) or hasattr(image, 'read'):
                if hasattr(image, 'read'):
                    image = image.read()
                nparr = np.frombuffer(image, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            # Case 4: PIL Image
            if isinstance(image, Image.Image):
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            raise ValueError(f"Unsupported image type: {type(image)}")

        except Exception as e:
            raise ValueError(f"Failed to convert image: {str(e)}")

    @staticmethod
    def blur_mask(mask: Union[str, bytes, np.ndarray, BinaryIO], 
                  blur_radius: int = 3,
                  blur_type: str = 'gaussian') -> np.ndarray:
        """
        Apply blur to a mask image.
        
        Args:
            mask: Input mask in various formats
            blur_radius (int): Radius of the blur effect (must be odd)
            blur_type (str): Type of blur ('gaussian', 'median', or 'bilateral')
            
        Returns:
            np.ndarray: Blurred mask
        """
        # Convert to cv2 format
        mask_cv2 = ImageProcessor._convert_to_cv2(mask)

        # Ensure blur_radius is odd
        blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1

        # Convert to grayscale if not already
        if len(mask_cv2.shape) > 2:
            mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)

        # Apply the specified blur
        if blur_type.lower() == 'gaussian':
            return cv2.GaussianBlur(mask_cv2, (blur_radius, blur_radius), 0)
        elif blur_type.lower() == 'median':
            return cv2.medianBlur(mask_cv2, blur_radius)
        elif blur_type.lower() == 'bilateral':
            return cv2.bilateralFilter(mask_cv2, blur_radius, 75, 75)
        else:
            raise ValueError(f"Unsupported blur type: {blur_type}")
        
    def _process_image_response(self, response: requests.Response, key: str) -> Optional[Image.Image]:
        """
        Process API response and convert base64 image to PIL Image.
        
        Args:
            response (requests.Response): API response
            key (str): Key to extract base64 image from response
            
        Returns:
            Optional[Image.Image]: Processed image if successful, None otherwise
        """
        if response.status_code == 200:
            try:
                response_data = response.json()
                base64_image = response_data[key]
                image_data = base64.b64decode(base64_image)
                return Image.open(io.BytesIO(image_data))
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                return None
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

    def generate_mask(self, 
                     image: Union[str, bytes, np.ndarray, Image.Image, BinaryIO],
                     annotations: Dict[str, List],
                     borders: bool = False) -> Optional[Image.Image]:
        """
        Generate a mask using the provided annotations.
        
        Args:
            image: Input image (file path, bytes, numpy array, PIL Image, or file object)
            annotations (dict): Dictionary containing points, point_labels, and rectangles
            borders (bool): Whether to include borders in the mask
            
        Returns:
            Optional[Image.Image]: Generated mask image if successful, None otherwise
        """
        # Convert image to bytes
        image_bytes, ext = self._convert_to_bytes(image)

        # Prepare payload
        payload = {
            "input_points": json.dumps(annotations.get('points', [])),
            "input_point_labels": json.dumps(annotations.get('point_labels', [])),
            "input_rectangles": json.dumps(annotations.get('rectangles', [])),
            "borders": borders
        }

        # Prepare files
        files = {
            "init_image": (f"image.{ext}", image_bytes, f"image/{ext}")
        }

        # Send request
        response = requests.post(self.masking_endpoint, data=payload, files=files)
        return self._process_image_response(response, 'mask_image')

    def inpaint_image(self,
                     image: Union[str, bytes, np.ndarray, Image.Image, BinaryIO],
                     mask: Union[str, bytes, np.ndarray, Image.Image, BinaryIO],
                     positive_prompt: str,
                     negative_prompt: str = "",
                     blur_mask: bool = False,
                     blur_radius: int = 3,
                     blur_type: str = 'gaussian',
                     use_refiner: bool = True,
                     seed: Optional[int] = None,
                     n_steps: int = 70,
                     high_noise_frac: float = 0.8) -> Optional[Image.Image]:
        """
        Perform inpainting on an image using a mask.
        
        Args:
            image: Input image in various formats
            mask: Mask image in various formats
            positive_prompt (str): Positive prompt for inpainting
            negative_prompt (str): Negative prompt for inpainting
            blur_mask (bool): Whether to blur the mask before inpainting
            blur_radius (int): Radius of the blur effect if blur_mask is True
            blur_type (str): Type of blur ('gaussian', 'median', or 'bilateral')
            use_refiner (bool): Whether to use refiner
            seed (Optional[int]): Random seed for reproducibility
            n_steps (int): Number of denoising steps
            high_noise_frac (float): High noise fraction
            
        Returns:
            Optional[np.ndarray]: Inpainted image if successful
        """
        # Process mask if blurring is requested
        if blur_mask:
            mask = self.blur_mask(mask, blur_radius, blur_type)

        # Convert images to bytes
        image_bytes, image_ext = self._convert_to_bytes(image)
        mask_bytes, mask_ext = self._convert_to_bytes(mask)

        # Prepare files
        files = {
            'init_image': (f'image.{image_ext}', image_bytes, f'image/{image_ext}'),
            'mask_image': (f'mask.{mask_ext}', mask_bytes, f'image/{mask_ext}')
        }

        # Prepare data
        data = {
            'prompt': positive_prompt,
            'negative_prompt': negative_prompt,
            'use_refiner': use_refiner,
            'seed': seed,
            'n_steps': n_steps,
            'high_noise_frac': high_noise_frac
        }

        # Send request
        response = requests.post(self.inpainting_endpoint, files=files, data=data)
        return self._process_image_response(response, 'inpainted_image')

    def process_full_pipeline(self,
                            image: Union[str, bytes, np.ndarray, Image.Image, BinaryIO],
                            annotations: Dict[str, List],
                            positive_prompt: str,
                            negative_prompt: str = "",
                            blur_mask: bool = False,
                            blur_radius: int = 3,
                            blur_type: str = 'gaussian',
                            use_refiner: bool = True,
                            seed: Optional[int] = None,
                            n_steps: int = 70,
                            high_noise_frac: float = 0.8) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Run the complete pipeline: generate mask and perform inpainting.
        
        Args:
            image: Input image (file path, bytes, numpy array, PIL Image, or file object)
            annotations (dict): Annotations for mask generation
            positive_prompt (str): Positive prompt for inpainting
            negative_prompt (str): Negative prompt for inpainting
            use_refiner (bool): Whether to use refiner
            seed (Optional[int]): Random seed for reproducibility
            n_steps (int): Number of denoising steps
            high_noise_frac (float): High noise fraction
            
        Returns:
            Tuple[Optional[Image.Image], Optional[Image.Image]]: (inpainted image, mask image) if successful
        """
        # Generate mask
        mask_image = self.generate_mask(image, annotations)
        if mask_image is None:
            return None, None

        # Perform inpainting
        inpainted_image = self.inpaint_image(
            image,
            mask_image,
            positive_prompt,
            negative_prompt,
            blur_mask,
            blur_radius,
            blur_type,
            use_refiner,
            seed,
            n_steps,
            high_noise_frac
        )

        return inpainted_image, mask_image