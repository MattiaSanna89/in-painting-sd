import cv2
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class Sam2HF:
    def __init__(self, hf_version: str="facebook/sam2.1-hiera-large") -> None:
        self.sam_version = hf_version
        self.mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(self.sam_version, min_mask_region_area=10.0, use_m2m=True)
        self.predictor = self.mask_generator.predictor   

    def generate_masks(self, image: np.array):
        return self.mask_generator.generate(image)
    
    def generate_masks_from_prompt(self, image:np.array, input_points, input_labels, input_rectangles):
        self.predictor.set_image(image)

        return self.predictor.predict(
                    point_coords=input_points if len(input_points) > 0 else None,
                    point_labels=input_labels if len(input_labels) > 0 else None,
                    box=input_rectangles if len(input_rectangles) > 0 else None,
                    multimask_output=False,
                )
    
    @staticmethod
    def get_rgba_mask(masks, random_color=False, borders=True, black_and_white=False):
        for i, mask in enumerate(masks):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
    
            mask = mask.astype(np.float32)
    
            if i > 0:
                mask_image +=  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            else:
                mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

            if black_and_white:
                black_array = np.array([0., 0., 0., 0.])
                white_array = np.array([1., 1., 1., 1.])
                mask_image = np.where(mask_image != black_array, white_array, mask_image)
            
            if borders:
                # Convert mask to uint8 format for cv2.findContours()
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
        return mask_image
    
    @staticmethod
    def image_overlay(image, segmented_image):
        alpha = 0.6 # transparency for the original image
        beta = 0.4 # transparency for the segmentation map
        gamma = 0 # scalar added to each sum
    
        segmented_image = np.array(segmented_image, dtype=np.float32)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    
        image = np.array(image, dtype=np.float32) / 255.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
        return image
    