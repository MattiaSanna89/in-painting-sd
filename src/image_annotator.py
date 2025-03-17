import cv2
import numpy as np
from PIL import Image
import time

class ImageAnnotator:
    def __init__(self):
        self.clicked = []
        self.labels = []
        self.rectangles = []
        self.mode = 'point'
        self.ix, self.iy = -1, -1
        self.drawing = False
        self.last_point_time = 0
        self.delay = 0.2
        self.show_image = None
        self.window_name = 'image'
        self.is_running = False

    def _draw_callback(self, event, x, y, flags, param):
        current_time = time.time()
        
        if self.mode == 'point':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.clicked.append([x, y])
                self.labels.append(1)
                cv2.circle(self.show_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(self.window_name, self.show_image)
            elif event == cv2.EVENT_MBUTTONDOWN:
                self.clicked.append([x, y])
                self.labels.append(0)
                cv2.circle(self.show_image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(self.window_name, self.show_image)
            elif event == cv2.EVENT_MOUSEMOVE:
                if flags & cv2.EVENT_FLAG_LBUTTON:
                    if current_time - self.last_point_time >= self.delay:
                        self.clicked.append([x, y])
                        self.labels.append(1)
                        cv2.circle(self.show_image, (x, y), 5, (0, 255, 0), -1)
                        cv2.imshow(self.window_name, self.show_image)
                        self.last_point_time = current_time
        elif self.mode == 'rectangle':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix, self.iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    img = self.show_image.copy()
                    cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                    cv2.imshow(self.window_name, img)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                cv2.rectangle(self.show_image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, self.show_image)
                self.rectangles.append([self.ix, self.iy, x, y])

    def start_annotation(self, image_path):
        """
        Start the annotation process for the given image.
        
        Args:
            image_path (str): Path to the image file to annotate
        """
        # Load the image
        image = cv2.imread(image_path)
        self.show_image = image
        
        # Create window and set callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._draw_callback)
        
        self.is_running = True
        print("Controls:")
        print("- Left click: Add positive point (green)")
        print("- Middle click: Add negative point (red)")
        print("- 'p': Switch to point mode")
        print("- 'r': Switch to rectangle mode")
        print("- 'q': Quit")
        
        while self.is_running:
            cv2.imshow(self.window_name, self.show_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p'):
                self.mode = 'point'
                print("Switched to point mode")
            elif key == ord('r'):
                self.mode = 'rectangle'
                print("Switched to rectangle mode")
            elif key == ord('q'):
                self.is_running = False
        
        cv2.destroyAllWindows()

    def get_annotations(self):
        """
        Returns the collected annotations.
        
        Returns:
            dict: Dictionary containing points (with labels) and rectangles
        """
        return {
            'points': list(self.clicked),
            'point_labels': list(self.labels),
            'rectangles': self.rectangles
        }

    def clear_annotations(self):
        """
        Clear all annotations
        """
        self.clicked = []
        self.labels = []
        self.rectangles = []