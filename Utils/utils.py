import numpy as np
import cv2

class Utils:
    
    gaussian_blur_kernel_size=77
    
    @staticmethod
    def create_video_capture(capture_id : int) -> cv2.VideoCapture:
        """
        Creates a video capture object for a given device or video file.

        Args:
            capture_id (int): Device index for camera input or path to video file.

        Returns:
            cv2.VideoCapture: Video capture object.

        Raises:
            ValueError: If the video capture could not be opened.
        """
        cap = cv2.VideoCapture(capture_id)
        if not cap.isOpened():
            raise ValueError(f'Failed to open video capture with ID: {capture_id}')
        return cap
    
    @staticmethod
    def read_image(img_path : str) -> np.ndarray:
        """
        Reads an image from the specified file path.

        Args:
            img_path (str): Path to the image file.

        Returns:
            np.ndarray: The image as a NumPy array.

        Raises:
            ValueError: If the image is not found at the specified path.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'The image is not found in the path provided: {img_path}')
        return img
    
    @staticmethod
    def convert_rgb_to_gray(img: np.ndarray) -> np.ndarray:
        """
        Converts an RGB image to grayscale.

        Args:
            img (np.ndarray): The RGB image to be converted. Should have shape (H, W, 3).

        Returns:
            np.ndarray: The resulting grayscale image with shape (H, W).

        Raises:
            ValueError: If the input image is not a 3-channel RGB image.
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('Input image must be an RGB image with 3 channels.')
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray_img
    
    @staticmethod
    def write_string_to_frame(frame: np.ndarray, text: str, position: tuple = (50, 50),
                          font_scale: float = 1.0, color: tuple = (255, 255, 255), thickness: int = 2) -> np.ndarray:
        """
        Writes a string onto a given frame with specified font size and color.

        Args:
            frame (np.ndarray): The image/frame where the text will be written.
            text (str): The string to write on the frame.
            position (tuple): The (x, y) coordinates for the bottom-left corner of the text.
            font_scale (float): Font scale factor that is multiplied by the base font size.
            color (tuple): The color of the text in BGR format (default is white).
            thickness (int): Thickness of the text strokes.

        Returns:
            np.ndarray: The frame with the string written on it.
        """
        # Choose the font
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Write the text on the frame
        cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
        return 
    
    @staticmethod
    def face_bluring(img: np.ndarray,detections: tuple):
        """
        Applies a Gaussian blur to detected faces in an image.

        Args:
        - img (np.ndarray): The input image in which faces are detected. It must be a valid image (loaded using OpenCV).
        - detections: A list or array of detected faces. Each detection consists of bounding box coordinates and confidence
                      values in the form [x, y, w, h, confidence].

        Functionality:
        - Checks if the image or the list of detections is None. If so, the original image is returned.
        - For each face detection:
            1. Extracts the bounding box coordinates (x, y, w, h) and converts them to integers.
            2. Validates that the bounding box is within the image dimensions.
            3. If valid:
                - If the smaller of w or h is greater than the predefined Gaussian blur kernel size,
                  applies a Gaussian blur to the face region.
                - Otherwise, adjusts the kernel size to fit the face and ensures it is an odd number.
            4. Replaces the original face region with the blurred version.

        Returns:
        - np.ndarray: The image with faces blurred. If no valid faces are detected, returns the original image.
        """

        if img is None or detections[1] is None:
            return img
        
        for detection in detections[1]:
            # Extract bounding box and confidence
            x, y, w, h = detection[:4]
            confidence = detection[-1]

            # Convert bounding box to integer coordinates
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            if w > 0 and h > 0 and y + h < img.shape[0] and x + w < img.shape[1]:

                if min(h,w)>Utils.gaussian_blur_kernel_size:
                    face_subimg=cv2.GaussianBlur(img[y:y+h,x:x+w],(Utils.gaussian_blur_kernel_size,Utils.gaussian_blur_kernel_size), 0)
                else:
                    if min(h,w)%2==1:
                        face_subimg=cv2.GaussianBlur(img[y:y+h,x:x+w],(min(h,w),min(h,w)), 0)
                    else:
                        face_subimg=cv2.GaussianBlur(img[y:y+h,x:x+w],(min(h,w)+1,min(h,w)+1), 0)


                
                img[y:y+h,x:x+w]=face_subimg
        return img
    
    @staticmethod
    def draw_detections(image: np.ndarray,detections: tuple):
        """
        Draws bounding boxes and confidence scores around detected faces on an image.

        Parameters:
        - image (np.ndarray): The input image where detections will be drawn. This should be a valid image (loaded using OpenCV).
        - detections: A list or array of detected faces. Each detection contains bounding box coordinates and a confidence 
                      score in the form [x, y, w, h, confidence].

        Functionality:
        - Checks if face detections exist. If not, returns the original image.
        - For each detection:
            1. Extracts the bounding box coordinates (x, y, w, h) and converts them to integers.
            2. Draws a green rectangle around the detected face region.
            3. Displays the confidence score above the rectangle in green text.

        Returns:
        - np.ndarray: The image with bounding boxes and confidence scores drawn around detected faces.
        """
        
        if detections[1] is not None:
            for detection in detections[1]:
                # Extract bounding box and confidence
                x, y, w, h = detection[:4]
                confidence = detection[-1]

                # Convert bounding box to integer coordinates
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                # Draw rectangle around the face
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the confidence score on the rectangle
                label = f'Confidence: {confidence:.2f}'

                
                Utils.write_string_to_frame(image,label,(x,y-5),font_scale=0.5,color=(0, 255, 0),thickness=1)
                
        return image