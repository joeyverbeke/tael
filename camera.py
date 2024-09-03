import cv2

class Camera:
    def __init__(self):
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

    def capture_image(self):
        """Capture an image from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image.")
            return None

        # Center crop the image to a square aspect ratio
        height, width = frame.shape[:2]
        new_dim = min(height, width)
        center_x, center_y = width // 2, height // 2
        crop_x1 = center_x - new_dim // 2
        crop_y1 = center_y - new_dim // 2
        crop_x2 = crop_x1 + new_dim
        crop_y2 = crop_y1 + new_dim
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize the image to 672x672
        resized_frame = cv2.resize(cropped_frame, (672, 672), interpolation=cv2.INTER_AREA)

        # Save the resized image for verification
        cv2.imwrite("image.jpg", resized_frame)

        return resized_frame

    def release(self):
        """Release the webcam."""
        self.cap.release()

# Usage example
if __name__ == "__main__":
    cam = Camera()
    image = cam.capture_image()
    if image is not None:
        print("Image captured and processed successfully.")
    else:
        print("Image capture failed.")
    cam.release()




'''
import cv2

class Camera:
    def __init__(self):
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

    def capture_image(self):
        """Capture an image from the webcam and process it to the desired size."""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image.")
            return None

        # Calculate center crop dimensions for a square crop
        crop_size = 672
        start_x = (1600 - crop_size) // 2
        start_y = (1200 - crop_size) // 2

        # Crop the center of the image
        cropped_frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

        # Resize the cropped image to 672x672 if needed (here it's already 672x672)
        resized_frame = cv2.resize(cropped_frame, (672, 672), interpolation=cv2.INTER_AREA)

        return resized_frame

    def release(self):
        """Release the webcam."""
        self.cap.release()

# Usage example
if __name__ == "__main__":
    cam = Camera()
    image = cam.capture_image()
    if image is not None:
        print("Image captured and processed successfully.")
    else:
        print("Image capture failed.")
    cam.release()
'''