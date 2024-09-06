import cv2

class Camera:
    def __init__(self):
        """Initialize the camera."""
        self.cap = cv2.VideoCapture(0)
        # Set the resolution to 3264x2448
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)

    def capture_image(self):
        """Capture an image from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image.")
            return None

        # Crop to the specified square area that contains the text
        crop_x1, crop_x2 = 1100, 2600
        crop_y1, crop_y2 = 400, 1900
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize the cropped image to 672x672
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
