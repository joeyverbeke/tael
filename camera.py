import cv2
from PIL import Image

def capture_image():
    """Capture an image from the webcam at full resolution."""
    cap = cv2.VideoCapture(0)

    # Set the resolution to the maximum supported by your webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Retrying...")
        cap.release()
        return None

    # Convert the image to PIL format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Save the image (optional)
    image.save("image.jpg")
    
    cap.release()
    return image
