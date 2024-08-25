import cv2
from PIL import Image

def capture_image():
    """Capture an image from the webcam."""
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Retrying...")
        return None

    # Convert the image to PIL format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Save the image (optional)
    image.save("image.jpg")
    
    cap.release()
    return image
