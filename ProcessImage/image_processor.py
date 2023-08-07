import cv2

class ImageProcessor:
    def __init__(self):
        pass

    def rgba_to_rgb(self, image_data):
        if image_data.shape[-1] != 4:
            raise ValueError("Input image_data should have shape (height, width, 4) in RGBA format.")
        rgb_image = image_data[:, :, :3]
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def preprocess_frame(self, frame):
        return self.rgba_to_rgb(frame)
