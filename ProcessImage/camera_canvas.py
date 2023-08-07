from kivy.graphics import Rectangle, Color, Ellipse
from kivy.graphics import Line
from kivy.core.text import Label
from kivy.uix.camera import Camera
from kivy.input.recorder import Recorder
import numpy as np
import time


class CameraCanvas:
    def __init__(self, camera_resolution):
        self.camera = Camera(resolution=camera_resolution, play=False)
        self.camera_canvas = self.camera.canvas
        self.drawing_canvas = self.camera.canvas.after
        self.recorder = Recorder(filename='output.mp4')  # Video recording object

        ###
        # drawing vars
        ###
        self.line_width = 2
        self.line_color = (1, 0, 0)  # Red color (R, G, B)
        self.bboxes = []
        self.labels = []
        ###

    def get_current_frame(self):
        # Get the current frame from the camera as a Texture.
        texture = self.camera.texture

        # Convert the Texture to a NumPy array.
        if texture is not None:
            width, height = texture.size
            buffer = texture.pixels
            fmt = texture.colorfmt
            if fmt == 'rgba':
                # Convert the RGBA string buffer to a NumPy array.
                image_data = np.frombuffer(buffer, dtype=np.uint8)
                image_data = image_data.reshape(height, width, 4)  # 4 channels for RGBA
                return image_data
            else:
                print("Unsupported color format. Cannot convert to NumPy array.")
        else:
            print("No texture data available. Make sure the camera is running.")

        return None

    def draw_annotation(self, bboxes, labels, conf, mask_points=None):
        """

        :param bboxes: list of bboxes
        :param labels: list of lables
        :param conf: listof conf values
        :param mask_points scaled points of mask

        :return:
        """
        with self.drawing_canvas:
            Color(*self.line_color)
            for i, bbox in enumerate(bboxes):
                rect = Line(rectangle=bbox, width=self.line_width)
                self.bboxes.append(rect)

                # Drawing label for bbox placed at xmin, ymax
                ymax = bbox[1] + bbox[3]
                pos = (bbox[0], ymax)
                text = f"{labels[i]} \t {conf[i]}"
                self.draw_label(pos, text)

                if mask_points and i < len(mask_points):  # Check if mask_points exist and i is within range
                    self.draw_mask_points(mask_points[i])

    def draw_label(self, pos, text):
        with self.drawing_canvas:
            core_label = Label(text=text, font_size=20, color=(1, 0, 0, 1))
            core_label.refresh()
            label_texture = core_label.texture
            label_texture_size = list(label_texture.size)
            label = Rectangle(pos=pos, size=label_texture_size, texture=label_texture)
            self.labels.append(label)

    def draw_mask_points(self, mask_points):
        with self.drawing_canvas:
            Line(points=mask_points, width=self.line_width)


    def draw_dot(self):
        """
        usefuly function for testing coorrds
        :return:
        """
        with self.drawing_canvas:
            # Color in RGBA format. Green color with full opacity
            Color(0, 1, 0, 1)
            # Ellipse position and size. We will draw it in the center of the camera widget
            x = self.camera.center_x - self.camera.resolution[0] /2
            y = self.camera.center_y - self.camera.resolution[1] /2
            d = 100 # Diameter of the dot
            Ellipse(pos=(x - d / 2, y - d / 2), size=(d, d))

    def capture(self, instance):
        '''
        Function to capture the images and give them names
        according to their captured time and date.
        '''
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.camera.export_to_png(f"IMG_{timestr}.png")
        print("Captured")

    def test_record(self):
        rec = Recorder(filename='myrecorder.kvi')
        rec.play = True
        
    def start_recording(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = f"VIDEO_{timestr}.h264"
        self.camera.start_recording(filename)
        print(f"Started recording to {filename}")

    def stop_recording(self):
        self.camera.stop_recording()
        print("Stopped recording")
        self.is_recording = False

    def clear_canvas(self):
        # Remove the rectangles
        self.drawing_canvas.clear()
