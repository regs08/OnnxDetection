from supervision import Detections
import numpy as np
import cv2


class PredictionProcessor:
    """
    A wrapper class for predictions obtained from the model.

    This class utilizes the `supervision` library to store the prediction
    and provides methods to scale bounding boxes and masks to match the image.

    :param prediction: The prediction object obtained from the model.
    :param detections_format: The format of the detections ('yolov8' supported).
    :param camera: The camera object representing the canvas view (optional).
    """

    def __init__(self, prediction, pred_image_shape, detections_format='yolov8', camera=None):
        self.detections_format = detections_format
        self.prediction = prediction
        self.detections = self.get_detections_object()
        self.pred_image_height, self.pred_image_width = pred_image_shape
        self.camera = camera

        if self.camera:
            # Calculate scaling factors and camera position for later use
            self.camera_xmin = self.camera.center_x - self.camera.resolution[0] / 2
            self.camera_ymin = self.camera.center_y - self.camera.resolution[1] / 2
            self.scale_x = self.camera.resolution[0] / self.pred_image_width
            self.scale_y = self.camera.resolution[1] / self.pred_image_height

            # Scale bounding boxes and masks to the camera view if available
            self.scaled_bboxes = self.scale_bboxes_to_camera()
            if self.detections.mask is not None:
                self.scaled_masks = self.scale_masks_to_camera()
            else:
                self.scaled_masks = None

    def scale_bboxes_to_camera(self):
        """
        Scale and adjust bounding boxes to match the camera view.

        This method flips and scales the y-values, and scales the x-values
        to match the camera resolution and position on the canvas.

        :return: List of scaled bounding boxes in Kivy format (x_min, y_min, width, height).
        """
        bboxes = []
        for i, bbox in enumerate(self.detections.xyxy):
            # Flip and scale y values, and scale x values
            x_min = int((bbox[0] * self.scale_x) + self.camera_xmin)
            y_min = int(((self.pred_image_height - bbox[3]) * self.scale_y) + self.camera_ymin)
            x_max = int((bbox[2] * self.scale_x) + self.camera_xmin)
            y_max = int(((self.pred_image_height - bbox[1]) * self.scale_y) + self.camera_ymin)

            width = x_max - x_min
            height = y_max - y_min
            rect = (x_min, y_min, width, height)
            bboxes.append(rect)

        return bboxes

    def scale_masks_to_camera(self):
        """
        Scale and adjust mask points to match the camera view.

        This method resizes the mask, finds its contours, transforms the points
        to match the camera resolution, and scales them based on the camera's position.

        :return: List of flattened points (alternating x and y coordinates) for each mask.
        """
        points_out = []
        for mask in self.detections.mask:
            resized_mask = cv2.resize(mask.astype(np.uint8), self.camera.resolution,
                                      interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = []
            for contour in contours:
                mask_points = contour.reshape(-1, 2).tolist()
                # Flip y axis and scale masks to camera
                transformed_points = [(x, self.camera.resolution[1] - y) for x, y in mask_points]
                scaled_points = [(x + self.camera_xmin, y + self.camera_ymin) for x, y in transformed_points]
                # Flatten for list of [x, y, ..., xn, yn] to be used for Kivy Line
                flattened_points = [coord for point in scaled_points for coord in point]

                points.append(flattened_points)
            points_out.extend(points)

        return points_out

    def get_pred_shape(self):
        """
        Get the shape of the prediction image.

        :return: Tuple containing the height and width of the prediction image.
        """
        return self.prediction.boxes.orig_shape

    def get_detections_object(self):
        """
        Load detections object from the specified format.

        :return: Detections object representing the prediction.
        :raises UnsupportedFormatException: If an unsupported format is provided.
        """
        if self.detections_format == 'yolov8':
            return Detections.from_yolov8(self.prediction[0])
        if self.detections_format == 'onnx':
            detections = Detections.empty()
            detections.xyxy = self.prediction[0]
            detections.confidence = self.prediction[1]
            detections.class_id = self.prediction[2]
            detections.mask = self.prediction[3]
            #print(len(self.prediction.boxes_xyxy))
            return detections
        else:
            raise UnsupportedFormatException(f"Unsupported detection format: {self.detections_format}")


class UnsupportedFormatException(Exception):
    """
    Custom exception to be raised when an unsupported detection format is provided.
    """
    pass

