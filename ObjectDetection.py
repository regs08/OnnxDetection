
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.clock import Clock

from ObjectDetectionApp1.App.ProcessImage.image_processor import ImageProcessor
from ObjectDetectionApp1.App.ProcessImage.camera_canvas import CameraCanvas
from ObjectDetectionApp1.App.ProcessPredictions.prediction_processor import PredictionProcessor
from ObjectDetectionApp1.App.ProcessPredictions.onnx_model_instance import YOLOSeg


class ObjectDetection(BoxLayout):
    def __init__(self, camera_resolution, model_path, class_list, **kwargs):
        super(ObjectDetection, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.camera_resolution = camera_resolution
        self.class_list = class_list
        self.id_label_map = self.create_id_label_map()
        self.image_processor = ImageProcessor()
        self.seg_model = YOLOSeg(path=model_path)
        self.frame_counter = 0
        #model tasks
        self.run_tasks = {
            'detect': False,
            'segment': False,
        }
        ###
        # Camera and Canvas Drawer class
        self.camera_canvas = CameraCanvas(self.camera_resolution)
            #kivy's camera class
        self.add_widget(self.camera_canvas.camera)
        ###
        # Buttons setup
        ###
        play_button = ToggleButton(text='Play', size_hint_y=None, height='48dp')
        play_button.bind(on_press=self.toggle_camera)
        self.add_widget(play_button)

        capture_button = Button(text='Capture', size_hint_y=None, height='48dp')
        capture_button.bind(on_press=self.camera_canvas.capture)
        self.add_widget(capture_button)

        detect_button = ToggleButton(text='Detect', size_hint_y=None, height='48dp')
        detect_button.bind(on_press=self.toggle_detect)
        self.add_widget(detect_button)

        segment_button = ToggleButton(text='Segment',  size_hint_y=None, height='48dp')
        segment_button.bind(on_press=self.toggle_segment)
        self.add_widget(segment_button)

    def toggle_camera(self, instance):
        self.camera_canvas.camera.play = not self.camera_canvas.camera.play

    def toggle_task(self, instance, task_name):
        # Toggle the task state in the dictionary
        if task_name in self.run_tasks:
            self.run_tasks[task_name] = not self.run_tasks[task_name]
        # Clear the queue
        Clock.unschedule(self.detect)

        # If any task is set to run, call the detect method
        if any(self.run_tasks.values()):
            Clock.schedule_interval(self.detect, 1.0 / 30)  # Call detect every 1/30 seconds

        self.camera_canvas.clear_canvas()

    def toggle_detect(self, instance):
        self.toggle_task(instance, 'detect')

    def toggle_segment(self, instance):
        self.toggle_task(instance, 'segment')

    def detect(self, instance):
        self.camera_canvas.clear_canvas()
        self.frame_counter += 1
        if self.frame_counter % 100 != 0:  # Skip every 100th frame
            if self.run_tasks['detect'] or self.run_tasks['segment']:
                current_frame = self.camera_canvas.get_current_frame()
                # check if camera is loading image
                if current_frame is not None:
                    processed_predictions = self.seg_model.predict_and_process_frame(current_frame, camera_canvas=self.camera_canvas.camera)
                    if len(processed_predictions.scaled_bboxes) > 0:
                        cls_labels = self.get_labels_from_map(processed_predictions.detections.class_id)
                        conf = self.get_confidences(processed_predictions.detections.confidence)

                        if self.run_tasks['detect']:
                            self.camera_canvas.draw_annotation(bboxes=processed_predictions.scaled_bboxes,
                                                               labels=cls_labels,
                                                               conf=conf,
                                                               mask_points=None)

                        if self.run_tasks['segment']:
                            self.camera_canvas.draw_annotation(bboxes=processed_predictions.scaled_bboxes,
                                                           labels=cls_labels,
                                                           conf=conf,
                                                           mask_points=processed_predictions.scaled_masks)

    def create_id_label_map(self):
        id_label_map = {}
        for idx, label in enumerate(self.class_list):
            id_label_map[idx] = label
        return id_label_map

    def get_labels_from_map(self,  class_ids):
        return [self.id_label_map[id] for id in class_ids]

    def get_confidences(self, confidences):
        return [round(conf * 100, 2) for conf in confidences]


if __name__ == '__main__':

    from ObjectDetectionApp1.App.Config.config_loader import default_config

    # Adjust the paths in the config

    CAMERA_RESOLUTION = tuple(default_config['camera_resolution'])
    MODEL_PATH = default_config['model_paths']['onnx-seg-n']
    class_list = default_config['class_lists']['COCO']


    class ObjectionDetectionApp(App):
        def build(self):
            return ObjectDetection(camera_resolution=CAMERA_RESOLUTION, model_path=MODEL_PATH, class_list=class_list)

    ObjectionDetectionApp().run()