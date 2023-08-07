import json

class ConfigLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.file_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"No config file found at {self.file_path}. Using default config.")
            return self.default_config()
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from config file at {self.file_path}. Using default config.")
            return self.default_config()

    def default_config(self):
        return {
            'model_paths': {'yolov8n': '/default/path/here'},
            # Add your other default configurations here...
        }
json_path = "/Users/cole/PycharmProjects/kivyTutorial/ObjectDetectionApp1/App/Config/config.json"
default_config = ConfigLoader(json_path).load_config()


