import dill
import pickle
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from data.text_cleaning import generate_test_preprocess_pipeline

class ModelWrapper:
        def __init__(self, title_pipeline, description_pipeline, category_pipeline):
            self.title_pipeline = title_pipeline
            self.description_pipeline = description_pipeline
            self.category_pipeline = category_pipeline
            self.text_cleaner = generate_test_preprocess_pipeline()

        def save_model(self, model,model_loc):
            with open(model_loc, 'wb') as f:
                f.write(model)
            self.model_path = model_loc

        def serialize(self, location):
            with open(location, 'wb') as f:
                dill.dump(self, f)

        def deserialize(location):
            with open(location, 'rb') as f:
                return dill.load(f)
            
        def load_model(self):
            self.interpreter = Interpreter(self.model_path)
            self.interpreter.allocate_tensors()

        def preprocess_input(self, title, description, category, duration):
            title = self.text_cleaner.process_text(title)
            description = self.text_cleaner.process_text(description)
            title = self.title_pipeline[0].encode(title)
            description = self.description_pipeline[0].encode(description)
            category = self.category_pipeline[0].encode(category)
            duration = [duration]

            title = self.title_pipeline[1].encode(title)
            description = self.description_pipeline[1].encode(description)
            category = self.category_pipeline[1].encode(category)

            title = np.array(title).reshape(1, -1).astype(np.float32)
            description = np.array(description).reshape(1, -1).astype(np.float32)
            category = np.array(category).reshape(1, -1).astype(np.float32)
            duration = np.array(duration).astype(np.float32)

            return title, description, category, duration


        def predict(self, title, description, category, duration):
            title, description, category, duration = self.preprocess_input(title, description, category, duration)
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            input_dict = {
                "serving_default_title_input:0": title,
                "serving_default_desc_input:0": description,
                "serving_default_cat_input:0": category,
                "serving_default_dur_input:0": duration
            }

            for detail in input_details:
                self.interpreter.set_tensor(detail['index'], input_dict[detail['name']])

            self.interpreter.invoke()

            output = self.interpreter.get_tensor(output_details[0]['index'])
            return output[0][0]
