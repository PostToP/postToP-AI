import dill
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from data.text_cleaning import generate_test_preprocess_pipeline
from model.pipeline import Pipeline


class ModelWrapper:
    def __init__(
        self,
        title_pipeline: Pipeline,
        description_pipeline: Pipeline,
        category_pipeline: Pipeline,
    ) -> None:
        self.title_pipeline = title_pipeline
        self.description_pipeline = description_pipeline
        self.category_pipeline = category_pipeline
        self.text_cleaner = generate_test_preprocess_pipeline()

    def save_model(self, model: bytes, model_loc: str) -> None:
        with open(model_loc, "wb") as f:
            f.write(model)
        self.model_path = model_loc

    def serialize(self, location: str) -> None:
        with open(location, "wb") as f:
            dill.dump(self, f)

    def deserialize(location: str) -> bytes:  # noqa: N805
        with open(location, "rb") as f:
            return dill.load(f)

    def load_model(self) -> None:
        self.interpreter = Interpreter(self.model_path)
        self.interpreter.allocate_tensors()

    def warmup(self) -> None:
        self.predict("Warmup title", "Warmup description", ["Music"], [300])

    def preprocess_input(
        self,
        title: str,
        description: str,
        category: list,
        duration: list,
    ) -> tuple:
        title = self.text_cleaner.process_text(title)
        description = self.text_cleaner.process_text(description)
        title = self.title_pipeline.process(title)
        description = self.description_pipeline.process(description)
        category = self.category_pipeline.process(category)
        duration = [duration]

        title = np.array(title).reshape(1, -1).astype(np.float32)
        description = np.array(description).reshape(1, -1).astype(np.float32)
        category = np.array(category).reshape(1, -1).astype(np.float32)
        duration = np.array(duration).astype(np.float32)

        return title, description, category, duration

    def predict(
        self,
        title: str,
        description: str,
        category: list,
        duration: list,
    ) -> float:
        title, description, category, duration = self.preprocess_input(title, description, category, duration)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_dict = {
            "serving_default_title_input:0": title,
            "serving_default_desc_input:0": description,
            "serving_default_cat_input:0": category,
            "serving_default_dur_input:0": duration,
        }

        for detail in input_details:
            self.interpreter.set_tensor(detail["index"], input_dict[detail["name"]])

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(output_details[0]["index"])
        return output[0][0]
