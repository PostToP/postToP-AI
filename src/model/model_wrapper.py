import tarfile
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ai_edge_litert.interpreter import Interpreter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from data.text_cleaning import generate_new_pipeline


class ModelWrapper:
    def __init__(
        self,
        title_pipeline: Pipeline,
        description_pipeline: Pipeline,
        category_pipeline: Pipeline,
        duration_pipeline: Pipeline,
        interpreter: object,
    ) -> None:
        self.title_pipeline = title_pipeline
        self.description_pipeline = description_pipeline
        self.category_pipeline = category_pipeline
        self.duration_pipeline = duration_pipeline
        self.text_cleaner = generate_new_pipeline(verbose=False)
        self.interpreter = interpreter

    def serialize(self, location: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            components = {
                "title_pipeline.joblib": self.title_pipeline,
                "description_pipeline.joblib": self.description_pipeline,
                "category_pipeline.joblib": self.category_pipeline,
                "duration_pipeline.joblib": self.duration_pipeline,
                "model.tflite": self.interpreter,
            }

            for filename, obj in components.items():
                filepath = Path(tmpdir) / filename
                if filename.endswith(".joblib"):
                    self.__serialize_pipeline(obj, filepath)
                elif filename.endswith(".tflite"):
                    self.__serialize_model(obj, filepath)

            with tarfile.open(location, "w:gz") as tar:
                for filename in components:
                    filepath = Path(tmpdir) / filename
                    tar.add(filepath, arcname=filename)

    @staticmethod
    def __serialize_pipeline(pipeline: Pipeline, filepath: str) -> None:
        with open(filepath, "wb") as f:
            joblib.dump(pipeline, f)

    @staticmethod
    def __serialize_model(model: object, filepath: str) -> None:
        with open(filepath, "wb") as f:
            f.write(model)

    @staticmethod
    def deserialize(location: str) -> "ModelWrapper":
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(location, "r:gz") as tar:
                tar.extractall(tmpdir, filter="data")
                members = tar.getmembers()

            components = {}
            interpreter = None
            for member in members:
                member_path = Path(tmpdir) / member.name
                if member.name.endswith(".joblib"):
                    with open(member_path, "rb") as f:
                        components[member.name] = joblib.load(f)
                elif member.name.endswith(".tflite"):
                    with open(member_path, "rb") as f:
                        model_bytes = f.read()
                        interpreter = Interpreter(model_content=model_bytes)
        interpreter.allocate_tensors()
        return ModelWrapper(
            title_pipeline=components["title_pipeline.joblib"],
            description_pipeline=components["description_pipeline.joblib"],
            category_pipeline=components["category_pipeline.joblib"],
            duration_pipeline=components["duration_pipeline.joblib"],
            interpreter=interpreter,
        )

    def warmup(self) -> None:
        self.predict("Warmup title", "Warmup description", ["Music"], [300])

    def preprocess_input(
        self,
        title: str,
        description: str,
        category: list,
        duration: list,
    ) -> tuple:
        temp_df = pd.DataFrame(
            {
                "Title": [title],
                "Description": [description],
            },
        )
        trans = ColumnTransformer(
            [
                ("title", self.text_cleaner, "Title"),
                ("description", self.text_cleaner, "Description"),
            ],
            remainder="drop",
        )
        temp = trans.fit_transform(temp_df)
        title = temp[0][0]
        description = temp[0][1]
        title = self.title_pipeline.transform([title])
        description = self.description_pipeline.transform([description])
        category = self.category_pipeline.transform([category])
        duration = self.duration_pipeline.transform([duration])

        title = np.array(title).reshape(1, -1).astype(np.float32)
        description = np.array(description).reshape(1, -1).astype(np.float32)
        category = np.array(category).reshape(1, -1).astype(np.float32)
        duration = np.array(duration).reshape(1, -1).astype(np.float32)

        return title, description, category, duration

    def predict(
        self,
        title: str,
        description: str,
        category: list,
        duration: list,
    ) -> float:
        duration = [duration]  # ensure duration is a list
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
