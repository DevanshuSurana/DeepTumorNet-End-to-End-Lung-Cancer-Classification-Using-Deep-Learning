import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json

tf.keras.__version__ = tf.__version__

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.10)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            class_mode="categorical",
            **dataflow_kwargs,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Tracking URL Type: {tracking_url_type_store}")

        with mlflow.start_run() as run:
            print(f"Active Run ID: {run.info.run_id}")
            try:
                mlflow.log_params(self.config.all_params)
                print("Logged params:", self.config.all_params)
                mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
                print("Logged metrics:", {"loss": self.score[0], "accuracy": self.score[1]})

                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(
                        self.model, "model", registered_model_name="VGG16Model"
                    )
                    print("Model registered with name: VGG16Model")
                else:
                    mlflow.keras.log_model(self.model, "model")
                    print("Model logged locally.")
            except Exception as e:
                print("MLflow logging failed:", str(e))

        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        
        # Predictions
        y_pred_probs = self.model.predict(self.valid_generator)
        y_pred = y_pred_probs.argmax(axis=1)
        y_true = self.valid_generator.classes

        # Calculate precision, recall, and F1-score
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Compute specificity
        cm = confusion_matrix(y_true, y_pred)
        tn = cm[0, 0]  # True negatives
        fp = cm[0, 1]  # False positives
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.performance_scores = {
            "loss": self.score[0],
            "accuracy": self.score[1],
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity
        }
        print("Metric           Value")
        print("-----------------------------")
        for metric, value in self.performance_scores.items():
            print(f"{metric:<17} {value}")

        self.save_score()



