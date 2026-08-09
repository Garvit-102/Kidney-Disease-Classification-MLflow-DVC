import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):

        self.model = self.load_model(
            self.config.path_of_model
        )

        self._valid_generator()

        self.score = self.model.evaluate(
            self.valid_generator
        )

        self.save_score()

    def save_score(self):

        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1])
        }

        save_json(
            path=Path("scores.json"),
            data=scores
        )

    def log_into_mlflow(self):

        # Set MLflow tracking server
        mlflow.set_tracking_uri(
            self.config.mlflow_uri
        )

        experiment_name = "Kidney-Disease-Classification"

        # Get existing experiment
        experiment = mlflow.get_experiment_by_name(
            experiment_name
        )

        # Create experiment if it doesn't exist
        if experiment is None:

            experiment_id = mlflow.create_experiment(
                experiment_name
            )

        else:

            experiment_id = experiment.experiment_id

        print(
            "Tracking URI:",
            mlflow.get_tracking_uri()
        )

        print(
            "Experiment ID:",
            experiment_id
        )

        # Start MLflow run
        with mlflow.start_run(
            experiment_id=experiment_id
        ):

            print("MLflow run started")

            # Log parameters
            mlflow.log_params(
                self.config.all_params
            )

            # Log metrics
            mlflow.log_metrics({
                "loss": float(self.score[0]),
                "accuracy": float(self.score[1])
            })

            # Log Keras model
            mlflow.keras.log_model(
                self.model,
                artifact_path="model"
            )

            print(
                "MLflow logging completed"
            )