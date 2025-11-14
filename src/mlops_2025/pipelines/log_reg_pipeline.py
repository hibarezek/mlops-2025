from mlops_2025.preprocessing.Preprocessor import Preprocessor
from mlops_2025.data_splitter.Data_Splitter import DataSplitter
from mlops_2025.Features.features_computer import FeatureComputer
from mlops_2025.modeling.log_reg_model import LogisticRegressionModel
from mlops_2025.Evaluator.log_reg_evaluator import LogisticRegressionEvaluator
from mlops_2025.Predictor.log_reg_predictor import LogRegPredictor
from mlops_2025.io_handler.io_handler import IOHandler
from .base_pipeline import ModelPipeline


def build_log_reg_pipeline():
    """Build and return a ModelPipeline configured for Logistic Regression."""
    preprocessor = Preprocessor()
    data_splitter = DataSplitter()
    feature_computer = FeatureComputer()
    model = LogisticRegressionModel()
    evaluator_cls = LogisticRegressionEvaluator
    predictor_cls = LogRegPredictor
    io_handler = IOHandler()
    
    return ModelPipeline(
        preprocessor, 
        data_splitter,
        feature_computer, 
        model, 
        evaluator_cls, 
        predictor_cls, 
        io_handler
    )
