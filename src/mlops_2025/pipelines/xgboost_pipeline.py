from mlops_2025.preprocessing.Preprocessor import Preprocessor
from mlops_2025.data_splitter.Data_Splitter import DataSplitter
from mlops_2025.Features.features_computer import FeatureComputer
from mlops_2025.modeling.XGBoost_model import XGBoostModel
from mlops_2025.Evaluator.xgboost_evaluator import XGBoostEvaluator
from mlops_2025.Predictor.XGBoost_predictor import XGBoostPredictor
from mlops_2025.io_handler.io_handler import IOHandler
from .base_pipeline import ModelPipeline


def build_xgboost_pipeline():
    """Build and return a ModelPipeline configured for XGBoost."""
    preprocessor = Preprocessor()
    data_splitter = DataSplitter()
    feature_computer = FeatureComputer()
    model = XGBoostModel()
    evaluator_cls = XGBoostEvaluator
    predictor_cls = XGBoostPredictor
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
