from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import pandas as pd
import numpy as np


class BaseHandler(ABC):
    """Abstract base class for I/O handling predictions and metrics."""
    
    @abstractmethod
    def save_predictions(self, predictions: Union[np.ndarray, list], path: str, **kwargs):
        """Saves predictions to CSV."""
        pass    
    
    @abstractmethod
    def save_metrics(self, metrics: Dict[str, Any], path: str):
        """Saves metrics to JSON."""
        pass