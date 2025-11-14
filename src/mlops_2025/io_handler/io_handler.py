from .base_handler import BaseHandler
import json
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import numpy as np


class IOHandler(BaseHandler):
    """Concrete I/O handler for saving predictions (CSV) and metrics (JSON)."""
    
    def __init__(self):
        """Initialize the handler."""
        pass

    def save_predictions(self, 
                        predictions: Union[np.ndarray, list], 
                        path: str,
                        input_df: pd.DataFrame = None,
                        probabilities: np.ndarray = None,
                        prob_column_prefix: str = "Probability_Class_") -> None:
        """
        Save predictions to CSV.
        
        Args:
            predictions: Array-like of class predictions
            path: Output file path (CSV)
            input_df: Optional DataFrame to append predictions to
            probabilities: Optional probability array (n_samples x n_classes)
            prob_column_prefix: Prefix for probability column names
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if input_df is not None:
            out = input_df.copy()
            out["Prediction"] = predictions
        else:
            out = pd.DataFrame({"Prediction": predictions})
        
        if probabilities is not None:
            proba_df = pd.DataFrame(
                probabilities,
                columns=[f"{prob_column_prefix}{i}" for i in range(probabilities.shape[1])]
            )
            out = pd.concat([out.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
        
        out.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to {output_path}")
    
    def save_metrics(self, metrics: Dict[str, Any], path: str) -> None:
        """
        Save metrics dict to JSON.
        
        Args:
            metrics: Dictionary of metrics
            path: Output file path (JSON)
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Metrics saved to {output_path}")