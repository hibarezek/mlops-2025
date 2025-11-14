from pathlib import Path
import pandas as pd
from typing import Any, Dict
from sklearn.model_selection import train_test_split


class ModelPipeline:
    """
    Minimal generic pipeline orchestrator.
    Delegates all logic to specialized classes (preprocessor, feature_computer, model, etc).
    """
    
    def __init__(self, preprocessor, data_splitter, feature_computer, model, evaluator_cls, predictor_cls, io_handler):
        """
        Initialize pipeline with concrete implementations.
        
        Args:
            preprocessor: Instance with .process(train_df, test_df) → combined_df
            data_splitter: Instance with .split_data(combined_df) → (train_df, test_df)
            feature_computer: Instance with .compute_features(df, is_train: bool) → df
            model: Instance with .train(X, y) and .save(path)
            evaluator_cls: Evaluator class (constructor takes model_path)
            predictor_cls: Predictor class (constructor takes model_path)
            io_handler: IOHandler instance with .save_predictions() and .save_metrics()
        """
        self.preprocessor = preprocessor
        self.data_splitter = data_splitter
        self.feature_computer = feature_computer
        self.model = model
        self.evaluator_cls = evaluator_cls
        self.predictor_cls = predictor_cls
        self.io_handler = io_handler

    def run(
        self,
        train_csv: str,
        test_csv: str,
        out_dir: str,
        model_name: str,
        do_validation: bool = True,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Run pipeline: load → preprocess → split → featurize → train → evaluate → predict.
        All logic delegated to classes; pipeline just orchestrates.
        
        Returns:
            Dictionary with paths to models, metrics, predictions
        """
        out_dir = Path(out_dir)
        (out_dir / "models").mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print(f"Pipeline: {model_name.replace('.pkl', '').upper()}")
        print("="*60)
        
        # 1. Load
        print("\n[1/5] Loading raw data...")
        train_raw = pd.read_csv(train_csv)
        test_raw = pd.read_csv(test_csv)

        # 2. Preprocess (delegated to Preprocessor class)
        print("[2/5] Preprocessing...")
        combined = self.preprocessor.process(train_raw, test_raw)

        # 3. Split (delegated to DataSplitter class)
        print("[3/5] Splitting data...")
        train_proc, test_proc = self.data_splitter.split_data(combined)

        # 4. Featurize (delegated to FeatureComputer class)
        print("[4/5] Feature engineering...")
        train_feat = self.feature_computer.compute_features(train_proc, is_train=True)
        test_feat = self.feature_computer.compute_features(test_proc, is_train=False)

        # Prepare training data
        X = train_feat.drop(columns=["Survived"])
        y = train_feat["Survived"]

        # Optional validation split
        if do_validation:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, None, y, None

        # 5. Train & Evaluate
        print("[5/5] Training & Evaluating...")
        self.model.train(X_train, y_train)
        
        model_path = out_dir / "models" / model_name
        self.model.save(str(model_path))
        
        results = {"model_path": str(model_path)}

        # Evaluate if validation data available
        if do_validation and X_val is not None:
            evaluator = self.evaluator_cls(str(model_path))
            metrics = evaluator.evaluate(X_val, y_val)
            
            metrics_path = out_dir / "metrics" / f"{model_name.replace('.pkl', '')}_metrics.json"
            self.io_handler.save_metrics(metrics, str(metrics_path))
            results["metrics_path"] = str(metrics_path)
            results["metrics"] = metrics
            
            print("\n      Metrics:")
            for k, v in metrics.items():
                print(f"        {k}: {v:.4f}")

        # Predict
        test_X = test_feat.reindex(columns=X.columns, fill_value=0)
        predictor = self.predictor_cls(str(model_path))
        preds = predictor.predict(test_X)
        
        proba = None
        if hasattr(predictor.model, "predict_proba"):
            try:
                proba = predictor.model.predict_proba(test_X)
            except Exception:
                pass

        pred_path = out_dir / "predictions" / f"{model_name.replace('.pkl', '')}_predictions.csv"
        self.io_handler.save_predictions(preds, str(pred_path), input_df=test_proc.reset_index(drop=True), probabilities=proba)
        results["predictions_path"] = str(pred_path)

        print("\n" + "="*60)
        print("✓ Pipeline Complete!")
        print("="*60)
        print(f"Model:       {results['model_path']}")
        if "metrics_path" in results:
            print(f"Metrics:     {results['metrics_path']}")
        print(f"Predictions: {results['predictions_path']}")
        print("="*60 + "\n")

        return results
