"""
XGBoost training pipeline with hyperparameter tuning and threshold optimization.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import pickle
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


class PII_Classifier_Trainer:
    """Train unified XGBoost classifier for PII validation."""
    
    def __init__(self, config_path: str = None):
        """Initialize trainer with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "entity_thresholds.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.thresholds = {}
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and labels
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Separate features and target
        # Exclude metadata columns
        exclude_cols = ['text', 'is_pii']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['is_pii'].values
        
        # Encode categorical features
        categorical_features = ['entity_type', 'country']
        
        for col in categorical_features:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col])
                else:
                    X[col] = self.label_encoders[col].transform(X[col])
        
        self.feature_names = X.columns.tolist()
        
        return X.values, y, self.feature_names
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
              use_grid_search: bool = True) -> xgb.XGBClassifier:
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset
            use_grid_search: Whether to perform grid search
        
        Returns:
            Trained XGBoost model
        """
        print("Preparing training data...")
        X_train, y_train, feature_names = self.prepare_data(train_df)
        X_val, y_val, _ = self.prepare_data(val_df)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Number of features: {len(feature_names)}")
        
        if use_grid_search:
            print("\nPerforming hyperparameter tuning...")
            model = self._grid_search(X_train, y_train)
        else:
            print("\nTraining with default parameters...")
            model = xgb.XGBClassifier(
                max_depth=8,
                learning_rate=0.05,
                n_estimators=300,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
        
        self.model = model
        
        # Evaluate on validation set
        print("\n=== Validation Set Performance ===")
        self._evaluate(X_val, y_val, val_df)
        
        return model
    
    def _grid_search(self, X_train, y_train) -> xgb.XGBClassifier:
        """Perform grid search for hyperparameter tuning."""
        param_grid = {
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [200, 300, 500],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def optimize_thresholds(self, val_df: pd.DataFrame) -> dict:
        """
        Optimize classification thresholds per entity type.
        
        Args:
            val_df: Validation dataset
        
        Returns:
            Dictionary of optimal thresholds per entity type
        """
        print("\n=== Optimizing Thresholds per Entity Type ===")
        
        X_val, y_val, _ = self.prepare_data(val_df)
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        thresholds = {}
        
        for entity_type in val_df['entity_type'].unique():
            # Get samples for this entity type
            mask = val_df['entity_type'] == entity_type
            y_true_entity = y_val[mask]
            y_proba_entity = y_proba[mask]
            
            # Find optimal threshold using F1 score
            precision, recall, thresholds_pr = precision_recall_curve(y_true_entity, y_proba_entity)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds_pr[optimal_idx] if optimal_idx < len(thresholds_pr) else 0.5
            
            thresholds[entity_type] = float(optimal_threshold)
            
            print(f"{entity_type}: {optimal_threshold:.3f} (F1={f1_scores[optimal_idx]:.3f})")
        
        self.thresholds = thresholds
        return thresholds
    
    def _evaluate(self, X, y, df: pd.DataFrame):
        """Evaluate model performance."""
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        print("\nOverall Performance:")
        print(classification_report(y, y_pred, target_names=['Not PII', 'PII']))
        
        print(f"\nAUC-ROC: {roc_auc_score(y, y_pred_proba):.4f}")
        
        # Performance by entity type
        print("\n=== Performance by Entity Type ===")
        for entity_type in df['entity_type'].unique():
            mask = df['entity_type'] == entity_type
            y_true_entity = y[mask]
            y_pred_entity = y_pred[mask]
            
            from sklearn.metrics import f1_score, precision_score, recall_score
            
            f1 = f1_score(y_true_entity, y_pred_entity)
            precision = precision_score(y_true_entity, y_pred_entity)
            recall = recall_score(y_true_entity, y_pred_entity)
            
            print(f"{entity_type}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n=== Top {top_n} Most Important Features ===")
        print(importance_df.head(top_n))
        
        return importance_df
    
    def save_model(self, output_dir: str):
        """Save trained model and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_file = output_path / "xgboost_pii_classifier.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save label encoders
        encoders_file = output_path / "label_encoders.pkl"
        with open(encoders_file, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save thresholds
        thresholds_file = output_path / "optimized_thresholds.json"
        with open(thresholds_file, 'w') as f:
            json.dump(self.thresholds, f, indent=2)
        
        # Save feature names
        features_file = output_path / "feature_names.json"
        with open(features_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        print(f"\n✓ Model saved to: {output_path}")
        print(f"  - {model_file.name}")
        print(f"  - {encoders_file.name}")
        print(f"  - {thresholds_file.name}")
        print(f"  - {features_file.name}")


if __name__ == "__main__":
    # Load datasets
    data_dir = Path(__file__).parent / "datasets"
    
    print("Loading datasets...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # Initialize trainer
    trainer = PII_Classifier_Trainer()
    
    # Train model
    model = trainer.train(train_df, val_df, use_grid_search=False)
    
    # Optimize thresholds
    thresholds = trainer.optimize_thresholds(val_df)
    
    # Show feature importance
    importance_df = trainer.get_feature_importance(top_n=30)
    
    # Final evaluation on test set
    print("\n=== Test Set Performance ===")
    X_test, y_test, _ = trainer.prepare_data(test_df)
    trainer._evaluate(X_test, y_test, test_df)
    
    # Save model
    model_dir = Path(__file__).parent / "models"
    trainer.save_model(model_dir)
    
    print("\n✓ Training complete!")
