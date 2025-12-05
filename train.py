"""
CI/CD Pipeline Failure Prediction - Model Training with MLflow
Trains two iterations of models and tracks experiments
"""

import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Configure MLflow for cross-platform compatibility
os.makedirs('mlruns', exist_ok=True)
os.makedirs('models', exist_ok=True)
mlflow.set_tracking_uri(os.path.abspath("mlruns"))

#mlflow.set_tracking_uri("file:////Users/maadzee/Downloads/Task2/mlruns")
#mlflow.set_experiment("CICD_Failure_Prediction")

def load_and_prepare_data():
    """Load cleaned dataset and prepare features"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = pd.read_csv('data/ci_cd_logs_cleaned.csv')
    
    # Select features
    features = [
        'hour', 'day_of_week', 'month',
        'stage_encoded', 'job_encoded', 
        'task_encoded', 'environment_encoded'
    ]
    
    X = df[features]
    y = df['is_failure']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Failure rate (train): {y_train.mean():.2%}")
    print(f"Failure rate (test): {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, features

def evaluate_model(model, X_train, X_test, y_train, y_test, features):
    """Comprehensive model evaluation"""
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    metrics['cv_f1_mean'] = cv_scores.mean()
    metrics['cv_f1_std'] = cv_scores.std()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    return metrics, feature_importance, cm

def train_iteration_1(X_train, X_test, y_train, y_test, features):
    """Iteration 1: Baseline Random Forest"""
    print("\n" + "="*70)
    print("ITERATION 1: BASELINE RANDOM FOREST")
    print("="*70)
    
    with mlflow.start_run(run_name="Iteration_1_Baseline_RandomForest"):
        
        # Model parameters
        params = {
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=params['random_state'],
            n_jobs=-1
        )
        
        print("Training Random Forest...")
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics, feature_importance, cm = evaluate_model(
            model, X_train, X_test, y_train, y_test, features
        )
        
        # Log metrics (works in CI)
        mlflow.log_metrics(metrics)
        
        # Skip artifact logging in CI (causes Windows path issues)
        # mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
        # mlflow.log_dict({
        #     'confusion_matrix': cm.tolist(),
        #     'labels': ['success', 'failure']
        # }, "confusion_matrix.json")
        # mlflow.sklearn.log_model(model, "model")
        
        # Save model locally (always works)
        joblib.dump(model, 'models/model_rf_v1.pkl')
        
        # Print results
        print("\nResults:")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"Precision:      {metrics['test_precision']:.4f}")
        print(f"Recall:         {metrics['test_recall']:.4f}")
        print(f"F1 Score:       {metrics['test_f1']:.4f}")
        print(f"ROC-AUC:        {metrics['test_roc_auc']:.4f}")
        print(f"CV F1 (5-fold): {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
        
        print("\nTop 5 Important Features:")
        print(feature_importance.head())
        
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return model, metrics

def train_iteration_2(X_train, X_test, y_train, y_test, features):
    """Iteration 2: Optimized XGBoost"""
    print("\n" + "="*70)
    print("ITERATION 2: OPTIMIZED XGBOOST")
    print("="*70)
    
    with mlflow.start_run(run_name="Iteration_2_Optimized_XGBoost"):
        
        # Model parameters (optimized)
        params = {
            'model_type': 'XGBoost',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            gamma=params['gamma'],
            random_state=params['random_state'],
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        print("Training XGBoost...")
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics, feature_importance, cm = evaluate_model(
            model, X_train, X_test, y_train, y_test, features
        )
        
        # Log metrics (works in CI)
        mlflow.log_metrics(metrics)
        
        # Skip artifact logging in CI (causes Windows path issues)
        # mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
        # mlflow.log_dict({
        #     'confusion_matrix': cm.tolist(),
        #     'labels': ['success', 'failure']
        # }, "confusion_matrix.json")
        # mlflow.sklearn.log_model(model, "model")
        
        # Save model locally (always works)
        joblib.dump(model, 'models/model_xgb_v2.pkl')
        
        # Print results
        print("\nResults:")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"Precision:      {metrics['test_precision']:.4f}")
        print(f"Recall:         {metrics['test_recall']:.4f}")
        print(f"F1 Score:       {metrics['test_f1']:.4f}")
        print(f"ROC-AUC:        {metrics['test_roc_auc']:.4f}")
        print(f"CV F1 (5-fold): {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
        
        print("\nTop 5 Important Features:")
        print(feature_importance.head())
        
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return model, metrics

def compare_iterations(metrics_v1, metrics_v2):
    """Compare both iterations"""
    print("\n" + "="*70)
    print("ITERATION COMPARISON")
    print("="*70)
    
    comparison = pd.DataFrame({
        'Iteration 1 (RF)': metrics_v1,
        'Iteration 2 (XGB)': metrics_v2
    })
    
    print("\n", comparison)
    
    # Determine best model (prioritize F1 for imbalanced data)
    if metrics_v2['test_f1'] > metrics_v1['test_f1']:
        best_model = 'XGBoost (Iteration 2)'
        best_path = 'models/model_xgb_v2.pkl'
        best_f1 = metrics_v2['test_f1']
    else:
        best_model = 'RandomForest (Iteration 1)'
        best_path = 'models/model_rf_v1.pkl'
        best_f1 = metrics_v1['test_f1']
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   F1 Score: {best_f1:.4f}")
    print(f"   ROC-AUC: {max(metrics_v1['test_roc_auc'], metrics_v2['test_roc_auc']):.4f}")
    
    # Copy best model
    import shutil
    shutil.copy(best_path, 'models/best_model.pkl')
    print(f"\n✅ Best model saved as 'models/best_model.pkl'")
    
    # Save comparison
    comparison.to_csv('models/iteration_comparison.csv')
    print("✅ Comparison saved as 'models/iteration_comparison.csv'")

def main():
    """Main training pipeline"""
    
    # Set MLflow experiment
    mlflow.set_experiment("CICD_Failure_Prediction")
    
    print("="*70)
    print("CI/CD PIPELINE FAILURE PREDICTION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test, features = load_and_prepare_data()
    
    # Train Iteration 1
    model_v1, metrics_v1 = train_iteration_1(
        X_train, X_test, y_train, y_test, features
    )
    
    # Train Iteration 2
    model_v2, metrics_v2 = train_iteration_2(
        X_train, X_test, y_train, y_test, features
    )
    
    # Compare iterations
    compare_iterations(metrics_v1, metrics_v2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. View experiments: mlflow ui")
    print("2. Test API: python app.py")
    print("3. Run tests: pytest tests/")

if __name__ == "__main__":
    main()