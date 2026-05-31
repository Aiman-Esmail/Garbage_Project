import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf

# اتصال بـ MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Garbage-Classifier")

# معلومات النموذج
with mlflow.start_run(run_name="MobileNetV2-v1"):
    
    # Parameters
    mlflow.log_param("model", "MobileNetV2")
    mlflow.log_param("input_size", "128x128")
    mlflow.log_param("num_classes", 12)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("epochs", 20)
    mlflow.log_param("batch_size", 32)
    
    # Metrics
    mlflow.log_metric("val_accuracy", 0.9433)
    mlflow.log_metric("train_accuracy", 0.9975)
    mlflow.log_metric("val_loss", 0.2641)
    
    print("✅ Run logged to MLflow!")