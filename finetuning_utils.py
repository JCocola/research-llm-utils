from typing import Dict, Tuple, Any
from openai import OpenAI
import json
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import time


def run_fine_tuning(
    training_data_path: str,
    model_id: str,
    hyperparameters: Dict[str, Any],  # Use Dict from typing
    log_dir: str = "fine_tuning_logs",
    csv_path: str = "fine_tuning_experiments.csv",
) -> Tuple[Any, Any]:  # Use Tuple from typing
    """
    Run a fine-tuning job and log all relevant information.
    """
    client = OpenAI()

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Get training data statistics
    with open(training_data_path, "r") as f:
        training_data = [json.loads(line) for line in f]

    data_stats = {
        "n_samples": len(training_data),
        "n_messages": sum(len(sample["messages"]) for sample in training_data),
        "has_system_message": any(
            "system" in msg["role"]
            for sample in training_data
            for msg in sample["messages"]
        ),
        "training_data_path": str(Path(training_data_path).absolute()),
        "training_data_size_bytes": os.path.getsize(training_data_path),
    }

    # Upload training file
    print("Uploading training file...")
    with open(training_data_path, "rb") as file:
        training_file = client.files.create(file=file, purpose="fine-tune")

    # Create fine-tuning job
    print(f"Starting fine-tuning job with model {model_id}...")
    job = client.fine_tuning.jobs.create(
        model=model_id, training_file=training_file.id, hyperparameters=hyperparameters
    )

    # Wait for completion
    print("Waiting for job completion...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"Status: {job.status}")

        if job.status in ["succeeded", "failed", "cancelled"]:
            break

        time.sleep(60)

    # Save detailed log with training data statistics
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "job_id": job.id,
        "base_model": job.model,
        "fine_tuned_model": job.fine_tuned_model,
        "training_file": job.training_file,
        "validation_file": job.validation_file,
        "status": job.status,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "trained_tokens": job.trained_tokens,
        "hyperparameters": hyperparameters,
        "training_data_stats": data_stats,
    }

    # Save JSON log
    json_filename = (
        f"fine_tuning_job_{job.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    json_filepath = os.path.join(log_dir, json_filename)
    with open(json_filepath, "w") as f:
        json.dump(log_entry, f, indent=2)
    print(f"Detailed log saved to: {json_filepath}")

    # Update CSV log with additional data statistics
    experiment_data = {
        "date": [datetime.now()],
        "job_id": [job.id],
        "base_model": [job.model],
        "model_id": [job.fine_tuned_model],
        "training_file": [job.training_file],
        "training_data_path": [data_stats["training_data_path"]],
        "n_samples": [data_stats["n_samples"]],
        "n_messages": [data_stats["n_messages"]],
        "has_system_message": [data_stats["has_system_message"]],
        "training_data_size_bytes": [data_stats["training_data_size_bytes"]],
        "n_epochs": [hyperparameters.get("n_epochs")],
        "batch_size": [hyperparameters.get("batch_size")],
        "learning_rate_multiplier": [hyperparameters.get("learning_rate_multiplier")],
        "trained_tokens": [job.trained_tokens],
        "status": [job.status],
    }

    df = pd.DataFrame(experiment_data)
    if Path(csv_path).exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"Experiment logged to: {csv_path}")

    if job.status == "succeeded":
        print(
            f"Fine-tuning completed successfully! New model ID: {job.fine_tuned_model}"
        )
    else:
        print(f"Fine-tuning ended with status: {job.status}")

    return job, job.fine_tuned_model
