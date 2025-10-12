#!/usr/bin/env python3
"""
Script to create a structured output evaluation dataset in Evalap.

This version is simplified: it directly uses an in-memory sample dataset
instead of reading/writing from CSV.
"""

import os
import sys
import requests
import pandas as pd
import json

# Configuration
API_URL = "http://localhost:8000/v1"  # Adjust if running on different host/port
DATASET_NAME = "structured_output_test_1"



def create_dataset():
    """Create the dataset in Evalap."""
    # Auth token (optional)
    auth_token = os.getenv("ADMIN_TOKEN")
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    # Dataset metadata
    dataset_df = pd.read_csv('./datasets/mdso-dataset.csv')
    dataset = {
        "name": DATASET_NAME,
        "readme": """
# Structured Output Evaluation Dataset

This dataset is designed to evaluate Large Language Models' ability to extract structured information from raw text, similar to Named Entity Recognition (NER) tasks.

## Dataset Structure
- **query**: Extraction task description
- **output_true**: Expected structured output as JSON string
        """.strip(),
        "default_metric": "llm_structured_output",
        "df": dataset_df.to_json(orient="records")
    }

    print(f"Creating dataset '{DATASET_NAME}'...")

    try:
        response = requests.post(
            f"{API_URL}/dataset",
            headers=headers,
            json=dataset,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            dataset_id = result.get("id")
            print(f"✅ Dataset created successfully!")
            print(f"   Dataset ID: {dataset_id}")
            print(f"   Name: {result.get('name')}")
            print(f"   Rows: {result.get('nb_rows', 'unknown')}")
            return dataset_id
        else:
            print(f"❌ Failed to create dataset: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def main():
    print("Evalap Structured Output Dataset Creator")
    print("=" * 40)

    dataset_id = create_dataset()

    if dataset_id:
        print("🎉 Dataset creation completed!")
        print(f"   You can now use dataset ID '{dataset_id}' in experiments")
        print(f"   Default metric: llm_structured_output")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
