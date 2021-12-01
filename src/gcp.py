import json
import os
import subprocess

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery, storage
from tqdm import tqdm

load_dotenv()

########################## Storage bucket related ##########################


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str):
    """Uploads a file to the GCP bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """Downloads a file to the GCP bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = storage.Blob(source_blob_name, bucket)
    blob.download_to_filename(destination_file_name)

    print(f"File {source_blob_name} downloaded to {destination_file_name}.")


def archive_run(batch_name: str, id: int):
    """Archive a run to GCP bucket."""

    run_name = f"{batch_name}_r{id:04d}"
    batch_folder = f"models/{batch_name}"
    run_folder = f"models/{batch_name}/{run_name}"
    tgz_file = run_name + ".tar.gz"
    tgz_full_path = os.path.join(batch_folder, tgz_file)

    subprocess.run(["tar", "-czf", tgz_full_path, run_folder])
    upload_blob(batch_name, tgz_full_path, tgz_file)
    subprocess.run(["rm", tgz_full_path])
    subprocess.run(["rm", "-rf", run_folder])


def retrieve_run(batch_name: str, id: int):
    """Retrieve a run from GCP bucket."""
    run_name = f"{batch_name}_r{id:04d}"
    batch_folder = f"models/{batch_name}"
    tgz_file = run_name + ".tar.gz"
    tgz_full_path = os.path.join(batch_folder, tgz_file)

    download_blob(batch_name, tgz_file, tgz_full_path)
    subprocess.run(["tar", "-xzf", tgz_full_path])
    subprocess.run(["rm", tgz_full_path])


########################## Big query related ##########################


def batch_config_to_bigquery(batch_cfgs_json: str, dataset_name: str, table_name: str):
    """Push batch configs to BQ."""

    with open(batch_cfgs_json) as f:
        batch_cfgs = json.load(f)

    df = pd.DataFrame()
    for i, cfg in enumerate(batch_cfgs):
        # get_uuid from saved model_json
        model_config_json = os.path.join(
            cfg["params"]["tf_root"], cfg["model_folder"], "model_config.json"
        )
        with open(model_config_json) as f:
            model_config = json.load(f)

        # Copy uuid from model config to batch config
        cfg["params"]["uuid"] = model_config["uuid"]

        # Gather config to a dataframe
        df = pd.concat([df, pd.DataFrame(cfg["params"], index=[i])])

    # Create connection to BQ and push data
    client = bigquery.Client()
    dataset = client.create_dataset(dataset_name, exists_ok=True)
    table_ref = dataset.table(table_name)
    job = client.load_table_from_dataframe(df, table_ref)
    job.result()
    print("Loaded dataframe to {}".format(table_ref.path))


def csv_to_bigquery(csv_file: str, dataset_name: str, table_name: str):
    """Upload a csv file to BQ."""

    # Create connection to BQ and push data
    client = bigquery.Client()
    dataset = client.create_dataset(dataset_name, exists_ok=True)
    table_ref = dataset.table(table_name)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True
    )

    with open(csv_file, "rb") as f:
        job = client.load_table_from_file(f, table_ref, job_config=job_config)

    job.result()
    print(f"Loaded {job.output_rows} rows into {dataset_name}:{table_ref.path}")


def df_to_bigquery(df: pd.DataFrame, dataset_name: str, table_name: str):
    """Upload a pandas dataframe to BQ."""

    # Create connection to BQ and push data
    client = bigquery.Client()
    dataset = client.create_dataset(dataset_name, exists_ok=True)
    table_ref = dataset.table(table_name)
    job = client.load_table_from_dataframe(df, table_ref)
    job.result()
    print(f"Loaded {job.output_rows} rows into {dataset_name}:{table_ref.path}")


def push_train_eval_to_bq(batch_name: str, id: int):
    """Push all training set eval csvs to BQ."""

    for i in tqdm(range(12)):
        f = f"models/{batch_name}/{batch_name}_r{id:04d}/eval/train_batch_{i}_triangle.csv"
        df = pd.read_csv(f, index_col=0)
        df_to_bigquery(df, "station_3", "train")
