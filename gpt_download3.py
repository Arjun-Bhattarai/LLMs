import os
import requests  # requests library file download garna use garxa
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm  # progress bar dekhauna

def download_and_load_gpt2(model_size, models_dir):
    # Model size validate garna
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    #  Path define garna
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    #  File download garna
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    ## yaha samma aai pugda sabai model files local ma download bhai sakxa

    #  Settings ra params load garna
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    try:
        # File download request pathaune (SSL verify disable garera)
        response = requests.get(url, stream=True, verify=False)

        #File size header bata nikalne
        file_size = int(response.headers.get("content-length", 0))

        # File already exists bhaye size check garne
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Chunk size define garne (1KB)
        block_size = 1024

        # Progress bar setup garne
        progress_bar_description = url.split("/")[-1]
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    #  Parameters dictionary initialize garne (layers ko lagi)
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    #  Checkpoint bhitra sabai variables iterate garne
    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        #  Variable name process garne (model/ prefix hataune)
        variable_name_parts = name.split("/")[1:]

        #  Target dictionary select garne
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Nested dictionary create/access garne
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        #  Last key ma variable assign garne
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
