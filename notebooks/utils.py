"""A collection of utils to accompany folktexts notebooks."""
import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime


def get_current_timestamp() -> str:
    """Return a timestamp representing the current time up to the second."""
    return datetime.now().strftime("%Y.%m.%d-%H.%M.%S")


def load_json(path: str | Path) -> object:
    """Loads a JSON file from disk and returns the deserialized object."""
    with open(path, "r") as f_in:
        return json.load(f_in)


def find_files(root_folder, pattern):
    """Iteratively yield file paths that match the given pattern."""
    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if regex.match(filename):
                # If the filename matches the pattern, add it to the list
                yield os.path.join(dirpath, filename)


def num_features_helper(feature_subset_val, max_features_return=-1):
    """Returns how many features the given subset corresponds to."""
    if isinstance(feature_subset_val, str):
        assert feature_subset_val.lower() == "full"
        return max_features_return    # by deafult return -1 for all features
    try:
        assert isinstance(feature_subset_val, (list, tuple, set))
        return len(feature_subset_val)
    except Exception:
        return max_features_return


def parse_model_name(name: str) -> str:
    idx = name.find("--")
    if idx != -1:
        name = name[idx+2:]
    return name


def get_non_instruction_tuned_name(name):
    """Returns the name of the equivalent base model."""
    name = name.replace("-Instruct", "")    # Llamma / Mistral
    name = name.replace("-Chat", "")        # Yi
    name = name.replace("-it", "")          # Gemma
    name = name.replace("-1.1", "")         # Gemma version
    name = name.replace("-v0.2", "-v0.1")   # Mistral version
    return name


def parse_results_dict(dct) -> dict:
    """Parses results dict and brings all information to the top-level."""
    model_col = "config_model_name"
    feature_subset_col = "config_feature_subset"

    # Make a copy so we don't modify the input object
    dct = dct.copy()

    # Discard plots' paths
    dct.pop("plots", None)

    # Bring configs to top-level
    config = dct.pop("config", {})
    for key, val in config.items():
        dct[f"config_{key}"] = val

    # Parse model name
    dct[model_col] = parse_model_name(dct[model_col])
    dct["base_name"] = get_non_instruction_tuned_name(dct[model_col])
    dct["name"] = prettify_model_name(dct[model_col])

    # Is instruction-tuned model?
    dct["is_inst"] = dct["base_name"] != dct[model_col] or "(it)" in dct["name"].lower()

    # Log number of features
    dct["num_features"] = num_features_helper(dct[feature_subset_col], max_features_return=-1)
    dct["uses_all_features"] = (dct[feature_subset_col] is None) or (dct["num_features"] == -1)

    if dct[feature_subset_col] is None:
        dct[feature_subset_col] = "full"

    # Assert all results are at the top-level
    assert not any(isinstance(val, dict) for val in dct.values())
    return dct


def prettify_model_name(name: str) -> str:
    """Get prettified version of the given model name."""
    dct = {
        "Meta-Llama-3-70B": "Llama 3 70B",
        "Meta-Llama-3-70B-Instruct": "Llama 3 70B (it)",
        "Meta-Llama-3-8B": "Llama 3 8B",
        "Meta-Llama-3-8B-Instruct": "Llama 3 8B (it)",
        "Llama-3-8B": "Llama 3 8B",
        "Llama-3-8B-Instruct": "Llama 3 8B (it)",
        "Llama-3.1-8B": "Llama 3.1 8B",
        "Llama-3.1-8B-Instruct": "Llama 3.1 8B (it)",
        "Mistral-7B-Instruct-v0.2": "Mistral 7B (it)",
        "Mistral-7B-v0.1": "Mistral 7B",
        "Mistral-7B-Instruct-v0.1": "Mistral 7B (it)",
        "Mixtral-8x22B-Instruct-v0.1": "Mixtral 8x22B (it)",
        "Mixtral-8x22B-v0.1": "Mixtral 8x22B",
        "Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B (it)",
        "Mixtral-8x7B-v0.1": "Mixtral 8x7B",
        "Yi-34B": "Yi 34B",
        "Yi-34B-Chat": "Yi 34B (it)",
        "gemma-1.1-2b-it": "Gemma 2B (it)",
        "gemma-1.1-7b-it": "Gemma 7B (it)",
        "gemma-2b": "Gemma 2B",
        "gemma-7b": "Gemma 7B",
        "gemma-2-9b": "Gemma 2 9B",
        "gemma-2-9b-it": "Gemma 2 9B (it)",
        "gemma-2-27b": "Gemma 2 27B",
        "gemma-2-27b-it": "Gemma 2 27B (it)",
        "openai/gpt-4o-mini": "GPT 4o mini (it)",
        "openai/gpt-4o": "GPT 4o (it)",
    }

    if name in dct:
        return dct[name]
    else:
        logging.error(f"Couldn't find prettified name for {name}.")
        return name