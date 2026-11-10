"""Generative-only RNAbpFlow inference with dropout off.

Samples a fixed number of structures per target from the trained RNAbpFlow model. The model stays in eval mode, so
the only source of variation between samples is the generative sampler. Targets and per-target sample counts are read
from <input_dir>/list.txt (one "<target_id> [n_samples]" per line).

Run from the RNAbpFlow source directory:
    python run_inference.py [--config configs/inference.yaml] [--output-dir Predictions]
"""
import argparse
import os
import pickle

import yaml
from omegaconf import DictConfig

from inference import Sampler
from src.data.data_transform import make_atom_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--output-dir", default="Predictions")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        yaml_content = yaml.safe_load(f)

    input_dir = yaml_content["inference"]["input_dir"]
    output_dir = args.output_dir
    yaml_content["inference"]["output_dir"] = output_dir
    cfg = DictConfig(yaml_content)

    sampler = Sampler(cfg)
    # Sampler construction leaves the model in eval() mode, so dropout stays off.

    list_file_path = os.path.join(input_dir, "list.txt")
    with open(list_file_path, "r") as f:
        lines = f.readlines()

    id_list = []
    sample_count_list = []
    for line in lines:
        tokens = line.strip().split()
        id_list.append(tokens[0])
        if len(tokens) == 2:
            sample_count_list.append(int(tokens[1]))

    print(f"Sampling {len(id_list)} targets, "
          f"writing to {output_dir}/")

    for idx, target_id in enumerate(id_list):
        if len(sample_count_list) == len(id_list):
            sampler._samples_cfg.samples_per_sequence = sample_count_list[idx]

        target_dir = os.path.join(output_dir, target_id)
        os.makedirs(target_dir, exist_ok=True)

        atom_dict = make_atom_mask(target_id, input_dir)
        pickle_file_path = os.path.join(target_dir, "map.pkl")
        with open(pickle_file_path, "wb") as f:
            pickle.dump(atom_dict, f)

        sampler.run_sampling(target_id)

        os.remove(pickle_file_path)

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(id_list)} targets complete")

    print("Inference complete.")


if __name__ == "__main__":
    main()
