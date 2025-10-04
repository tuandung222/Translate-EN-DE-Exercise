"""
Prepare and push dataset to HuggingFace Hub.
"""

import os
import argparse
import logging
from datasets import Dataset, DatasetDict
from utils import normalizeString

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_dataset_if_needed(filepath="german-english.txt"):
    """Download dataset from Google Drive if not present."""
    if not os.path.exists(filepath):
        import gdown

        url = "https://drive.google.com/uc?id=1CVMenH5xgDsiq9ZaGAFfkFLG5uWDZvdL"
        gdown.download(url, filepath, quiet=False)
    return filepath


def load_and_prepare_data(filepath, max_length=64):
    """Load data from file and prepare for HuggingFace format."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    en_text = normalizeString(parts[0])
                    de_text = normalizeString(parts[1])

                    if len(en_text.split()) < max_length and len(de_text.split()) < max_length:
                        pairs.append({"translation": {"en": en_text, "de": de_text}})

    return pairs


def split_data(pairs, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test."""
    import random

    random.shuffle(pairs)

    total = len(pairs)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size : train_size + val_size]
    test_pairs = pairs[train_size + val_size :]

    return train_pairs, val_pairs, test_pairs


def create_dataset(pairs):
    """Create HuggingFace Dataset from pairs."""
    return Dataset.from_list(pairs)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for HuggingFace Hub")
    parser.add_argument(
        "--input_file", type=str, default="german-english.txt", help="Input file path"
    )
    parser.add_argument("--max_length", type=int, default=64, help="Maximum sentence length")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="tuandunghcmut/translation-de-en-exercise",
        help="HuggingFace Hub repository ID",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--save_local", type=str, default="dataset_cache", help="Local directory to save dataset"
    )

    args = parser.parse_args()

    filepath = download_dataset_if_needed(args.input_file)
    pairs = load_and_prepare_data(filepath, args.max_length)

    train_pairs, val_pairs, test_pairs = split_data(pairs, args.train_ratio, args.val_ratio)

    train_dataset = create_dataset(train_pairs)
    val_dataset = create_dataset(val_pairs)
    test_dataset = create_dataset(test_pairs)

    dataset_dict = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    if args.save_local:
        dataset_dict.save_to_disk(args.save_local)

    if args.push_to_hub:
        dataset_dict.push_to_hub(args.repo_id)


if __name__ == "__main__":
    main()
