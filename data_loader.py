"""
Data loading and processing using Hugging Face datasets and torch DataLoader.
"""

import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

from utils import Lang, normalizeString, filterPair, SOS_token, EOS_token

logger = logging.getLogger(__name__)


def download_dataset_if_needed(filepath="data/german-english.txt"):
    """Download dataset from Google Drive if not present."""
    if not os.path.exists(filepath):
        import gdown

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        url = "https://drive.google.com/uc?id=1CVMenH5xgDsiq9ZaGAFfkFLG5uWDZvdL"
        gdown.download(url, filepath, quiet=False)
    return filepath


class TranslationDataset(Dataset):
    """Custom dataset for translation pairs."""

    def __init__(self, hf_dataset, input_lang, output_lang, max_length=64):
        self.data = []
        self.input_lang = input_lang
        self.output_lang = output_lang

        for item in hf_dataset:
            # Input: German, Output: English
            src_text = normalizeString(item["translation"]["de"])
            tgt_text = normalizeString(item["translation"]["en"])

            pair = [src_text, tgt_text]
            if filterPair(pair, max_length):
                self.data.append(pair)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]

        src_indexes = [self.input_lang.word2index.get(word, 0) for word in pair[0].split(" ")]
        src_indexes.append(EOS_token)

        tgt_indexes = [self.output_lang.word2index.get(word, 0) for word in pair[1].split(" ")]
        tgt_indexes.append(EOS_token)

        return torch.tensor(src_indexes, dtype=torch.long), torch.tensor(
            tgt_indexes, dtype=torch.long
        )


def collate_fn(batch):
    """Collate function to pad sequences in a batch."""
    src_batch, tgt_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_batch, tgt_batch


def build_vocab(hf_dataset, lang_name, max_length=64):
    """Build vocabulary from Hugging Face dataset."""
    lang = Lang(lang_name)

    # Map language names to dataset keys
    key = "de" if lang_name == "ger" else "en"

    for item in hf_dataset:
        text = normalizeString(item["translation"][key])
        lang.addSentence(text)

    return lang


def prepareData(
    data_source="local",
    lang1="ger",
    lang2="eng",
    reverse=False,
    max_length=64,
    train_ratio=0.8,
    val_ratio=0.1,
    batch_size=32,
    num_workers=0,
    seed=42,
):
    """
    Load data from Hugging Face datasets and prepare train/val/test splits.

    Args:
        data_source: 'tatoeba' or 'local' (for german-english.txt)
        lang1: First language code (German)
        lang2: Second language code (English)
        reverse: Whether to reverse the language pair
        max_length: Maximum sentence length for filtering
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        seed: Random seed for reproducible data splits and vocabulary (default: 42)

    Returns:
        input_lang, output_lang, train_loader, val_loader, test_loader
    """

    if data_source == "local":
        from datasets import Dataset as HFDataset

        filepath = download_dataset_if_needed("data/german-english.txt")

        pairs = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        # Reverse: German (parts[1]) -> English (parts[0])
                        pairs.append({"translation": {"de": parts[1], "en": parts[0]}})

        import random

        # FIXED: Use deterministic seed for reproducible vocabulary
        random.seed(seed)
        random.shuffle(pairs)

        total = len(pairs)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_data = HFDataset.from_list(pairs[:train_size])
        val_data = HFDataset.from_list(pairs[train_size : train_size + val_size])
        test_data = HFDataset.from_list(pairs[train_size + val_size :])
    else:
        dataset = load_dataset(data_source, f"{lang2}-{lang1}")

        if "train" in dataset:
            train_val = dataset["train"].train_test_split(
                test_size=(val_ratio + (1 - train_ratio - val_ratio)), seed=42
            )
            train_data = train_val["train"]

            val_test = train_val["test"].train_test_split(
                test_size=(1 - train_ratio - val_ratio)
                / (val_ratio + (1 - train_ratio - val_ratio)),
                seed=42,
            )
            val_data = val_test["train"]
            test_data = val_test["test"]
        else:
            train_data = dataset["train"] if "train" in dataset else dataset
            val_data = dataset["validation"] if "validation" in dataset else dataset
            test_data = dataset["test"] if "test" in dataset else dataset

    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    # Build vocabulary from ALL data (train + val + test) to avoid unknown words
    # Concatenate all datasets for complete vocabulary coverage
    from datasets import concatenate_datasets

    all_data = concatenate_datasets([train_data, val_data, test_data])
    temp_input_lang = build_vocab(all_data, lang1 if not reverse else lang2, max_length)
    temp_output_lang = build_vocab(all_data, lang2 if not reverse else lang1, max_length)

    input_lang.word2index = temp_input_lang.word2index
    input_lang.word2count = temp_input_lang.word2count
    input_lang.index2word = temp_input_lang.index2word
    input_lang.n_words = temp_input_lang.n_words

    output_lang.word2index = temp_output_lang.word2index
    output_lang.word2count = temp_output_lang.word2count
    output_lang.index2word = temp_output_lang.index2word
    output_lang.n_words = temp_output_lang.n_words

    train_dataset = TranslationDataset(train_data, input_lang, output_lang, max_length)
    val_dataset = TranslationDataset(val_data, input_lang, output_lang, max_length)
    test_dataset = TranslationDataset(test_data, input_lang, output_lang, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return input_lang, output_lang, train_loader, val_loader, test_loader
