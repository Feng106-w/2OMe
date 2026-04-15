import argparse
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from Bio import SeqIO
import fm
from torch.serialization import add_safe_globals

add_safe_globals([argparse.Namespace])

TARGET_LEN = 41

def _load_cgr_txt(txt_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Read CGR txt file and return:
      - acc2img: accession -> [H, W] float32 CGR image
      - acc2lab: accession -> int label (pos=1, neg=0)
    """
    acc2img: Dict[str, np.ndarray] = {}
    acc2lab: Dict[str, int] = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")

            # Compatible with lines where the "figure" field itself may contain commas
            if len(parts) > 3:
                figure = ",".join(parts[:-2])
                label = parts[-2]
                acc = parts[-1]
            else:
                figure, label, acc = parts[0], parts[1], parts[2]

            vec = np.array(list(map(float, figure.strip().split())), dtype=np.float32)
            side = int(np.sqrt(len(vec)))
            img = vec.reshape(side, side).astype(np.float32)

            acc2img[acc] = img
            acc2lab[acc] = 1 if label.strip().lower().startswith("pos") else 0

    return acc2img, acc2lab


_FM_SINGLETON = {"model": None, "alphabet": None, "device": None}
FM_CACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()
MAX_CACHE = None  # set an int if you want LRU cache limit


def _ensure_fm_loaded(pretrained_dir: str, device: str):
    if _FM_SINGLETON["model"] is not None:
        return

    pth = Path(pretrained_dir, "RNA-FM_pretrained.pth")
    fm_model, alphabet = fm.pretrained.rna_fm_t12(pth)

    if torch.cuda.is_available() and device.startswith("cuda") and torch.cuda.device_count() > 1:
        fm_model = nn.DataParallel(fm_model)

    fm_model.to(device).eval()
    _FM_SINGLETON["model"] = fm_model
    _FM_SINGLETON["alphabet"] = alphabet
    _FM_SINGLETON["device"] = device


def _embed_batch(names: List[str], seqs: List[str]) -> List[torch.Tensor]:
    """
    Compute RNA-FM embeddings for a batch of sequences.
    Returns list of [TARGET_LEN, D] float32 tensors.
    """
    miss_indices = []
    out_list: List[torch.Tensor] = [None] * len(names)  # type: ignore

    for i, nm in enumerate(names):
        if nm in FM_CACHE:
            val = FM_CACHE.pop(nm)
            FM_CACHE[nm] = val
            out_list[i] = val.to(torch.float32)
        else:
            miss_indices.append(i)

    if not miss_indices:
        return out_list

    fm_model = _FM_SINGLETON["model"]
    alphabet = _FM_SINGLETON["alphabet"]
    device = _FM_SINGLETON["device"]
    batch_converter = alphabet.get_batch_converter()

    miss_names = [names[i] for i in miss_indices]
    miss_seqs = [seqs[i] for i in miss_indices]
    _, _, toks = batch_converter(list(zip(miss_names, miss_seqs)))

    try_cuda = device.startswith("cuda") and torch.cuda.is_available()

    try:
        with torch.no_grad():
            rep = fm_model(
                toks.to(device if try_cuda else "cpu"),
                repr_layers=[12]
            )["representations"][12]

        reps = []
        with torch.no_grad():
            for b in range(rep.shape[0]):
                t = rep[b]  # [L, D]
                t = F.interpolate(
                    t.permute(1, 0).unsqueeze(0),   # [1, D, L]
                    size=TARGET_LEN,
                    mode="linear",
                    align_corners=False
                ).squeeze(0).permute(1, 0)         # [TARGET_LEN, D]
                reps.append(t.detach().cpu().to(torch.float16).contiguous())

        if try_cuda:
            torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            with torch.no_grad():
                rep = fm_model(
                    toks.to("cpu"),
                    repr_layers=[12]
                )["representations"][12]

            reps = []
            for b in range(rep.shape[0]):
                t = rep[b]
                t = F.interpolate(
                    t.permute(1, 0).unsqueeze(0),
                    size=TARGET_LEN,
                    mode="linear",
                    align_corners=False
                ).squeeze(0).permute(1, 0)
                reps.append(t.detach().to(torch.float16).contiguous())
        else:
            raise

    for loc, tens_f16 in zip(miss_indices, reps):
        nm = names[loc]
        if (MAX_CACHE is not None) and (len(FM_CACHE) >= MAX_CACHE):
            FM_CACHE.popitem(last=False)
        FM_CACHE[nm] = tens_f16
        out_list[loc] = tens_f16.to(torch.float32)

    return out_list


class FusedCapsDataset(Dataset):
    def __init__(
        self,
        fasta_path: str,
        cgr_txt_path: str,
        pretrained_dir: str = "models_folder",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device

        recs = list(SeqIO.parse(fasta_path, "fasta"))
        self.names = [r.id for r in recs]
        self.seqs = [str(r.seq) for r in recs]

        acc2img, acc2lab = _load_cgr_txt(cgr_txt_path)

        self.cgr_list: List[torch.Tensor] = []
        self.labels: List[int] = []

        for acc in self.names:
            img = torch.from_numpy(acc2img[acc]).unsqueeze(0)  # [1, H, W]
            self.cgr_list.append(img)
            self.labels.append(acc2lab[acc])

        _ensure_fm_loaded(pretrained_dir, device)

        print(
            f"[FusedCapsDataset] N={len(self.labels)}; "
            f"CGR shape={tuple(self.cgr_list[0].shape)}"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        name = self.names[i]
        seq = self.seqs[i]
        cgr = self.cgr_list[i].to(torch.float32)
        y = torch.tensor(float(self.labels[i]), dtype=torch.float32)
        return (name, seq, cgr), y

    def get_all_features_labels(self):
        X_index = np.arange(len(self.labels))
        y = np.array(self.labels, dtype=np.int64)
        return X_index, y


def fm_collate(batch: List[Tuple[Tuple[str, str, torch.Tensor], torch.Tensor]]):
    names, seqs, cgrs, labels = [], [], [], []
    for (name, seq, cgr), y in batch:
        names.append(name)
        seqs.append(seq)
        cgrs.append(cgr)
        labels.append(y)

    fm_list = _embed_batch(names, seqs)
    fm_batch = torch.stack(fm_list, dim=0).to(torch.float32)  # [B, 41, D]
    cgr_batch = torch.stack(cgrs, dim=0)                      # [B, 1, H, W]
    y_batch = torch.stack(labels, dim=0)                      # [B]

    return (fm_batch, cgr_batch), y_batch