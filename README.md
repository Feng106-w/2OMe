# 🧬 Caps-2OMe: An interpretable multimodal deep learning framework for accurate RNA 2′-O-methylation site prediction

This repository provides the implementation of Caps-2OMe, a deep learning framework for RNA 2′-O-methylation site prediction. The model fuses CGR and RNA-FM features and uses a capsule network for classification and interpretability analysis.

## ✨ Key features
- **Multimodal feature learning**: combines CGR-based spatial encoding and RNA-FM-based contextual sequence representation.
- **Capsule-based classification**: uses dynamic routing to model hierarchical feature relationships and improve discrimination.
- **Interpretable design**: supports analysis of capsule routing patterns, prediction confidence, and motif-level biological signals.
- **Robust performance**: achieves strong results on both cross-validation and independent test sets.

## 🧩 Framework
![Caps-2OMe framework](Workflow.png)

## 📁 Repository Structure

```
Caps-2OMe/
├── data/
│   ├── fused_dataset.py
│   ├── Benchmark_Set.fasta
│   ├── Independent_Test_Set.fasta
│   ├── Benchmark_Set8.txt
│   └── Independent_Test_Set8.txt
├── models/
│   ├── fusion_frontend.py
│   └── capsnet_8x8.py
├── train.py
```

## ⚙️ Installation

To get started, first clone the repository to your local machine and install the required Python packages:

```
git clone https://github.com/Feng106-w/2OMe.git
cd Caps-2OMe
pip install -r requirements.txt
```

Then, download the pretrained RNA-FM and place it under the specified folder.
```
models_folder/RNA-FM_pretrained.pth
```

## 🚀 Training

```
python train_caps_fusion.py \
  --train_fasta data/Benchmark_Set.fasta \
  --test_fasta data/Independent_Test_Set.fasta \
  --train_cgr data/Benchmark_Set8.txt \
  --test_cgr data/Independent_Test_Set8.txt \
  --pretrained_dir models_folder \
  --batch_size 64 \
  --epochs 100 \
  --caps_nums 8
```

