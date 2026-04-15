# Caps-2OMe: An interpretable multimodal deep learning framework for accurate RNA 2′-O-methylation site prediction

Caps-2OMe is a multimodal deep learning framework for identifying RNA 2′-O-methylation (2OMe) sites from sequence data.  
It integrates Chaos Game Representation (CGR), RNA-FM embeddings, and a capsule network to capture complementary spatial and contextual features for accurate and interpretable prediction.

## Key features
- **Multimodal feature learning**: combines CGR-based spatial encoding and RNA-FM-based contextual sequence representation.
- **Capsule-based classification**: uses dynamic routing to model hierarchical feature relationships and improve discrimination.
- **Interpretable design**: supports analysis of capsule routing patterns, prediction confidence, and motif-level biological signals.
- **Robust performance**: achieves strong results on both cross-validation and independent test sets.

## Framework
![Caps-2OMe framework](Workflow.png)
