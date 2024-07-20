# Neuro-Symbolic Sentiment Analysis with Dynamic Word Sense Disambiguation

This repository contains an implementation of the paper **"Neuro-Symbolic Sentiment Analysis with Dynamic Word Sense Disambiguation,"** written by **Xulang Zhang et al.** This work was accepted at **EMNLP 2023**, and you can access the paper [here](https://aclanthology.org/2023.findings-emnlp.587/).

## Current Status

- **Implementation**: Due to resource constraints, I could only perform simple experiments to validate the basic functionality of the implementation. The models have not yet been trained on publicly available datasets, and therefore, pre-trained weights are not available at this time. I plan to train the models on standard datasets as soon as I have access to a proper GPU.

- **Lexical Substitution Task**: The implementation uses **RoBERTa** instead of the pretrained ALM model introduced in the paper **"MetaPro 2.0: Computational Metaphor Processing on the Effectiveness of Anomalous Language Modeling,"** as the latter has not been published yet.

- **Dynamic Rewarding Strategy**: The dynamic rewarding strategy proposed in the paper is not implemented yet.

For any questions or feedback, feel free to open an issue in this repository.
