# Neural-machine-translation-with-a-Transformer
# Arabic-English Neural Machine Translation

This repository contains code and experiments for building and evaluating Neural Machine Translation (NMT) models for translating text from Arabic to English. We explore training a custom Transformer model from scratch on two different datasets and fine-tuning a pre-trained MBart model.

## Datasets

Two parallel corpora were used for training and evaluation:

1.  **Smaller Dataset:**
    * Source: `ara_eng.txt`
    * Link: [https://raw.githubusercontent.com/SamirMoustafa/nmt-with-attention-for-ar-to-en/master/ara_.txt](https://raw.githubusercontent.com/SamirMoustafa/nmt-with-attention-for-ar-to-en/master/ara_.txt)
    * Size (after filtering): **10742** sentence pairs (8593 Train, 2149 Test).
    * Characteristics: Relatively simpler sentences, smaller vocabulary, shorter sequence lengths.

2.  **Larger Dataset:**
    * Source: Kaggle Dataset (`samirmoustafa/arabic-to-english-translation-sentences`)
    * Link: [https://www.kaggle.com/datasets/samirmoustafa/arabic-to-english-translation-sentences](https://www.kaggle.com/datasets/samirmoustafa/arabic-to-english-translation-sentences)
    * Size (after filtering): **24638** sentence pairs (19710 Train, 4928 Test).
    * Characteristics: More complex sentences, larger vocabulary, longer sequence lengths.

## Models and Experiments

We conducted experiments with three different model scenarios:

1.  **Custom Transformer (Trained on Smaller Dataset):**
    * Architecture: Custom implementation of the Transformer model (Encoder-Decoder).
    * Training Data: Smaller Dataset (GitHub `ara_eng.txt`).
    * Approach: Trained from scratch using TensorFlow/Keras.
    * Key Observation: Achieved moderate performance on this dataset, showing the potential of the Transformer architecture on smaller data.

2.  **Custom Transformer (Trained on Larger Dataset):**
    * Architecture: Same custom Transformer model.
    * Training Data: Larger Dataset (Kaggle).
    * Approach: Trained from scratch using TensorFlow/Keras.
    * Key Observation: Struggled significantly with the increased complexity and size of the larger dataset when trained from scratch, resulting in lower performance metrics compared to the smaller dataset.

3.  **Pre-trained MBart (Fine-tuned on Larger Dataset):**
    * Architecture: Fine-tuned `facebook/mbart-large-50-many-to-many-mmt` from Hugging Face Transformers.
    * Training Data: Larger Dataset (Kaggle).
    * Approach: Fine-tuning a large pre-trained multilingual model using Hugging Face `Seq2SeqTrainer`.
    * Key Observation: Expected to perform significantly better than the custom model on the larger dataset due to transfer learning from massive pre-training. Converged faster in terms of epochs.

## Results Summary

Here is a brief comparison of the final training outcomes:

| Feature             | Custom Transformer (GitHub) | Custom Transformer (Kaggle) | Pre-trained MBart (Kaggle) |
| :------------------ | :-------------------------- | :-------------------------- | :------------------------- |
| **Dataset Size** | Smaller (~10k pairs)        | Larger (~24k pairs)         | Larger (~24k pairs)        |
| **Model Type** | From Scratch                | From Scratch                | Fine-tuned Pre-trained     |
| **Epochs** | 60                          | 45                          | 4                          |
| **Total Train Time**| ~10.7 min                   | ~2.4 hours                  | ~3.4 hours                 |
| **Final Train Loss**| 0.1164                      | 1.2217                      | **0.2109** |
| **Final Test BLEU** | **0.0391** | 0.0049                      | *Not available in logs* |

*Note: Final test metrics for the MBart model were not available in the provided logs, but a pre-trained model is expected to achieve a significantly higher BLEU score than models trained from scratch on this dataset size.*

## Code Structure

*(Add a brief description here of how the code is organized in your repository, e.g., `data_preprocessing.py`, `transformer_model.py`, `mbart_finetuning.py`, `inference.py`, etc.)*

## Getting Started

*(Add instructions on how to set up the environment, download data, and run the code)*
