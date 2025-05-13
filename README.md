# Neural-machine-translation-with-a-Transformer
# Arabic-English Neural Machine Translation

This repository contains code and experiments for building and evaluating Neural Machine Translation (NMT) models for translating text from Arabic to English. We explore training a custom Transformer model from scratch on two different datasets and fine-tuning a pre-trained MBart model.

Our Reference to build a custom Transformer model from scratch:  [https://www.tensorflow.org/text/tutorials/transformer](https://www.tensorflow.org/text/tutorials/transformer)

[Our Presentation Link](https://gamma.app/docs/Arabic-English-Neural-Machine-Translation-using-Transformer-3zdxhjuta1hld8f) 
## Datasets

Two parallel corpora were used for training and evaluation:

1.  **Smaller Dataset:**
    * Source: GitHub Dataset (`SamirMoustafa/nmt-with-attention-for-ar-to-en`)
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
    * Training Data: Smaller Dataset (GitHub).
    * Approach: Trained from scratch using TensorFlow/Keras.
    * Key Observation: Achieved moderate performance on this dataset, showing the potential of the Transformer architecture on smaller data.
    * [Notebook Link](https://colab.research.google.com/drive/1_6JOFjpYvalwfpG7UlW77r0yUW6CK_7F?usp=sharing)

2.  **Custom Transformer (Trained on Larger Dataset):**
    * Architecture: Same custom Transformer model.
    * Training Data: Larger Dataset (Kaggle).
    * Approach: Trained from scratch using TensorFlow/Keras.
    * Key Observation: Struggled significantly with the increased complexity and size of the larger dataset when trained from scratch, resulting in lower performance metrics compared to the smaller dataset.
    * [Notebook Link](https://colab.research.google.com/drive/1CFkn5e8aHyzrqel5RbzaSyAk_Af7OJKq?usp=sharing)

3.  **Pre-trained MBart (Fine-tuned on Larger Dataset):** "From English to Arabic"
    * Architecture: Fine-tuned `facebook/mbart-large-50-many-to-many-mmt` from Hugging Face Transformers.
    * Training Data: Larger Dataset (Kaggle).
    * Approach: Fine-tuning a large pre-trained multilingual model using Hugging Face `Seq2SeqTrainer`.
    * Key Observation: Expected to perform significantly better than the custom model on the larger dataset due to transfer learning from massive pre-training. Converged faster in terms of epochs.
    * [Notebook Link](https://colab.research.google.com/drive/1cSEgvEQO-jQmnculkoiTahavuQgiExsZ?usp=sharing)

## Results Summary

Here is a brief comparison of the final training outcomes:

| Feature             | Custom Transformer (GitHub) | Custom Transformer (Kaggle) | Pre-trained MBart (Kaggle) |
| :------------------ | :-------------------------- | :-------------------------- | :------------------------- |
| **Dataset Size** | Smaller (~10k pairs)        | Larger (~24k pairs)         | Larger (~24k pairs)        |
| **Model Type** | From Scratch                | From Scratch                | Fine-tuned Pre-trained     |
| **Epochs** | 60                          | 45                          | 4                          |
| **Total Train Time**| ~10.7 min                   | ~2.4 hours                  | ~3.4 hours                 |
| **Final Train Loss**| 0.1164                      | 1.2217                      | **0.2109** |
| **Final Test BLEU** | 0.0391 | 0.0049                      | **43.117** |

