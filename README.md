# RAG Improvement via LoRA FineTuning

>[!NOTE]
This respiratory contains codes for Final project of 'Natural Language Processing` Course. Almost every code for this project is given as skeleton code.

## Overview

This project aims to **complete the default GPT-small → Summary/Classification → RAG pipeline** and then **improve the generator with LoRA (PEFT)** so that the final system produces **more fluent and consistent answers under a retrieval-augmented setting**—without paying the full cost of end-to-end fine-tuning.
- **Attribution (important)**: The LoRA fine-tuning pipeline is **adapted directly from the official Self-RAG code (`finetune.py`)** by Asai et al., with **minimal modifications** to fit our dataset, tokenizer, and training flags.  
     *We explicitly acknowledge Self-RAG as the source of the fine-tuning script design and implementation.* 

1. **Baseline completion (Default Project)**
 
   **1) Pretraining** : Implement a decoder-only GPT-small with RoPE, RMSNorm, GQA, and causal LM training.
   - Pretrain the model on a block-packed corpus; fine-tune on **summary** and **classification** tasks.
   - Used **`chengjunyan1/smollm-12.5-corpus` (subset: `cosmopedia-v2`)** for GPT pretraining, **`abisee/cnn_dailymail`** for summary fine-tuning, and **`SetFit/20_newsgroups`** for classification fine-tuning.

   **2) RAG Finetuning**: Build a **RAG** pipeline (sparse retriever + generator).
   - Finetune RAG on pretrained small-GPT model.
   - **Zeroshot RAG**: Finetune RAG on **Llama-3.2-1B-Instruct**.
   - Used **DPR-style Natural Questions Open (NQ-Open)** dataset.
   - In zero-shot RAG, we evaluate three prompt templates: **Original**, **Chain-of-Thought (CoT)**, and **Custom (My Prompt)**.

2. **LoRA-based Improvement of the Generator**

    - **Method : Use `finetune.py` in Self-RAG**  
      In Self-RAG method, there is step for finetuning model with **Instruction-following generation** task, using **LoRA**.  
      Used the same `finetune.py` script with minor modifications to fit our model configuration.
  
    - **Dataset**  
      Use same dataset as in Self-RAG finetuning.  
      Dataset consists of a collection of available sources, including **FLAN v2**, **ARC_Easy**, **NQ**, and others.  
      For fast training, randomly select **30%** of data (**45,000 instructions–output pair**) from original dataset.
  
      *(In `finetune.py`, LoRA is implemented by **PEFT** library.)*
   
      Example (dataset format):
      ``` 
      Instruction:
      Q: Is there a negative or positive tone to this product review?
      ...
      Response:
      [No Retrieval] Negative [Utility:5]
      ```


For further information, please refer to **`25_06_09 NLP Presentation.pdf`**, and **`NLP_Final_Project_20201200.pdf`**.

## **Usage Instructions**

All training/evaluation steps are orchestrated in **`main.ipynb`**.

1) **Set paths** in the first cell:
```python
DRIVE_CACHE_PATH = "..."   # data/index cache
LOG_PATH         = "..."   # logs
OUTPUT_PATH      = "..."   # checkpoints / final models
RESULTS_PATH     = "..."   # metrics / text outputs
```

Also, you can change the variables for training.

2) **Training flags** (toggled in `main.ipynb`):
```python

DO_PRETRAIN # Do pretrain small-GPT model.
DO_FINETUNE_SM # Finetuning pretrained model for summary task.
DO_FINETUNE_CF # Finetuning pretrained model for classification task.
DO_FINETUNE_RAG # Finetune RAG on pretrained model.
DO_ZEROSHOT_RAG # Do RAG finetuning on Llama-3.2-1B-Instruct model.
DO_PRE_FINETUNED_RAG # Do RAG finetuning on LoRA finetuned model.
DO_SUBMISSION # Do Compression for submission
DO_SUMMARY_LORA_FINETUNE # LoRA finetuning on Summary finetuned model.
DO_RAG_LORA_FINETUNE # Do RAG finetuning, then do LoRA finetuning.
DO_ZEROSHOT_RAG_PROMPT # Do LoRA finetuning on Llama-3.2-1B-Instruct model.

```

3) **Typical run order**
```
[1] DO_PRETRAIN
[2] DO_FINETUNE_SM / DO_FINETUNE_CF
[3] DO_FINETUNE_RAG
[4] DO_ZEROSHOT_RAG
[5] DO_SUMMARY_LORA_FINETUNE
[6] DO_PRE_FINETUNED_RAG
```

## Experiments

- Evaluate ROGUE/Accuracy(EM) score of each model.

### 1) Default Project (GPT-small)
- Pipeline: **Pretrain → Summary/Classification fine-tune**.

### 2) RAG (GPT-small)
- Setup: **BM25 retriever + GPT-small generator**.

### 3) LoRA Fine-Tuning
- **Summary + LoRA**: Improves ROUGE and reduces repetition, resulting more natural answer.
- **RAG -> LoRA**: Improves ROUGE, and **EM** often improves.
- **LoRA -> RAG** : Improves ROUGE score, but **EM** is decreased.


