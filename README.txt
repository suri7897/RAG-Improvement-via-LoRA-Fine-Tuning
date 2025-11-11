NLP Final Project : Default project & improvement by LoRA / 20201200 KangJun Lee (이강준)

This is README.txt file for Final Project code implementation.

Almost every training is done in main.ipynb.

All training is done using A6000 GPU in ssh server using visual studio code environment. (Not Google Colab)

So, Path location should be modified for implementation (DRIVE_CACHE_PATH, LOCAL_CACHE_PATH etc..),
since it is modified from the first draft to fit in my environment.

Also, there are added arguments, DO_PRE_FINETUNED_RAG, DO_SUMMARY_LORA_FINETUNE, DO_RAG_LORA_FINETUNE.

DO_PRE_FINETUNED_RAG : True -> Do LoRA finetuning at pretrained model (trained in DO_PRETRAIN), and then finetune rag to that model.
DO_SUMMARY_LORA_FINETUNE : True -> Do LoRA finetuning at summary model (trained in DO_FINETUNE_SM), and then evaluate.
DO_RAG_LORA_FINETUNE : True -> Do LoRA finetuning at rag model (trained in DO_FINETUNE_RAG), and then evaluate.

To finetune, finetune.py is needed. This code is for finetuning gpt-model using lora.
Original code is from self-rag paper by Akari Asai, et al. (https://arxiv.org/abs/2310.11511), and slightly modified. 
(github : https://github.com/AkariAsai/self-rag.git)

Arguments of this finetune.py is 
python finetune.py   
--train_file train.jsonl ..
--model_name_or_path ..
--tokenizer_name ..
--output_dir ..
--use_lora   
--use_special_tokens   
--num_train_epochs ..
--per_device_train_batch_size ..   
--gradient_accumulation_steps  ..
--learning_rate ..
--max_seq_length ..
--overwrite_cache   
--low_cpu_mem_usage

(For actual implementation, refer to implementation code in main.ipynb).

Training data for finetune.py is train.jsonl, which is also from the self-rag paper.
Training data is 30% sliced data from original, for fast training.

Output is in the defaultproject_supplementaries.zip file, including best model and log, result for all cases.

Also, there was slight change in original code for implementation. (Such as collate_fn in summary.py)