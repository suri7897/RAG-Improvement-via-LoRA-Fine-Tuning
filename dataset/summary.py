import os
import datasets
import itertools
import functools
import torch


def prepare_summary_dataset(tokenizer,
                             dataset_name_or_path: str = "abisee/cnn_dailymail",
                             dataset_subset: str = "3.0.0",
                             context_max_length: int = 896,
                             target_max_length: int = 128,
                             context_column_name: str = "article",
                             target_column_name: str = "highlights",
                             prompt: str = "Context: {context}\n Summary:\n",
                             train_sample_size: int =-1,
                             cache_path:str="cache"):
    if cache_path is not None and os.path.exists(os.path.join(cache_path,"summary")):
        print(f"Using pre-downloaded dataset from {cache_path}.")
        train = datasets.load_from_disk(os.path.join(cache_path,"summary","train"))
        validation = datasets.load_from_disk(os.path.join(cache_path,"summary","eval"))
        return train, validation
    else:
        print(f"Download and prerpocessing dataset from {dataset_name_or_path} on subset {dataset_subset}...")
        dataset = datasets.load_dataset(dataset_name_or_path, dataset_subset)
        if "eval" in dataset:
            validation_split = "eval"
        elif "test" in dataset:
            validation_split = "test"
        else:
            print("Using train split for validation.")
            dataset = datasets.train_test_split(test_size=0.1)
            validation_split = "test"
        
        train = dataset["train"] if train_sample_size == -1 else dataset["train"].select(range(train_sample_size))
        validation = dataset[validation_split]

        processing_lambda = functools.partial(
            _preprocess,
            tokenizer=tokenizer,
            context_column_name=context_column_name,
            target_column_name=target_column_name,
            context_max_length=context_max_length,
            target_max_length=target_max_length,
            prompt=prompt
        )
        train = train.map(
            processing_lambda,
            batched=True,
            remove_columns=train.column_names,
            num_proc=1,
            batch_size=64,
            desc="Preprocessing",
        )
        validation = validation.map(
            processing_lambda,
            batched=True,
            remove_columns=validation.column_names,
            num_proc=1,
            batch_size=64,
            desc="Preprocessing",
        )

        if cache_path is not None:
            os.makedirs(os.path.join(cache_path,"summary"), exist_ok=True)
            print(f"Saving dataset to {cache_path}...")
            train.save_to_disk(os.path.join(cache_path,"summary","train"), max_shard_size="500MB")
            validation.save_to_disk(os.path.join(cache_path,"summary","eval"), max_shard_size="500MB")
        return train, validation

def _preprocess(examples, tokenizer, context_column_name, target_column_name, context_max_length, target_max_length, prompt):
    contexts = tokenizer(examples[context_column_name], add_special_tokens=False, truncation=True, max_length=context_max_length-10)
    targets = tokenizer(examples[target_column_name], add_special_tokens=False, truncation=True, max_length=target_max_length)
    contexts = tokenizer.batch_decode(contexts["input_ids"], skip_special_tokens=True)
    targets = tokenizer.batch_decode(targets["input_ids"], skip_special_tokens=True)

    #! fill here 
    ######!
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for context, target in zip(contexts, targets):
        prompt_with_context = prompt.format(context=context)

        model_input = tokenizer(
            prompt_with_context,
            max_length=context_max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        label = tokenizer(
            target,
            max_length=target_max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        input_ids = model_input["input_ids"].squeeze(0)
        attention_mask = model_input["attention_mask"].squeeze(0)
        label_ids = label["input_ids"].squeeze(0)

        labels = torch.full_like(input_ids, -100)
        target_len = min(len(target_ids), len(input_ids))
        labels[-target_len:] = target_ids[-target_len:]  

        input_ids_list.append(input_ids.tolist())
        attention_mask_list.append(attention_mask.tolist())
        labels_list.append(labels.tolist())

    inputs = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

    ######!
    return inputs

def collate_fn_for_summary(batch, tokenizer, pad_to_multiple_of=64):
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    labels = [example["labels"] for example in batch]

    max_len = max(len(seq) for seq in input_ids)
    if pad_to_multiple_of:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    def pad(seq, pad_val): return seq + [pad_val] * (max_len - len(seq))

    input_ids = [pad(seq, tokenizer.pad_token_id) for seq in input_ids]
    attention_mask = [pad(seq, 0) for seq in attention_mask]
    labels = [pad(seq, -100) for seq in labels] 
    labels = [
        [label if mask == 1 else -100 for label, mask in zip(label_seq, mask_seq)]
        for label_seq, mask_seq in zip(labels, attention_mask)
    ]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }
