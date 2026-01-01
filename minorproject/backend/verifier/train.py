"""QLoRA fine-tuning loop for the FaithForge faithfulness verifier.

Fine-tune a 1-3B parameter instruction model (e.g., Qwen2.5-1.5B-Instruct) using
QLoRA (4-bit quantization + LoRA adapters) via HuggingFace PEFT. The verifier is
trained as an NLI-style classifier: given (claim, evidence) → (entailment_label, faithfulness_score).

Training data format (JSONL):
    {
        "claim": "string — the claim text",
        "evidence": "string — the evidence passage",
        "label": "entailment" | "contradiction" | "neutral",
        "faithfulness_score": float (0.0 to 1.0)
    }
"""

import json
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.logging import get_logger

logger = get_logger("faithforge.verifier.train")

# NLI input/output format for the verifier
INPUT_TEMPLATE = "Claim: {claim}\nEvidence: {evidence}"
TARGET_TEMPLATE = "{label}\nScore: {score:.2f}"


def load_training_data(data_path: Path, tokenizer, max_seq_length: int = 512) -> Dataset:
    """Load and preprocess training data from JSONL.

    Formats each example as an NLI pair:
        Input:  "Claim: {claim}\nEvidence: {evidence}"
        Target: "{label}\nScore: {score:.2f}"

    Args:
        data_path: Path to the JSONL training data file.
        tokenizer: The model's tokenizer for encoding.
        max_seq_length: Maximum sequence length for truncation.

    Returns:
        HuggingFace Dataset with tokenized input_ids, attention_mask, and labels.
    """
    logger.info("Loading training data from %s", data_path)

    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(row)

    logger.info("Loaded %d examples", len(examples))

    # Format as NLI pairs
    input_texts = []
    target_texts = []

    for ex in examples:
        input_text = INPUT_TEMPLATE.format(
            claim=ex["claim"],
            evidence=ex["evidence"][:1000],  # Truncate long evidence
        )
        target_text = TARGET_TEMPLATE.format(
            label=ex["label"],
            score=ex["faithfulness_score"],
        )
        input_texts.append(input_text)
        target_texts.append(target_text)

    # Tokenize
    def tokenize_fn(examples):
        # Combine input + target for causal LM training
        full_texts = [
            inp + "\n" + tgt
            for inp, tgt in zip(examples["input_text"], examples["target_text"])
        ]

        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )

        # Create labels (same as input_ids for causal LM, with padding masked)
        tokenized["labels"] = tokenized["input_ids"].copy()

        # Mask padding tokens in labels
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        for i in range(len(tokenized["labels"])):
            tokenized["labels"][i] = [
                -100 if token_id == pad_token_id else token_id
                for token_id in tokenized["labels"][i]
            ]

        return tokenized

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "input_text": input_texts,
        "target_text": target_texts,
    })

    # Tokenize in batches
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["input_text", "target_text"],
    )

    logger.info("Tokenized dataset: %d examples", len(tokenized_dataset))
    return tokenized_dataset


def create_lora_config():
    """Create the LoRA/PEFT configuration for the verifier.

    Returns:
        peft.LoraConfig with parameters suitable for a 1.5B model.

    Config:
        - r=16: LoRA rank (balances capacity vs. efficiency)
        - lora_alpha=32: scaling factor (2x rank for stable training)
        - target_modules=["q_proj", "v_proj"]: attention projection layers
        - lora_dropout=0.05: regularization
        - task_type="CAUSAL_LM": causal language modeling
    """
    from peft import LoraConfig

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none",
    )

    logger.info("LoRA config: r=%d, alpha=%d, targets=%s", config.r, config.lora_alpha, config.target_modules)
    return config


def train(
    base_model: str,
    train_data: Dataset,
    val_data: Dataset,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
) -> dict:
    """Run the QLoRA fine-tuning loop.

    Steps:
    1. Load base model in 4-bit (NF4 quantization via bitsandbytes)
    2. Apply LoRA adapters via PEFT
    3. Train with HuggingFace Trainer
    4. Save adapter weights + tokenizer
    5. Return training metrics

    Args:
        base_model: HuggingFace model ID (e.g., "Qwen/Qwen2.5-1.5B-Instruct").
        train_data: Tokenized training dataset.
        val_data: Tokenized validation dataset.
        output_dir: Where to save the trained adapter.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Learning rate for AdamW.
        max_seq_length: Maximum sequence length.

    Returns:
        Dict with training metrics.
    """
    from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
    from peft import get_peft_model, prepare_model_for_kbit_training

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting QLoRA training: base=%s, epochs=%d, batch=%d, lr=%s",
        base_model, epochs, batch_size, learning_rate,
    )

    # Step 1: Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Load base model in 4-bit
    logger.info("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Step 3: Prepare for k-bit training and apply LoRA
    logger.info("Applying LoRA adapters...")
    model = prepare_model_for_kbit_training(model)
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d / %d (%.2f%%)",
        trainable_params, total_params, 100 * trainable_params / total_params,
    )

    # Step 4: Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_accumulation_steps=4,
        report_to="none",  # Set to "wandb" for experiment tracking
        remove_unused_columns=False,
    )

    # Step 5: Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    # Step 6: Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Step 7: Save
    logger.info("Saving model to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Collect metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "epochs_completed": int(train_result.metrics.get("epoch", epochs)),
    }

    # Evaluate
    eval_result = trainer.evaluate()
    metrics["val_loss"] = eval_result.get("eval_loss", 0.0)

    logger.info("Training complete: %s", metrics)
    return metrics


def evaluate_on_ragtruth(
    model_path: Path,
    ragtruth_data: Dataset,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> dict:
    """Evaluate the fine-tuned verifier on the RAGTruth benchmark.

    Loads the base model + LoRA adapter and runs inference on each
    (claim, evidence) pair in the RAGTruth test set.

    Args:
        model_path: Path to the saved LoRA adapter.
        ragtruth_data: RAGTruth test split (HuggingFace Dataset).
        base_model: The base model ID used for training.

    Returns:
        Dict with evaluation metrics.
    """
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logger.info("Evaluating on RAGTruth: adapter=%s", model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base, str(model_path))
    model.eval()

    # Run inference
    predictions = []
    true_labels = []

    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}

    for example in ragtruth_data:
        input_text = INPUT_TEMPLATE.format(
            claim=example["claim"],
            evidence=example["evidence"][:1000],
        )

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only the new tokens
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated = generated.strip().lower()

        # Parse prediction
        pred_label = "neutral"  # default
        for label in label_map:
            if label in generated:
                pred_label = label
                break

        predictions.append(label_map.get(pred_label, 2))
        true_labels.append(label_map.get(example["label"], 2))

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted", zero_division=0,
    )

    # Per-label accuracy
    per_label = {}
    for label_name, label_idx in label_map.items():
        mask = [t == label_idx for t in true_labels]
        if any(mask):
            correct = sum(1 for p, t in zip(predictions, true_labels) if t == label_idx and p == t)
            per_label[label_name] = correct / sum(mask)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_label_accuracy": per_label,
        "total_examples": len(ragtruth_data),
    }

    logger.info("RAGTruth evaluation: %s", metrics)
    return metrics
