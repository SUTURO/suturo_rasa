import logging
import os
import sys
from pathlib import Path

import torch
from dataclasses import dataclass
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer
from matplotlib import pyplot as plt

# Utilizing logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


MODEL_NAME = "Qwen/Qwen3.5-4B"
# TRAIN_DATA = "train_data_complete.jsonl"
OUTPUT = os.path.join("fine_tune", MODEL_NAME)
OUTPUT_MERGED = os.path.join("fine_tune_merged", MODEL_NAME)

LORA_CONFIG = dict(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    use_rslora=True,
)

TRAIN_CONFIG = dict(
    max_length=2048,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    # warmup_steps=10,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    seed=42,
    logging_steps=5,
    dataset_text_field="text",
    eval_strategy="epoch",
)


@dataclass
class GPU:
    name: str
    vendor: str
    mem: float
    dtype: torch.dtype
    optim: str


def check_gpu() -> GPU:
    assert (
        torch.cuda.is_available()
    ), "No GPU found. Make sure CUDA or ROCm drivers are installed."

    name = torch.cuda.get_device_name()
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    is_amd = any(
        kw in name.upper() for kw in ("AMD", "RADEON", "INSTINCT", "VEGA", "NAVI")
    )

    if is_amd:
        gpu = GPU(
            name=name,
            vendor="amd",
            mem=mem,
            dtype=torch.bfloat16,
            optim="adamw_torch",
        )
    else:
        gpu = GPU(
            name=name,
            vendor="nvidia",
            mem=mem,
            dtype=torch.float16,
            optim="adamw_8bit",
        )

    log.info(f"GPU found: {gpu}")
    log.info(f"GPU mem: {gpu.mem:.1f} GB")
    log.info(f"GPU vendor: {gpu.vendor}")

    return gpu


def load_causal_language_model(gpu):
    log.info(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=gpu.dtype,
        device_map="auto",
    )

    log.info(f"Parameters: {model.num_parameters() / 1e6:.2f} M")
    return model, tokenizer


def add_LoRA(model):
    log.info(f"Adding LoRA")

    conf = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        **LORA_CONFIG,
    )

    model = get_peft_model(model, conf)
    model.print_trainable_parameters()

    return model


def prepare_dataset(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_data(data, tokenizer):
    log.info(f"Preparing data: {data}")

    dataset = load_dataset("json", data_files=str(data), split="train")
    dataset = dataset.map(
        lambda example: prepare_dataset(example, tokenizer),
        remove_columns=dataset.column_names,
    )
    # split dataset
    split = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    return split["train"], split["test"]


def train(model, tokenizer, train_data, eval_data, gpu):
    log.info("Starting training")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=SFTConfig(
            output_dir=OUTPUT,
            optim=gpu.optim,
            fp16=(gpu.dtype == torch.float16),
            bf16=(gpu.dtype == torch.bfloat16),
            **TRAIN_CONFIG,
        ),
    )

    trainer.train()
    return trainer


def create_plot(trainer):
    log_history = trainer.state.log_history

    train_losses = [ent["loss"] for ent in log_history if "loss" in ent]
    epoch_train = [ent["epoch"] for ent in log_history if "loss" in ent]
    eval_losses = [ent["eval_loss"] for ent in log_history if "eval_loss" in ent]
    epoch_eval = [ent["epoch"] for ent in log_history if "eval_loss" in ent]

    plt.plot(epoch_train, train_losses, label="Training Loss")
    if eval_losses:
        plt.plot(epoch_eval, eval_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    # plt.show()

    path = os.path.join(OUTPUT, "train-loss.png")
    plt.savefig(path, dpi=300)
    plt.close()

    log.info(f"Train Loss Plot saved in {path}")


def save(trainer, tokenizer):
    """Save LoRA adapter weights"""
    log.info(f"Saving LoRA & Tokenizer in: {OUTPUT}")

    trainer.model.save_pretrained(OUTPUT)
    tokenizer.save_pretrained(OUTPUT)

    log.info(f"Done.")


def save_merged(trainer, tokenizer):
    """Save merged model"""
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(OUTPUT_MERGED)
    tokenizer.save_pretrained(OUTPUT_MERGED)

    log.info(f"Merge complete.")
    log.info(f"Saved in {OUTPUT_MERGED}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", type=Path, help="Path to the Training Data File", required=True
    )
    args = parser.parse_args()
    try:
        gpu = check_gpu()
        model, tokenizer = load_causal_language_model(gpu)
        model = add_LoRA(model)
        train_data, eval_data = load_data(args.data, tokenizer)
        trainer = train(model, tokenizer, train_data, eval_data, gpu)

        save(trainer, tokenizer)
        create_plot(trainer)
        save_merged(trainer, tokenizer)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
