"""
05_train_hf.py — QLoRA fine-tuning with PEFT + TRL (RunPod / NVIDIA GPU).

Usage:
    python scripts/05_train_hf.py --config configs/training_p2c_mistral7b.yaml

Saves adapter checkpoints to adapter_path/checkpoint-N/ every eval_steps,
plus a final best_adapter/ directory (lowest eval loss).
"""

import argparse
import json
import os
import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(jsonl_path: str, tokenizer, max_seq_length: int) -> Dataset:
    rows = load_jsonl(jsonl_path)

    def fmt(sample):
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        # Tokenize for Trainer
        encoded = tokenizer(
            text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
        }

    ds = Dataset.from_list(rows).map(fmt, remove_columns=["messages"])
    return ds


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print(f"Model:  {cfg['model']}")
    print(f"Data:   {cfg['data']}")
    print(f"Output: {cfg['adapter_path']}")
    print()

    # ── load model (full precision with LoRA) ───────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False

    # ── attach LoRA ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── dataset ──────────────────────────────────────────────────────────────
    train_ds = build_dataset(
        os.path.join(cfg["data"], "train.jsonl"),
        tokenizer,
        cfg["max_seq_length"],
    )
    valid_ds = build_dataset(
        os.path.join(cfg["data"], "valid.jsonl"),
        tokenizer,
        cfg["max_seq_length"],
    )
    print(f"Train: {len(train_ds)} samples | Valid: {len(valid_ds)} samples")

    # ── trainer ──────────────────────────────────────────────────────────────
    os.makedirs(cfg["adapter_path"], exist_ok=True)

    bf16_supported = torch.cuda.is_bf16_supported()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_steps=cfg["warmup_steps"],
            max_steps=cfg["max_steps"],
            learning_rate=cfg["learning_rate"],
            fp16=not bf16_supported,
            bf16=bf16_supported,
            logging_steps=cfg["logging_steps"],
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_strategy="steps",
            save_steps=cfg["save_steps"],
            output_dir=cfg["adapter_path"],
            optim="adamw_8bit",
            seed=cfg["seed"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            gradient_checkpointing=True,
        ),
    )

    print("\nStarting training...")
    stats = trainer.train()
    print(f"\nTraining complete. Runtime: {stats.metrics['train_runtime']:.1f}s")

    # ── save best adapter ────────────────────────────────────────────────────
    best_dir = os.path.join(cfg["adapter_path"], "best_adapter")
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Best adapter saved → {best_dir}")

    # print val loss curve
    print("\nVal loss by step:")
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            print(f"  step {int(log['step']):4d}: {log['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
