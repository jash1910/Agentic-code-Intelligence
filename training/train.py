import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "training", "outputs")

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning on LangChain Dataset")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="deepseek-ai/DeepSeek-Coder-1.3B-instruct", 
        help="Base model ID (e.g., deepseek-ai/DeepSeek-Coder-6.7B-instruct)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick 1-step test run")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    return parser.parse_args()

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        # DeepSeek Coder Instruct format
        text = (
            f"You are a helpful and expert AI assistant specialized in LangChain. "
            f"Answer the following question about LangChain internals, code, or documentation.\n\n"
            f"### Instruction:\n{example['instruction'][i]}\n\n"
            f"### Response:\n{example['response'][i]}"
        )
        output_texts.append(text)
    return output_texts

def train():
    args = parse_args()
    
    # 1. Device detection and configurations
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("[Trainer] CUDA detected. Running on GPU.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[Trainer] MPS (Apple Silicon) detected. Running on Mac GPU.")
    else:
        print("[Trainer] Running on CPU. This will be slow!")

    # Set up wandb
    if args.no_wandb or device != "cuda":
        os.environ["WANDB_DISABLED"] = "true"
        print("[Trainer] Weights & Biases tracking disabled.")
    else:
        print("[Trainer] Weights & Biases tracking enabled. Ensure you are logged in via 'wandb login'.")

    # 2. Setup paths and verify datasets
    train_path = os.path.join(DATASET_DIR, "train.jsonl")
    val_path = os.path.join(DATASET_DIR, "val.jsonl")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"[Trainer] Error: Training data not found at {train_path}. Please run data_collection first.", file=sys.stderr)
        sys.exit(1)

    print(f"[Trainer] Loading dataset...")
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})

    # 3. Load Tokenizer
    print(f"[Trainer] Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Quantization configuration (QLoRA)
    # Quantization (bitsandbytes) is only supported on CUDA/GPU environments.
    quant_config = None
    if device == "cuda":
        print("[Trainer] Configuring 4-bit QLoRA Quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("[Trainer] Quantization skipped because CUDA is not available. Loading model in standard precision.")

    # 5. Load Base Model
    # On Mac/CPU, we load with float32 or float16 and no quantization config
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,
    }
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    elif device == "mps":
        model_kwargs["torch_dtype"] = torch.float16
        
    print(f"[Trainer] Loading model {args.model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        if device == "mps":
            model = model.to("mps")
    except Exception as e:
        print(f"[Trainer] Error loading model: {e}")
        print("[Trainer] Falling back to a tiny model (gpt2) for local execution/testing.")
        args.model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_id)
        if device == "mps":
            model = model.to("mps")

    # 6. Prepare model for training / PEFT LoRA Config
    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    print("[Trainer] Initializing LoRA configuration...")
    # Target common projection modules in Causal LMs
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # If using gpt2 fallback
    if "gpt2" in args.model_id:
        target_modules = ["c_attn", "c_proj"]

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSALLM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 7. Training Arguments
    # Adjust for dry run
    num_epochs = args.epochs
    max_steps = -1
    logging_steps = 10
    
    if args.dry_run:
        print("[Trainer] Dry run mode enabled. Setting epochs to 1 and max_steps to 2.")
        num_epochs = 1
        max_steps = 2
        logging_steps = 1

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=logging_steps * 2 if not args.dry_run else 1,
        save_strategy="steps",
        save_steps=100 if not args.dry_run else 2,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=(device == "cuda"), # Use fp16 on CUDA only
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none" if (args.no_wandb or device != "cuda") else "wandb",
        remove_unused_columns=False
    )

    # 8. SFTTrainer
    print("[Trainer] Starting SFTTrainer initialization...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("[Trainer] Starting training...")
    trainer.train()

    # 9. Save adapter weights
    print(f"[Trainer] Saving fine-tuned adapter weights to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[Trainer] Training complete and weights saved successfully.")

if __name__ == "__main__":
    train()
