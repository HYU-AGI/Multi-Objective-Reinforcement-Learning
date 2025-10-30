import os
from dataclasses import dataclass, field
from typing import Optional

import tyro
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, disable_caching

from src.trainer.modpo_trainer import MODPOTrainer
from src.data.configs import DEFAULT_PROMPT_TEMPLATE
from src.utils import (
    print_local_main, disable_progress_bar_non_local_main, set_seeds,
    prepare_model_for_peft, param_sharding_enabled, PeftAsPreTrained,
)
from src.utils.reward import RewardWrapperList, ImplicitRewardWrapper

disable_progress_bar_non_local_main()

@dataclass
class ScriptArguments:
    sft_model_name: str = field(metadata={"help": "base SFT model name"})
    margin_reward_model_name: str = field(metadata={"help": "path to margin LoRA checkpoint (best_checkpoint)"} )
    dataset_dir: str = field(default="./data/myset/trust", metadata={"help": "base dir containing train.jsonl/validation.jsonl"})
    use_flash_attention_2: Optional[bool] = field(default=False)
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE)
    dataset_caching: Optional[bool] = field(default=False)
    sanity_check: Optional[bool] = field(default=False)

    w: Optional[float] = field(default=0.5, metadata={"help": "weight for main vs margin"})
    beta: Optional[float] = field(default=0.1)
    max_length: Optional[int] = field(default=1024)
    num_proc: Optional[int] = field(default=4)
    generate_during_eval: Optional[bool] = field(default=True)

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/modpo",
            overwrite_output_dir=True,
            seed=42,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0,
            weight_decay=0.05,
            fp16=True,
            remove_unused_columns=False,
            run_name="dev_modpo",
            report_to="wandb",
            num_train_epochs=3,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
        )
    )

    peft: Optional[bool] = field(default=True)
    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
    )

def load_datasets_local(script_args):
    if not script_args.dataset_caching:
        disable_caching()
    ds = load_dataset(
        "json",
        data_files={
            "train": os.path.join(script_args.dataset_dir, "train.jsonl"),
            "validation": os.path.join(script_args.dataset_dir, "validation.jsonl"),
        }
    )
    def map_fn(ex):
        prompt = script_args.prompt_template.format(raw_prompt=ex["prompt"])
        return {"prompt": prompt, "raw_prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]}
    train_dataset = ds["train"].map(map_fn, remove_columns=ds["train"].column_names, num_proc=script_args.num_proc)
    eval_dataset  = ds["validation"].map(map_fn, remove_columns=ds["validation"].column_names, num_proc=script_args.num_proc)
    if script_args.sanity_check:
        train_dataset = train_dataset.select(range(min(128, len(train_dataset))))
        eval_dataset  = eval_dataset.select(range(min(128, len(eval_dataset))))
    return train_dataset, eval_dataset

def main():
    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.training_args.seed)
    if not script_args.peft:
        script_args.peft_config = None

    print_local_main("loading model...")
    base = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype="auto",
        **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
    )
    base.config.update({"use_cache": False, "pad_token_id": base.config.eos_token_id})
    print_local_main(base)
    print_local_main(script_args.peft_config)

    # attach trainable LoRA
    model = prepare_model_for_peft(base, peft_config=script_args.peft_config, args=script_args.training_args)
    # load frozen margin-reward LoRA weights
    model.load_adapter(script_args.margin_reward_model_name, adapter_name="margin_reward")

    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, eval_dataset = load_datasets_local(script_args)

    print_local_main("start training...")
    trainer = MODPOTrainer(
        model=model,
        beta=script_args.beta,
        args=script_args.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        num_proc=script_args.num_proc,
        generate_during_eval=script_args.generate_during_eval,
    )
    if Accelerator().is_local_main_process:
        trainer.model.print_trainable_parameters()

    trainer.set_wrapped_margin_reward_model_list(
        RewardWrapperList([
            ImplicitRewardWrapper(
                model=PeftAsPreTrained(trainer.model, "margin_reward"),  # 안전(마진) LoRA on
                ref_model=PeftAsPreTrained(trainer.model),               # 현재 정책(학습 LoRA on)
                tokenizer=tokenizer,
                beta=script_args.beta,
                prompt_template=script_args.prompt_template,
            )
        ]),
        w=(script_args.w, 1 - script_args.w),
        prepare=False,
    )

    trainer.train()

    save_name = "best_checkpoint" if script_args.training_args.load_best_model_at_end else "final_checkpoint"
    trainer.model.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
    trainer.tokenizer.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))

if __name__ == "__main__":
    main()
