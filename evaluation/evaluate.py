"""Evaluate a trained checkpoint on a test set with greedy decoding."""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, DynamicCache


# Evaluation reuses the training data pipeline.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from lapras.data.collator import LaprasDataCollatorWithPadding  # noqa: E402
from lapras.data.mm_plugin import FlamingoPlugin, SlipPlugin  # noqa: E402
from lapras.data.processor.lapras import LaprasDatasetProcessor  # noqa: E402
from lapras.data.processor.supervised import SupervisedDatasetProcessor  # noqa: E402
from lapras.data.template import get_template_and_fix_tokenizer  # noqa: E402
from lapras.extras.constants import IGNORE_INDEX  # noqa: E402
from lapras.hparams.data_args import DataArguments  # noqa: E402
from lapras.extras.constants import PROJECTION_WEIGHTS_NAME, RUN_CONFIG_NAME  # noqa: E402
from lapras.train.lapras.trainer import build_projection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_metrics, prepare_label_space, score_sample  # noqa: E402


# The remote modeling code of some backbones still calls the removed `get_usable_length`.
DynamicCache.get_usable_length = lambda self, seq_length=None, layer_idx=0: self.get_seq_length(layer_idx)


def load_run_config(ckpt: str) -> dict:
    path = os.path.join(ckpt, RUN_CONFIG_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Checkpoints trained with this repository write it next to the weights; it "
            "records the settings evaluation must reproduce (method, K, projection, layout, patch size)."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def disable_attention_dropout(model: nn.Module) -> None:
    """Zero the attention dropout of backbone code that applies it regardless of ``self.training``."""
    for module in model.modules():
        for attr in ("attention_dropout", "attn_dropout"):
            if isinstance(getattr(module, attr, None), float):
                setattr(module, attr, 0.0)


def load_model(ckpt: str, run_config: dict, device: str, torch_dtype=torch.bfloat16):
    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    ts_patch_size = run_config.get("ts_patch_size")
    if (
        ts_patch_size is not None
        and isinstance(getattr(config, "ts", None), dict)
    ):
        config.ts["patch_size"] = ts_patch_size  # ChatTS: the patch size the encoder was trained with

    model = AutoModelForCausalLM.from_pretrained(
        ckpt, config=config, trust_remote_code=True, torch_dtype=torch_dtype, low_cpu_mem_usage=False
    )

    # Latent methods feed each continuous thought back through the projection, or unchanged without one.
    projection = None
    if run_config["num_latent"] > 0:
        proj_cfg = run_config.get("projection")
        if proj_cfg is None:
            projection = nn.Identity()
        else:
            from safetensors.torch import load_file

            path = os.path.join(ckpt, PROJECTION_WEIGHTS_NAME)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"The run config declares a projection but {path} is missing.")
            projection = build_projection(model.config.hidden_size, proj_cfg["projection_dim"], 0.0)
            projection.load_state_dict(load_file(path), strict=True)
            projection = projection.to(torch_dtype)

    model = model.to(device).eval()
    disable_attention_dropout(model)
    if projection is not None:
        projection = projection.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    except Exception:  # SLIP / Flamingo checkpoints ship no processor
        processor = None
    return model, projection, tokenizer, processor


def build_data_args(run_config: dict, cutoff_len: int) -> DataArguments:
    data_args = DataArguments()
    data_args.template = run_config["template"]
    data_args.answer_anchor = run_config["answer_anchor"]
    data_args.cutoff_len = cutoff_len
    if run_config.get("use_cot") is not None:
        data_args.use_cot = run_config["use_cot"]
    if run_config.get("remove_eos") is not None:
        data_args.lapras_remove_eos = run_config["remove_eos"]
    return data_args


def to_aligned(sample: dict, ts_patch_size) -> dict:
    """One raw test record -> the aligned single-example batch the dataset processors consume."""
    raw_ts = sample.get("timeseries") or []
    ts_list = [np.asarray(t, dtype=np.float32) for t in raw_ts]
    return {
        "_prompt": [[{"role": "user", "content": sample["input"]}]],
        "_response": [[{"role": "assistant", "content": sample["output"]}]],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_timeseries": [ts_list],
        "_ts_patch_size": [ts_patch_size],
    }


def build_latent_features(processor: LaprasDatasetProcessor, samples: list[dict], ts_patch_size):
    features, kept = [], []
    for idx, sample in enumerate(samples):
        out = processor.preprocess_dataset(to_aligned(sample, ts_patch_size))
        if not out.get("encoder_input_ids"):
            continue
        features.append({k: v[0] for k, v in out.items()})
        kept.append(idx)
    return features, kept


def build_sft_features(processor: SupervisedDatasetProcessor, samples: list[dict], ts_patch_size):
    """Tokenize with the SFT processor and keep the prompt part (the model generates the rest)."""
    features, kept = [], []
    for idx, sample in enumerate(samples):
        out = processor.preprocess_dataset(to_aligned(sample, ts_patch_size))
        if not out.get("input_ids"):
            continue
        input_ids, labels = out["input_ids"][0], out["labels"][0]
        prompt_len = next((j for j, label in enumerate(labels) if label != IGNORE_INDEX), len(input_ids))
        features.append(
            {
                "input_ids": list(input_ids[:prompt_len]),
                "timeseries": out["timeseries"][0],
                "ts_patch_sizes": out["ts_patch_sizes"][0] if out.get("ts_patch_sizes") else None,
            }
        )
        kept.append(idx)
    return features, kept


def collate_sft(features: list[dict], template, processor, pad_id: int) -> dict:
    """Left-pad the prompts and encode the time series with the backbone's plugin (as in training)."""
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids = torch.full((len(features), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
    for i, f in enumerate(features):
        n = len(f["input_ids"])
        input_ids[i, -n:] = torch.tensor(f["input_ids"], dtype=torch.long)
        attention_mask[i, -n:] = 1

    # SLIP and Flamingo take one multivariate item per sample; ChatTS takes one item per channel.
    mm_plugin = template.mm_plugin
    per_sample = isinstance(mm_plugin, (SlipPlugin, FlamingoPlugin))
    ts_list, ts_patch_sizes = [], []
    for f in features:
        ts = f.get("timeseries") or []
        ps = f.get("ts_patch_sizes")
        if per_sample:
            if ts:
                ts_list.append(ts)
                ts_patch_sizes.append(ps[0] if ps else None)
        else:
            ts_list.extend(ts)
            if ps is not None:
                ts_patch_sizes.extend(ps)

    mm_inputs = mm_plugin.get_mm_inputs(
        [], [], [], [], [], [], [], processor, timeseries=ts_list, ts_patch_sizes=ts_patch_sizes or None
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, **mm_inputs}


def _move_nested(obj, device, dtype=None):
    if torch.is_tensor(obj):
        if dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, (list, tuple)):
        moved = [_move_nested(x, device, dtype) for x in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj


def _ts_kwargs(batch: dict, model) -> dict:
    device, dtype = model.device, getattr(model, "dtype", torch.bfloat16)
    kwargs = {}
    for key in ("timeseries", "sensor_attn_mask", "time_index"):
        if batch.get(key) is not None:
            kwargs[key] = _move_nested(batch[key], device, dtype=dtype)
    if batch.get("ts_patch_sizes") is not None:
        ps = batch["ts_patch_sizes"]
        kwargs["ts_patch_sizes"] = ps.to(device) if torch.is_tensor(ps) else ps
    return kwargs


@torch.no_grad()
def greedy_decode(model, tokenizer, past_kv, running_mask, next_token, eos_id: int, pad_id: int, max_new_tokens: int):
    """Greedy decoding off a KV cache, starting from ``next_token`` and stopping at EOS."""
    B = running_mask.size(0)
    embed_layer = model.get_input_embeddings()
    finished = torch.zeros(B, dtype=torch.bool, device=running_mask.device)
    generated: list[list[int]] = [[] for _ in range(B)]
    for _ in range(max_new_tokens):
        running_mask = torch.cat([running_mask, running_mask.new_ones((B, 1))], dim=1)
        position_ids = (running_mask.sum(dim=-1, keepdim=True) - 1).clamp(min=0).long()
        outputs = model(
            inputs_embeds=embed_layer(next_token),
            attention_mask=running_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for i, tok in enumerate(next_token.squeeze(1).tolist()):
            if finished[i]:
                continue
            if tok == eos_id:
                finished[i] = True
            else:
                generated[i].append(tok)
        if finished.all():
            break
        next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, pad_id), next_token)
    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated]


@torch.no_grad()
def generate_latent(model, projection, tokenizer, batch, num_latent, eot_id, eos_id, pad_id, max_new_tokens):
    """Lapras / COCONUT: encode prompt + <|bot|>, run K continuous thoughts, decode from <|eot|>."""
    device = model.device
    encoder_ids = batch["encoder_input_ids"].to(device)
    encoder_attn = batch["encoder_attention_mask"].to(device)
    B = encoder_ids.size(0)

    outputs = model(input_ids=encoder_ids, attention_mask=encoder_attn, use_cache=True, **_ts_kwargs(batch, model))
    past_kv = outputs.past_key_values
    post_mask = getattr(outputs, "attention_mask", None)  # ChatTS: mask of the TS-expanded prompt
    running_mask = post_mask if post_mask is not None else encoder_attn

    latent = projection(outputs.last_hidden_state[:, -1, :].unsqueeze(1))
    for _ in range(num_latent):
        running_mask = torch.cat([running_mask, running_mask.new_ones((B, 1))], dim=1)
        position_ids = (running_mask.sum(dim=-1, keepdim=True) - 1).clamp(min=0).long()
        outputs = model(
            inputs_embeds=latent,
            attention_mask=running_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        latent = projection(outputs.last_hidden_state[:, -1, :].unsqueeze(1))

    start = torch.full((B, 1), eot_id, dtype=torch.long, device=device)
    return greedy_decode(model, tokenizer, past_kv, running_mask, start, eos_id, pad_id, max_new_tokens)


@torch.no_grad()
def generate_text(model, tokenizer, batch, eos_id, pad_id, max_new_tokens):
    """No-CoT / CoT / iCoT: plain greedy decoding after the prompt."""
    device = model.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    B = input_ids.size(0)
    embed_layer = model.get_input_embeddings()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
        **_ts_kwargs(batch, model),
    )
    past_kv = outputs.past_key_values
    post_mask = getattr(outputs, "attention_mask", None)  # ChatTS: mask of the TS-expanded prompt
    running_mask = post_mask if post_mask is not None else attention_mask
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in range(B)]
    for i, tok in enumerate(next_token.squeeze(1).tolist()):
        if tok == eos_id:
            finished[i] = True
        else:
            generated[i].append(tok)

    # Without padding, decode without an attention mask (faster SDPA kernels).
    no_pad = bool(running_mask.all())
    base_len = int(running_mask.size(1))
    for step in range(max_new_tokens - 1):
        if no_pad:
            attn_arg = None
            position_ids = torch.full((B, 1), base_len + step, dtype=torch.long, device=device)
        else:
            running_mask = torch.cat([running_mask, running_mask.new_ones((B, 1))], dim=1)
            attn_arg = running_mask
            position_ids = (running_mask.sum(dim=-1, keepdim=True) - 1).clamp(min=0).long()
        outputs = model(
            inputs_embeds=embed_layer(next_token),
            attention_mask=attn_arg,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for i, tok in enumerate(next_token.squeeze(1).tolist()):
            if finished[i]:
                continue
            if tok == eos_id:
                finished[i] = True
            else:
                generated[i].append(tok)
        if finished.any():
            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, pad_id), next_token)
        if finished.all():
            break

    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory (contains lapras_run_config.json).")
    parser.add_argument("--test_file", required=True, help="Test jsonl (records with input / output / timeseries).")
    parser.add_argument("--output_dir", default=None, help="Default: <ckpt>/eval.")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--cutoff_len", type=int, default=10000, help="Maximum input length in tokens (as in training).")
    parser.add_argument(
        "--num_latent", type=int, default=None, help="Override the number of continuous thoughts (default: as trained)."
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Evaluate the first N samples only (debugging).")
    args = parser.parse_args()

    run_config = load_run_config(args.ckpt)
    stage = run_config["stage"]
    num_latent = run_config["num_latent"] if args.num_latent is None else args.num_latent
    is_latent = run_config["num_latent"] > 0
    if not is_latent and args.num_latent is not None:
        raise ValueError(f"--num_latent only applies to latent methods, but this is a `{stage}` checkpoint.")

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        rank, world_size, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    is_main = rank == 0
    torch.manual_seed(0)

    output_dir = args.output_dir or os.path.join(args.ckpt, "eval")
    suffix = f"_k{num_latent}" if args.num_latent is not None else ""
    pred_path = os.path.join(output_dir, f"predictions{suffix}.jsonl")
    summary_path = os.path.join(output_dir, f"summary{suffix}.json")

    model, projection, tokenizer, processor = load_model(args.ckpt, run_config, f"cuda:{local_rank}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id

    data_args = build_data_args(run_config, args.cutoff_len)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    ts_patch_size = run_config.get("ts_patch_size")

    with open(args.test_file, encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f]
    if args.max_samples:
        all_samples = all_samples[: args.max_samples]

    valid_labels, mcq, valid_letters = prepare_label_space(all_samples)
    samples = all_samples[rank::world_size]
    global_indices = list(range(rank, len(all_samples), world_size))

    if is_latent:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot|>")
        dataset_processor = LaprasDatasetProcessor(
            template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
        )
        collator = LaprasDataCollatorWithPadding(tokenizer=tokenizer, template=template, processor=processor)
        features, kept = build_latent_features(dataset_processor, samples, ts_patch_size)
    else:
        dataset_processor = SupervisedDatasetProcessor(
            template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
        )
        features, kept = build_sft_features(dataset_processor, samples, ts_patch_size)

    if is_main:
        mode = f"option letters {sorted(valid_letters)}" if mcq else f"{len(valid_labels)} text labels"
        k_info = f", K={num_latent}" if is_latent else ""
        print(f"[eval] {stage}{k_info} | {len(all_samples)} samples ({mode}) | world size {world_size}")

    results = []
    t0 = time.time()
    num_batches = (len(features) + args.batch_size - 1) // args.batch_size
    for b in tqdm(range(num_batches), desc=f"[rank {rank}]", disable=not is_main):
        rows = list(range(b * args.batch_size, min((b + 1) * args.batch_size, len(features))))
        batch_feats = [dict(features[i]) for i in rows]  # the collators pop the time series
        if is_latent:
            texts = generate_latent(
                model, projection, tokenizer, collator(batch_feats),
                num_latent, eot_id, eos_id, pad_id, args.max_new_tokens,
            )
        else:
            texts = generate_text(
                model, tokenizer, collate_sft(batch_feats, template, processor, pad_id),
                eos_id, pad_id, args.max_new_tokens,
            )

        for row, text in zip(rows, texts):
            sample = samples[kept[row]]
            gt_label, pred_label = score_sample(text, sample["output"], valid_labels, mcq, valid_letters)
            if gt_label is None:
                continue
            results.append(
                {
                    "idx": global_indices[kept[row]],
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "correct": pred_label == gt_label,
                    "pred_text": text,
                }
            )
    elapsed = time.time() - t0

    # Gather through per-rank files (avoids gather_object, which can hang on some fabrics).
    shard_dir = pred_path + ".shards"
    if distributed:
        if is_main:
            os.makedirs(shard_dir, exist_ok=True)
        dist.barrier()
        with open(os.path.join(shard_dir, f"rank_{rank}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f)
        dist.barrier()

    if is_main:
        if distributed:
            all_results = []
            for r in range(world_size):
                shard = os.path.join(shard_dir, f"rank_{r}.json")
                with open(shard, encoding="utf-8") as f:
                    all_results.extend(json.load(f))
                os.remove(shard)
            os.rmdir(shard_dir)
        else:
            all_results = results
        all_results.sort(key=lambda r: r["idx"])

        os.makedirs(output_dir, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        metrics = compute_metrics([r["gt_label"] for r in all_results], [r["pred_label"] for r in all_results])
        label_total = Counter(r["gt_label"] for r in all_results)
        label_correct = Counter(r["gt_label"] for r in all_results if r["correct"])
        summary = {
            "ckpt": os.path.abspath(args.ckpt),
            "test_file": os.path.abspath(args.test_file),
            "stage": stage,
            "use_cot": run_config.get("use_cot"),
            "num_latent": num_latent if is_latent else None,
            "max_new_tokens": args.max_new_tokens,
            "n_test": len(all_samples),
            **metrics,
            "per_class_accuracy": {
                label: 100.0 * label_correct[label] / label_total[label] for label in sorted(label_total)
            },
            "elapsed_s": elapsed,
            "world_size": world_size,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(
            f"[eval] accuracy {metrics['accuracy']:.2f} | macro-F1 {metrics['macro_f1']:.2f} | "
            f"n={metrics['n']} ({metrics['n_unparsed']} unparsed) -> {summary_path}"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
