# AGENTS.md

Quick orientation for humans and coding agents working in this repo.

## What this repo is

`parameter-golf` is the OpenAI Parameter Golf challenge repo. The goal is to train a language model that fits in a `16,000,000` byte submission artifact and is evaluated by tokenizer-agnostic compression on FineWeb validation data.

The main public overview and challenge rules live in `README.md`.

## Top-level map

- `train_gpt.py`: main CUDA/PyTorch training script and submission-artifact pipeline.
- `train_gpt_mlx.py`: Apple Silicon / MLX local training path.
- `data/`: dataset download, tokenizer export, and retokenization helpers.
- `records/`: example submissions and baselines, each with a frozen `train_gpt.py`, `README.md`, `submission.json`, and `train.log`.

Important: `records/` is an archive of self-contained submissions, not the live code you usually want to edit first.

## Fastest starting points

If you want to understand the repo quickly, read files in this order:

1. `README.md`
2. `train_gpt.py`
3. `data/README.md`
4. one or both example records under `records/`

## Main training entry points

### PyTorch / CUDA

`train_gpt.py` is the primary competitive baseline. Important areas:

- `Hyperparameters`: env-driven config surface for data paths, model shape, optimization, and wallclock cap.
- `eval_val(...)`: computes `val_loss` and tokenizer-agnostic `val_bpb`.
- `TokenStream` / `DistributedTokenLoader`: sequential shard streaming with simple deterministic distributed loading.
- `GPT`: the actual model; uses tied embeddings by default, grouped-query attention, RMSNorm, rotary embeddings, and encoder/decoder-style skip reuse.
- `main()`: distributed setup, compile/warmup, training loop, artifact serialization, and roundtrip validation.

### MLX / Apple Silicon

`train_gpt_mlx.py` mirrors the baseline for local iteration on Apple hardware. It keeps the old `int8+zlib` export path and is mainly a convenience path for smoke tests and local experimentation.

## Data workflow

Relevant files:

- `data/cached_challenge_fineweb.py`: downloads the published manifest, tokenizer artifacts, validation shards, and a requested number of train shards from Hugging Face.
- `data/download_hf_docs_and_tokenize.py`: rebuilds tokenizers and shard exports from the published docs cache.
- `data/tokenizer_specs.json`: tokenizer definitions used by the export script.
- `data/README.md`: canonical local layout and export/download instructions.

Shard format is validated in `train_gpt.py` and `train_gpt_mlx.py` using a simple binary header with magic/version metadata plus `uint16` token payloads.

One useful asymmetry: `train_gpt_mlx.py` is stricter about dataset/tokenizer pairing and validates against `manifest.json`, while `train_gpt.py` mostly checks that the tokenizer is a SentencePiece `.model` with the expected vocab size.

## Records layout

`records/track_10min_16mb/2026-03-17_NaiveBaseline/` is the baseline leaderboard example.

`records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` is an unlimited-compute example run.

These record folders are important because challenge submissions are expected to add a new self-contained folder under `records/`, rather than replacing the root training scripts.

## Current branch note

On the current branch, `train_gpt.py` no longer uses the old per-row `int8+zlib` artifact pipeline. It now uses a `nanoquant`-style low-rank binary factorization path:

- encode: `quantize_state_dict_nanoquant(...)`
- decode: `dequantize_state_dict_nanoquant(...)`
- artifact: `final_model.nanoquant.ptz`

Practical implications:

- large 2-D weights are compressed with packed binary factors plus FP16 scale vectors
- small tensors, control tensors, and sensitive tensors like `tok_emb` / `lm_head` stay as FP16 passthrough
- final validation log keys in PyTorch now use `final_nanoquant_roundtrip...`

The MLX script still uses the older `int8+zlib` flow, so the two top-level training scripts are no longer symmetric in artifact format.
The root `README.md` also still describes the older `final_int8_zlib_roundtrip` PyTorch output, so the docs do not fully match this branch's CUDA trainer.

This branch is also behind `origin/main`, so treat it as a side experiment on an older trainer snapshot rather than the latest canonical state of the repo.

## Good repo assumptions

- Root scripts are intended as readable baselines, not the best possible models.
- The README and record folders are part of the product surface; changes there matter.
- Competitive experiments should usually land in a new `records/...` folder, unless the change is clearly a baseline or infrastructure improvement.
- Keep an eye on script length: both root training scripts are intended to stay under `1500` lines.
- If you are extending `nanoquant`, decide early whether to keep working on this branch snapshot or rebase/port it onto current `main`.
