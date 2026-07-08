# Project Status

This file is a compact project memory for the local `llama.cpp` fork in this repo.
It exists so important context survives chat history compression.

## Identity

- Repository: `L:\AI_pictures_generate\Manual LLAMA_CPP\llama.cpp`
- Main working branch: `feature/prefill-checkpoints`
- Main personal remote: `nov` -> `https://github.com/NovNovikov/llama.cpp.git`
- Upstream remote: `origin` -> `https://github.com/ggml-org/llama.cpp.git`

## Current Git State

- Current branch: `feature/prefill-checkpoints`
- Latest local merge from upstream: `bbd927f51`
- Current upstream master integrated up to: `c198af4dc`
- The branch may show a large `ahead` count because several older local commits are equivalent or similar to PRs that were later merged upstream, but their history was not rewritten.

## Core Purpose of This Fork

This fork is focused on:

- better chat-completion prefill behavior, especially with Jinja/reasoning templates
- restored and improved server context checkpoint scheduling
- stronger speculative / MTP stability on server flows
- DeepSeek V4 / DSV4 performance and compatibility work
- keeping useful local server behavior even as upstream changes rapidly

## Important Local Features

### 1. Prefill / Chat Template Behavior

- Assistant prefill is preserved more consistently in chat completion flows that use Jinja / reasoning-enabled templates.
- This is specifically meant to avoid regressions around `enable_thinking` and final rendered prompt handling.

### 2. Checkpoint System

The server checkpoint behavior is intentionally not vanilla upstream.

- `-cpent`, `--checkpoint-every-n-tokens` is restored.
- Periodic checkpoints are enabled independently from latest-user-message gating.
- Checkpoints are currently placed with anchor logic:
  - quarter checkpoint around 25% of prompt
  - midpoint checkpoint around 50% of prompt
  - periodic checkpoints after 50%
- Current post-midpoint periodic interval: `8192`
- `--checkpoint-min-step` is treated as a spacing filter, not as a replacement for periodic scheduling.
- Checkpoint creation is batch-aligned so large prefill batches are not split just to hit checkpoint boundaries.
- Anchor checkpoint retention logic exists so important quarter/midpoint checkpoints are not discarded too aggressively under cache pressure.
- Prompt-cache eviction logic in server code was kept more conservative than upstream so the last state is not dropped unnecessarily.

### 3. Speculative / MTP / Draft Stability

- Draft/MTP state is preserved across checkpoint restore flows.
- Draft contexts must not inherit target embeddings / pooling mode.
- `ctx_other` handling is important for Gemma assistant / MTP style models.
- There is local logic to keep draft/MTP memory estimation active even with `--fit off`, so the server can still report projected memory needs.
- There is a local retry path for draft memory estimation with temporary `ctx_other` when needed for Gemma-style assistant drafts.

### 4. DeepSeek V4 / DSV4

This fork contains significant DSV4-specific work.

- CUDA `LIGHTNING_INDEXER` support is integrated.
- DeepSeek4 model path is wired to use the lightning indexer.
- Fused HC ops are present.
- Quantized KV-cache fixes for DSV4 are present.
- DSV4 state saving was updated to write only used rows in state.
- There is a local pending DSV4 partial-checkpoint follow-up that saves and restores base + block caches in partial state, because those caches are not safely recomputable from a short partial re-decode.
- The fork has repeatedly been tested against other public DSV4 forks and is intended to be the better-behaved server path for long-context DeepSeek V4 use.

### 5. DSpark

- DSpark draft-model runtime support exists for standalone draft-model style usage.
- Embedded DSpark stored inside a single DeepSeek target model is **not** fully supported yet.
- This limitation is intentional and known.

### 6. Laguna

- Laguna architecture / converter / template support is included.

## Known Limitations / Caveats

- DeepSeek V4 prompt processing can still degrade as context grows. This has been reduced, but not fully solved.
- Some experimental attempts to optimize DeepSeek sparse attention were intentionally abandoned because they caused OOMs, instability, or broken inference quality.
- Embedded DeepSeek DSpark support is still missing.
- Git history is not cleanly rebased against upstream. Some local commits duplicate or partially overlap PRs that later landed upstream.
- Because of that, `ahead` count is inflated relative to the number of truly unique local ideas.

## Important Non-Goals

The following experimental directions were tried and should be treated carefully:

- chunked exact top-k style DeepSeek experiments that multiply graph complexity
- broad rewrites of DeepSeek sparse mask logic without strict fallback paths
- history rewriting just to reduce `ahead` count without a dedicated cleanup branch

## Working Tree Notes

At the time this file was created, the working tree was not fully clean.
Known local, not-yet-committed items included:

- `src/llama-kv-cache-dsv4.cpp`
- `tools/server/README.md`
- `STATUS.md`
- `codex-backups/`

These should be reviewed before any history cleanup or release tagging.

## Recommended Safe Workflow

- Keep important state in files like this one instead of only in chat.
- Make small focused commits before risky merges or experiments.
- If rebasing or cleaning history, do it on a separate cleanup branch.
- Before adopting new upstream speculative / server changes, verify they do not overwrite:
  - local checkpoint scheduling
  - local prefill behavior
  - draft/MTP embedding/pooling protections
  - DeepSeek4 / DSV4 local server behavior

## Quick Mental Model

If something suddenly breaks after an upstream merge, the first places to inspect are:

- `tools/server/server-context.cpp`
- `tools/server/server-task.cpp`
- `common/speculative.cpp`
- `src/models/deepseek4.cpp`
- `src/llama-kv-cache-dsv4.cpp`
- `src/llama-context.cpp`

Those are the main collision zones between upstream behavior and this fork's local logic.
