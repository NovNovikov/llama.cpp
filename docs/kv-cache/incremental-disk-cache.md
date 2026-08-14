# Server disk KV cache

The server's automatic disk cache is opt-in. It is independent from the
`/slots` manual save/restore format: manual RAM checkpoints remain self-contained
server state, while this cache is a best-effort performance layer.

## Enable it

```text
--slot-save-path <directory> --slot-save-auto
```

Useful controls:

| Flag | Default | Purpose |
|---|---:|---|
| `--slot-save-block N` | server default | Only whole block-aligned token prefixes are indexed. |
| `--slot-save-min-tokens N` | 1024 | Do not persist a normal snapshot smaller than this. |
| `--slot-save-context-min-tokens N` | 4096 | Minimum leading shared context worth saving during cold prefill. |
| `--slot-restore-min-tokens N` | 0 | Recompute very short verified prefixes instead of loading them. |
| `--slot-save-incremental` | off | Store continuations as parent-linked deltas. |
| `--slot-save-max-count N` | unlimited | Bounded tree-aware cache size by snapshot count. |
| `--slot-save-max-mb N` | unlimited | Bounded tree-aware cache size by bytes. |
| `--slot-save-idle-seconds N` | disabled | Flush a warm idle slot without waiting for its reuse or server shutdown. |

Use a directory dedicated to this cache. Files are shared safely by multiple
server processes, but they are not a durable database or a replacement for a
manual `/slots` export.

## Snapshot formats

An automatic snapshot publishes a `.bin`, optional `.logits`, and `.meta` sidecar.
The metadata records a model/context/LoRA fingerprint, full token-cell prefix,
and, for multimodal snapshots, verified media identity records.

| Version | Snapshot |
|---:|---|
| v1 | whole text KV state |
| v2 | whole multimodal KV state |
| v3 | text delta with a parent link |
| v4 | multimodal delta with a parent link |

Delta metadata retains the complete token and media record tiling while its
`.bin` contains only the suffix after the parent. Restore validates the current
model fingerprint, media projector fingerprint, media identities, token prefix,
and every parent link before loading a root-to-tip chain. Any failed validation
falls back to ordinary prefill; no partially trusted state is used.

## Shared context and movable history

For a cold generative text request, the server can save a one-shot whole base at
the first user-message boundary, rounded down to `--slot-save-block`. This is
the stable leading system/developer/tool/RAG context. A later request with
different post-history instructions restores that base only when its verified
prefix reaches it, then recomputes the changed suffix. The server does not
assume that later user or assistant history is movable.

The existing manual RAM checkpoints continue to govern in-process rollback for
recurrent, hybrid, and sliding-window models. A disk restore reconstructs a
compatible RAM checkpoint when the memory implementation needs one; it neither
serializes nor replaces the manual `/slots` checkpoint format.

## Logging and eviction

At normal server log level the cache reports the RAM prefix length, disk candidate
count and type, accepted or rejected candidates with a reason, snapshot save
kind, shared-context boundary, checkpoint restoration, strict extension versus
divergence, and suffix recomputation. It does not log prompt text or tokens.

With `--slot-save-incremental`, eviction never removes a parent that has a live
delta child. Orphaned deltas are cleaned up, then the least-recently-used leaf is
evicted until the configured count/byte limit is met.
