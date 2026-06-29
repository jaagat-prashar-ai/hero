# research-project-template

A template for building ML research experiments on top of [research-core](research-core/), the shared ML infrastructure library.

This repo re-implements MNIST as a non-trivial example whose layout scales to real research projects. It demonstrates:

- Domain schemas and typed tensor contracts
- Composable model components (encoder + classifier as independent SchemaModules)
- Multiple SchemaLoss implementations (cross-entropy, label smoothing)
- Custom callbacks (confusion matrix logging)
- Structured config dataclasses
- A clean Lilypad training entrypoint with preemption resumption

`research-core` is included as a git submodule and accessed via Bazel `local_repository`.

---

## Repository Structure

```
mnist/
├── schema/           # TensorSchema constants and domain schemas
├── components/       # MNISTEncoder, MNISTClassifier (independent SchemaModules)
├── model.py          # MNISTNet: encoder + classifier orchestrator
├── config.py         # MNISTTrainingConfig + EncoderConfig (dataclasses)
├── data/             # MNISTDataModule (SchemaDataModule)
└── training/
    ├── train.py              # Lilypad entrypoint + _flat_to_config
    ├── lightning_module.py   # MNISTLightningModule
    ├── losses/               # CrossEntropyLoss, LabelSmoothingLoss (SchemaLoss)
    ├── callbacks/            # ConfusionMatrixCallback
    └── configs/              # local.yaml, cluster.yaml
```

| Component type | Location |
|----------------|----------|
| Tensor schemas / constants | `mnist/schema/` |
| Model config dataclasses | `mnist/config.py` |
| Model components | `mnist/components/` |
| Model orchestrator | `mnist/model.py` |
| Data module | `mnist/data/datamodule.py` |
| Loss functions | `mnist/training/losses/` |
| Training callbacks | `mnist/training/callbacks/` |
| Lightning module | `mnist/training/lightning_module.py` |
| Training entrypoint | `mnist/training/train.py` |
| Lilypad YAML configs | `mnist/training/configs/` |

---

## Using This Template for a New Project

1. Fork or copy this repository.
2. Rename `mnist/` to your project name (e.g., `mymodel/`).
3. Update `workspace(name = "mnist_template")` in `WORKSPACE` **and** the matching
   `workspace_name = "mnist_template"` in the `lilypad_workload_image` target.
4. Update `.dev_docker_name` to your project's dev-container name.
5. Update `bazel_target` and `training_fn` in your YAML configs.

---

## Building and Running

All build/test commands run **inside the dev container** — a NOOP layer over the
research-core dev base image. `docker/Dockerfile` is the single place to add
project-specific *system* requirements (Python deps come from Bazel, not the image).

### Start the dev container (once)

```bash
./docker/build.sh && ./docker/run.sh
```

Then run any command inside it with the helper (container name from `.dev_docker_name`):

```bash
./docker/in_docker.sh -c "bazel test //mnist/... --test_output=errors"
```

### Run all unit tests

```bash
./docker/in_docker.sh -c "bazel test //mnist/..."
```

### Build the Lilypad workload image

```bash
./docker/in_docker.sh -c "bazel build //mnist/training:lilypad_mnist_training"
```

### Launch a local training run

```bash
./docker/in_docker.sh -c "bazel-bin/external/python_deps_lilypad_py/rules_python_wheel_entry_point_lilypad \
  workload launch mnist/training/configs/local.yaml -n template-mnist-local"
```

Expected: ~500 steps, val/accuracy > 0.85 (~0.91 typical), confusion matrix and images in WandB under `research / mnist-template`.

---

## Dependencies

Dependencies **compose** across the two repos — every requirement is declared in
exactly one place, and the generated lockfile is their union:

- **Shared infrastructure deps** (torch, lightning, wandb, ray, lilypad, schema/s3
  deps, ...) are declared once in **`research-core/pyproject.toml`**. The template
  pulls them in transitively via the `research-core` path source in its own
  `pyproject.toml` — they are **never** duplicated here.
- **Project-specific deps** (deps your experiment needs beyond research-core) are
  declared in **this repo's `pyproject.toml`** (e.g. `matplotlib`).

`research-core` itself is excluded from the lockfile — Bazel supplies its source via
`local_repository`, so only its *dependency closure* is emitted (`generate_lockfiles.py`
passes `--no-emit-package research-core`).

### Adding a project-specific dependency

1. Add it to **this repo's** `pyproject.toml` under `dependencies`.
2. Regenerate the lockfile (resolves the union of both repos' deps):
   ```bash
   python3 lockfiles/generate_lockfiles.py pyproject.toml -o lockfiles/requirements_lock.txt
   ```
3. Reference it in BUILD files as `@python_deps//your_package`.

### Adding a shared dependency (needed by research-core)

1. Add it to **`research-core/pyproject.toml`** and commit it there.
2. Update the submodule pointer in this repo (`git -C research-core pull` / checkout).
3. Regenerate the lockfile here (step 2 above) — the new dep is unioned in automatically.

> The lockfile (`requirements_lock.txt`) and `uv.lock` are generated artifacts; commit
> them but never hand-edit. Resolver index config in `pyproject.toml` is intentionally
> repeated (uv does not inherit it through path deps) — that is configuration, not a
> requirement, so it does not violate the single-source-of-truth rule.

---

## Preemption Resumption

### Automatic (recommended)

Set `requeue_if_preempted: true` in your cluster config. When Lilypad requeues a preempted job it restarts with the **same W&B run ID**, so `fetch_latest_checkpoint(wandb.run.id, ...)` in `training_loop` finds the prior checkpoint automatically. `trainer.fit(ckpt_path=...)` restores model weights, optimizer state, LR scheduler, and `global_step`.

**What you need:** Set `checkpoint_interval` and `requeue_if_preempted: true`. Everything else is automatic.

### Manual resume (fallback)

For runs that crashed without being requeued:

```yaml
training_fn_config:
  resume_run_id: "abc12345"  # Only for manual re-launch; prefer requeue_if_preempted
```

### Cross-run weight transfer

To load weights from a prior run without resuming optimizer state (e.g. fine-tuning):

```yaml
training_fn_config:
  checkpoint_artifact: "research/mnist-template/model-abc12345:v3"
```
