# nanograd-components

Model components for [nanograd](https://github.com/iraikov/nanograd), a deep learning
framework for CHICKEN Scheme: a ResNet, MLP-Mixer, and LLAMA-style
Transformer.

Each model comes in two forms:

- a strict version on nanograd's eagerly evaluated tensors, which
  computes values as it goes and is trained with `backward!`;
- a lazy version based on
  [array-morphisms](https://github.com/iraikov/array-morphisms)
  tensors.  Its forward pass builds a graph. `am-training-step/ssa`
  compiles the forward pass, the backward pass and the parameter
  gradients of a whole training step into one SSA program, then
  replays that program on each new batch.

The lazy models compute the same functions as their strict counterparts
and keep the parameters in the same order. `am-copy-parameters!` can
therefore move weights from a strict model to a lazy one. 

## Contents

- [Models](#models)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Detailed model notes](#detailed-model-notes)

## Models

| Model | Strict module | Lazy module | Notes |
|---|---|---|---|
| Transformer | `nanograd-transformer` | `nanograd-am-transformer` | [notes/transformer_readme.md](notes/transformer_readme.md) |
| ResNet | `nanograd-resnet` | `nanograd-am-resnet` | [notes/resnet_readme.md](notes/resnet_readme.md) |
| MLP-Mixer | `nanograd-mlp-mixer` | `nanograd-am-mlp-mixer` | [notes/mlp_mixer_readme.md](notes/mlp_mixer_readme.md) |

`nanograd-am-common` provides the parts shared by the lazy models:
- parameter constructors;
- row-wise softmax, RMSNorm and LayerNorm;
- NHWC batch normalization;
- `am-half-mse-loss`;
- parameter copying and counting.

### Transformer

The architecture follows LLAMA2:
- token embedding;
- pre-norm blocks of RMSNorm, multi-head self-attention with rotary
  position embedding (RoPE), and a SwiGLU feed-forward network;
- a final RMSNorm;
- an output projection tied to the embedding.

Attention is causal by default.

- **Strict** (`make-transformer`): processes one sequence at a time as a
  `(seq_len, d_model)` matrix. A KV cache (`make-kv-cache`) supports
  incremental generation.
- **Lazy** (`make-am-transformer`): processes a batch of B sequences of
  length T as one `[B*T, vocab]` matrix of one-hot rows. Each attention
  head is selected by a constant matrix product. A block-diagonal causal
  mask keeps the sequences separate. The lazy model has no KV cache.

### ResNet

- **Strict**: `make-resnet-18`, `-34`, `-50`, `-101` and `-152` are the
  ImageNet-style models, with a 7x7 stem and max pooling, built from basic
  or bottleneck blocks. `make-resnet-cifar` has a 3x3 stem, no max
  pooling, and configurable stages and widths. The strict models accept
  `(C, H, W)` or `(N, C, H, W)` input. Batch normalization keeps running
  statistics for eval mode.
- **Lazy**: `make-am-resnet`, `make-am-resnet-8` and
  `make-am-resnet-18-narrow` take NHWC input `[N, H, W, C]`.
  - They match `make-resnet-cifar`.
  - Batch normalization always uses the statistics of the current batch.
  - There is no max pooling, because the SSA engine has no reduce-max
    gradient.
  - `am-nchw->nhwc` converts input data from the NCHW layout.

### MLP-Mixer

Blocks of token-mixing and channel-mixing MLPs with pre-normalization
(RMSNorm or LayerNorm) and skip connections. The architecture is intended
for small fixed grids, such as the Tiny Recursive Model.

The lazy `make-am-mlp-mixer` works on a token-major `[T*B, D]` batch.
`am-mixer-pack-batch` and `am-mixer-unpack-batch` convert between this
layout and one sequence per sample. With the `num-classes:` keyword, the
lazy model adds a classifier head.

## Installation

```bash
chicken-install
```

The lazy examples and the benchmarks also need the
`array-morphisms-crunch` egg by default (see [Examples](#examples)). 

## Quick start

### Strict Transformer

```scheme
(import (srfi 4) nanograd-autograd nanograd-layer nanograd-optimizer
        nanograd-transformer)

;; vocab 28, context 32, d_model 64, 2 layers, 4 heads
(define model (make-transformer 28 32 64 2 4 dtype: 'f32))
(define opt (make-adam (parameters model) learning-rate: 0.005))

(define (train-step! inputs targets)        ; lists of token ids
  (let ((loss (cross-entropy-loss
               (forward model inputs)       ; logits (seq_len, vocab)
               (make-tensor32 (list->f32vector (map exact->inexact targets))
                              (list (length targets)) requires-grad?: #f)
               from-logits: #t)))
    (backward! loss)
    (step! opt)
    (zero-grad-layer! model)
    loss))
```

### Lazy ResNet with SSA training

```scheme
(import (srfi 4) yasos array-morphisms-core array-morphisms-context
        (prefix array-morphisms-grad am:)
        nanograd-autograd nanograd-layer nanograd-optimizer
        nanograd-array-morphisms nanograd-am-common nanograd-am-resnet)

(define (vector->lazy data shape)
  (get-or-make-lazy (am:make-var (make-morphism data shape 'f32) #f)))

(define model (make-am-resnet '(1 1 1) '(16 32 64) 10 dtype: 'f32))
(define opt (make-adam (parameters model) learning-rate: 0.001))
(define ctx (make-morphism-context))      ; one compiled program per context

;; x: NHWC f32vector [N, H, W, 3]; t: one-hot f32vector [N, 10]
(define (train-step! x t n)
  (let ((x-lt (vector->lazy x (list n 32 32 3)))
        (t-lt (vector->lazy t (list n 10))))
    ;; The first call compiles the step; later calls replay it.
    (am-training-step/ssa ctx opt model
                          (lambda (logits) (am-cross-entropy-loss logits t-lt))
                          x-lt t-lt)))
```

A compiled program is fixed to the shapes of its first call. Use one
context per batch shape, and a separate context with `am-forward/ssa` for
inference.

## Examples

Run the examples from the repository root with `csi -s`. 

| Script | Contents | Time |
|---|---|---|
| [examples/transformer_example.scm](examples/transformer_example.scm) | Strict Transformer: character-level LM with KV-cache generation, copy task, training with and without RoPE, KV-cache speed and consistency, an unstable SGD run with and without gradient clipping | ~50 s |
| [examples/am_transformer_example.scm](examples/am_transformer_example.scm) | The same tasks with the lazy Transformer, and a check that it agrees with the strict model in logits and generated tokens | ~35 s |
| [examples/resnet_example.scm](examples/resnet_example.scm) | Strict ResNet: classification of synthetic images, learning-rate schedule, parameter counts of ResNet-18/34/50, output statistics, transfer learning | ~60 s |
| [examples/am_resnet_example.scm](examples/am_resnet_example.scm) | The same with the lazy ResNet, plus agreement with the strict model (logits, loss and per-parameter gradient norms) and compiled-inference timing | ~60 s |
| [examples/mlp-mixer-examples.scm](examples/mlp-mixer-examples.scm) | Strict MLP-Mixer: forward pass, TRM-style recursive refinement, deep supervision, training loop | <1 s |

The training loops use the gradient diagnostics of `nanograd-diagnostics`
through [examples/gradient-monitoring.scm](examples/gradient-monitoring.scm).
That file wraps the optimizer so that each `step!` does three things:
- records the gradient norms in a gradient monitor, which flags exploding,
  vanishing, non-finite and unbalanced gradients;
- optionally clips the global norm;
- applies the update.

The examples print three kinds of diagnostics:
- per-parameter gradient tables;
- the gradient norm over training;
- the summary of `diagnose-training`.

In the ResNet models, the monitor reports the convolution biases that come
before batch normalization as parameters without a gradient, because
batch normalization cancels them.

The lazy examples choose their backend with the `AM_EXAMPLE_BACKEND`
environment variable (see
[examples/am-example-backend.scm](examples/am-example-backend.scm)). The
choices are:
- `crunch` (default): the crunch kernels; needs `array-morphisms-crunch`.
- `blas`: the system BLAS egg backend.
- `scheme`: no backend, only the Scheme code of array-morphisms.

```bash
OPENBLAS_NUM_THREADS=8 AM_EXAMPLE_BACKEND=blas csi -s examples/am_resnet_example.scm
```

## Benchmarks

`examples/bench-am-{resnet,mlp-mixer,transformer}-crunch.scm` time one
SSA training step of each lazy model against its strict counterpart.
They are controlled by environment variables:
- `AM_BENCH_BACKEND`
- `AM_BENCH_CONFIG`
- `AM_BENCH_STEPS`
- `AM_BENCH_STRICT`
- `AM_BENCH_STRICT_STEPS`
- `AM_BENCH_MONITOR`: also time replay steps with gradient monitoring.

These variables are described in
[examples/bench-am-common.scm](examples/bench-am-common.scm). Results are
in [examples/bench-results.md](examples/bench-results.md).

Small configurations, in ms per training step:

| Backend | ResNet | MLP-Mixer | Transformer |
|---|---|---|---|
| crunch | 205 | 84 | 82 |
| blas | 952 | 316 | 274 |
| scheme | 1217 | 346 | 352 |
| strict model | 1997 | 77 | 114 |

Compile the benchmarks before running them, and run them with
`OPENBLAS_NUM_THREADS=1`:

```bash
csc -O3 examples/bench-am-resnet-crunch.scm
```

## Tests

```bash
csi -s tests/run.scm
```

The runner runs each test file in its own process. To run only some
files, name them: `csi -s tests/run.scm test-gradcheck`. The test files
are:

- `test-gradcheck`: finite-difference gradient checks of the three strict
  models, and a check of the KV cache.
- `test-am-models`: each lazy model against its strict model. Weights are
  copied, then each model takes one SGD step on the same batch. The
  losses and the gradients must agree to 1e-12 in f64.
- `test-transformer`, `test-resnet`, `test-resnet-numerical`,
  `test-mlp-mixer`: unit tests of the strict components.

## Model notes

The notes explain each model from the basics, for readers who know Scheme
but are new to machine learning. They describe the architecture, API and
design of the strict model in detail and end with a summary of the lazy
version:

- [Transformer](notes/transformer_readme.md): RoPE, the KV cache,
  multi-head attention, the SwiGLU FFN, training and generation, and the
  mathematical details.
- [ResNet](notes/resnet_readme.md): basic and bottleneck blocks, batch
  normalization, the ResNet family, and training and evaluation.
- [MLP-Mixer](notes/mlp_mixer_readme.md): token and channel mixing, the
  API reference, and use in the Tiny Recursive Model (TRM).

The lazy models are documented in the header comments of their source
files: [am-transformer.scm](am-transformer.scm),
[am-resnet.scm](am-resnet.scm), [am-mlp-mixer.scm](am-mlp-mixer.scm) and
[am-common.scm](am-common.scm).

## License

LGPL-3. 
