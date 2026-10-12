# MLP-Mixer for Nanograd

The `nanograd-mlp-mixer` module provides an MLP-Mixer: a network that
mixes information across the cells of a small, fixed grid using only
multi-layer perceptrons (MLPs), with no attention. It suits the Tiny
Recursive Model (TRM), which applies one small network many times to
refine an answer.

This note covers the *strict* model, which computes values eagerly on
nanograd tensors. The lazy version, `nanograd-am-mlp-mixer`, is described
in [Lazy (array-morphisms) version](#lazy-array-morphisms-version).

## Architecture

The input is one `(seq_len, hidden_dim)` matrix. Each row is a *token*,
such as a cell or patch of a small image or puzzle, and each column is a
*feature* of that token. The strict model takes one sample at a time; it
has no batch axis.

An MLP here is two dense layers with GELU between them (nanograd uses the
tanh approximation of GELU). A block applies two MLPs along different axes:

- **Token mixing** mixes across tokens, separately for each feature. It
  transposes the input to `(hidden_dim, seq_len)`, applies an MLP to each
  row, and transposes back. Its hidden layer has
  `token-expansion * seq_len` units.
- **Channel mixing** mixes the features of each token, with the same
  weights for every token. Its hidden layer has
  `channel-expansion * hidden_dim` units.

Each MLP is preceded by a normalization (pre-norm) and followed by a skip
connection that adds the MLP's output to its input:

```
x -> Norm -> TokenMixing -> (+ x) -> Norm -> ChannelMixing -> (+) -> output
```

The norm is RMSNorm by default, or LayerNorm with `use-rmsnorm: #f`.
RMSNorm divides each row by its root mean square and multiplies by a
learned scale. LayerNorm also subtracts the row mean and adds a learned
shift.

A model is a stack of `n-blocks` blocks. It returns one feature vector per
token, with the same shape as its input. It has no patch embedding and no
classifier head: the caller supplies the `(seq_len, hidden_dim)` input and
adds any head (see [Classification](#classification)).

## API

```scheme
(import nanograd-autograd nanograd-layer nanograd-mlp-mixer)
```

| Procedure | Description |
|---|---|
| `(make-mixer-mlp input-dim hidden-dim #!key (dtype 'f32) (name "MixerMLP"))` | Dense `input-dim -> hidden-dim`, GELU, dense `hidden-dim -> input-dim` |
| `(make-token-mixing-mlp seq-len hidden-dim #!key (expansion-factor 4) (dtype 'f32) (name "TokenMixing"))` | Token mixing MLP |
| `(make-channel-mixing-mlp hidden-dim #!key (expansion-factor 4) (dtype 'f32) (name "ChannelMixing"))` | Channel mixing MLP |
| `(make-mlp-mixer-block seq-len hidden-dim #!key (token-expansion 4) (channel-expansion 4) (use-rmsnorm #t) (dtype 'f32) name)` | One block |
| `(make-mlp-mixer seq-len hidden-dim n-blocks #!key (token-expansion 4) (channel-expansion 4) (use-rmsnorm #t) (dtype 'f32) name)` | Full model |
| `(make-rmsnorm-layer shape #!key (eps 1e-6) (dtype 'f32) (name "RMSNorm"))` | RMSNorm over the last axis; `shape` is `(hidden-dim)` |
| `(make-layer-norm shape #!key (eps 1e-6) (dtype 'f32) (name "LayerNorm"))` | LayerNorm over the last axis |

All of these return layer objects that follow nanograd's layer protocol:
`forward`, `parameters`, `zero-grad-layer!`, `set-training-mode!`,
`set-eval-mode!`, `layer-name` and `layer?`. A full model also answers
`mlp-mixer?`, `(get-blocks model)` and `(num-blocks model)`.

Within each block, the parameters are in this order: token MLP
(W1, b1, W2, b2), channel MLP (W1, b1, W2, b2), first norm, second norm.

```scheme
;; 6x6 grid (36 tokens), 128 features per token, 2 blocks
(define mixer (make-mlp-mixer 36 128 2))

;; data: f32vector of 36 * 128 numbers
(define output (forward mixer (make-tensor32 data '(36 128))))   ; (36 128)
```

## Parameter counts

For a 6x6 grid (L = 36 tokens, D = 128 features), expansion 4, including
biases:

| Part | Parameters |
|---|---|
| Token mixing MLP | 10,548 |
| Channel mixing MLP | 131,712 |
| One block (both MLPs and two RMSNorms) | 142,516 |
| Model with 2 blocks | 285,032 |

`tests/test-mlp-mixer.scm` checks these counts. For comparison,
multi-head attention on the same input has 4 (D^2 + D) = 66,048
parameters in its Q, K, V and output projections.

Token mixing grows with L^2 and attention with D^2, so the Mixer is the
smaller of the two when the grid is small (L < D). The TRM paper reports
87.4% accuracy for an MLP-Mixer against 74.7% for attention on the
Sudoku-Extreme benchmark (9x9 grids). In exchange, the number of tokens is
fixed when the model is built, and every token can influence every other
one: there is no masking.

## Usage patterns

### Classification

Average the token vectors and apply a dense layer:

```scheme
(define mixer (make-mlp-mixer 36 128 2))
(define head (make-dense-layer 128 10))

;; Mean over tokens as a product with a (1, 36) row of 1/36
(define averager
  (make-tensor32 (make-f32vector 36 (/ 1.0 36)) '(1 36) requires-grad?: #f))

(define features (forward mixer input))           ; (36 128), supplied by the caller
(define pooled (matmul-op averager features))     ; (1 128)
(define logits (forward head pooled))             ; (1 10)
```

### TRM recursive refinement

TRM updates a latent state `z` many times from the input `x`, the current
answer `y` and `z` itself. To keep memory small, most updates are
*detached*: the result is wrapped in a new tensor that shares its data but
has no graph history, so `backward!` stops there. Only the last few
updates keep their graph for the backward pass.

```scheme
(define (detach t)
  "A tensor with the data of T and no graph history: gradients stop here."
  (make-base-tensor (tensor-data t) (tensor-shape t) (tensor-dtype t) #t))

(define (iterate f x n #!key (detach? #f))
  "Apply F to X N times; with DETACH?, detach every result."
  (let loop ((i 0) (x x))
    (if (= i n)
        x
        (let ((y (f x)))
          (loop (+ i 1) (if detach? (detach y) y))))))

(define net (make-mlp-mixer 36 128 2))
(define (update-z x y z) (forward net (add (add x y) z)))

;; 6 detached updates, then 2 with gradients
(define z-final
  (iterate (lambda (z) (update-z x y z))
           (iterate (lambda (z) (update-z x y z)) z-init 6 detach?: #t)
           2))

(backward! (mean-tensor z-final))   ; goes through the last 2 updates only
```

Without detaching, every update stays in memory until `backward!`, so 1000
updates use 1000 times the memory of one. With detaching, memory stays
constant.

### Deep supervision

*Deep supervision* computes a loss after every refinement step, not only
the last one, so every step is pushed toward the target:

```scheme
(define states                       ; state after each of 4 steps
  (let loop ((i 0) (z z-init) (acc '()))
    (if (= i 4)
        (reverse acc)
        (let ((z2 (update-z x y z)))
          (loop (+ i 1) z2 (cons z2 acc))))))

(define losses (map (lambda (z) (mse-loss z target)) states))
(backward! (fold (lambda (l acc) (add acc l)) (car losses) (cdr losses)))
```

## Lazy (array-morphisms) version

`nanograd-am-mlp-mixer` provides `make-am-mlp-mixer`, which computes the
same function as `make-mlp-mixer` on a whole batch at once. Its parameters
are in the same order and have the same sizes, so `am-copy-parameters!`
from `nanograd-am-common` can copy the weights of a strict model into a
lazy one. The two models initialize their weights differently, so copying
is needed to make them agree.

The lazy forward pass builds a graph of the computation, which `am-training-step/ssa` compiles once into a program for
the whole training step and then replays on every new batch.

Differences from the strict model:

- **Batches**: B samples form one token-major `[T*B, D]` matrix; row
  `t*B + b` is token t of sample b.
  `(am-mixer-pack-batch samples seq-len hidden-dim dtype)` and
  `(am-mixer-unpack-batch data batch seq-len hidden-dim dtype)` convert
  between this layout and one `(T, D)` vector per sample.
- **Token mixing** uses left matrix products on the `[T, B*D]` view of
  the same data, `W1[Ht, T] @ X[T, B*D]` and then `W2[T, Ht] @ h`, so it
  needs no transposes.
- **Classifier head**: with `num-classes:`, the model averages the
  features over the tokens and adds a dense layer that gives
  `[B, num-classes]` scores. The head's parameters come last.
- **Norm epsilon**: the block and model take `norm-eps:` (default 1e-6).

The module also exports `make-am-token-mixing-mlp`,
`make-am-channel-mixing-mlp`, `make-am-mlp-mixer-block` and
`am-mlp-mixer?`. See the header comment of `am-mlp-mixer.scm` and
`examples/bench-am-mlp-mixer-crunch.scm`.

## Examples and tests

```bash
csi -s examples/mlp-mixer-examples.scm
```

The examples use 16 tokens (a 4x4 grid) and 32 features. They cover:

1. a forward pass and mean pooling over tokens;
2. TRM-style refinement with detached updates;
3. deep supervision;
4. parameter counts;
5. a training loop with Adam.

```bash
csi -s tests/run.scm test-mlp-mixer test-gradcheck test-am-models
```

`test-mlp-mixer` checks each component, blocks with different
configurations, full models, gradient flow and parameter counts.
`test-gradcheck` compares gradients with finite differences.
`test-am-models` checks that the lazy model agrees with the strict one.

## References

1. **MLP-Mixer**: "MLP-Mixer: An all-MLP Architecture for Vision" (Tolstikhin et al., 2021)
2. **TRM**: "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025)
3. **RMSNorm**: "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)
