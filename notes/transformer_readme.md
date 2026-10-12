# Transformer for Nanograd

The `nanograd-transformer` module provides a trainable LLAMA2-style
language model: it reads a sequence of tokens and predicts the token that
comes next. A token is a small unit of text, such as a character or a word
piece, represented by an integer id.

This note covers the *strict* model, which computes values eagerly on
nanograd tensors and processes one sequence at a time. The lazy version,
`nanograd-am-transformer`, is described in
[Lazy (array-morphisms) version](#lazy-array-morphisms-version).

## Architecture

```
Input token ids
  -> token embedding lookup            (vocab_size, d_model) table
  -> Transformer block 1 .. N
  -> RMSNorm
  -> output projection (logits)        product with the transposed embedding table
```

Each block uses *pre-normalization*: RMSNorm is applied before each
sublayer, and the sublayer's output is added back to its input (the
*residual* connection):

```
x -> RMSNorm -> Attention -> (+ x) -> RMSNorm -> FFN -> (+) -> output
```

The output has one row of *logits* per position: one score for every token
in the vocabulary. Softmax turns the logits into probabilities.

Differences from LLAMA2:

- **Training**: every operation has a backward pass.
- **Causal mask is a switch**: with the mask, each token can only look at
  itself and earlier tokens, as next-token prediction requires.
  `make-transformer` uses it by default (`causal?: #t`). The stand-alone
  attention and block constructors default to `causal?: #f`.
- **Biases**: the linear projections have bias vectors, which LLAMA2
  omits.
- **Tied output projection**: the output projection reuses the token
  embedding matrix.
- **No grouped-query attention**: every head has its own keys and values.

## Components

### RoPE (Rotary Position Embedding)

```scheme
(define rope (make-rope-embedding 128 16 base: 10000.0 dtype: 'f32))
;;                                max_seq_len, head_size
```

Attention on its own does not know the order of the tokens. RoPE adds
this information by rotating each query and key vector by an angle that
grows with the token's position. It splits each head's vector into pairs
`(x_2i, x_2i+1)` and rotates pair `i` at position `m` by the angle
`theta = m / base^(2i / head_size)`:

```
(a, b) -> (a cos theta - b sin theta,  a sin theta + b cos theta)
```

After rotation, the dot product of a query and a key depends only on the
distance between their positions. RoPE has no parameters; the cos and sin
tables are computed once. The backward pass rotates the gradient by
`-theta`.

`(rope-rotate-rows x rope start-pos)` rotates all rows of a `(seq_len, d)`
matrix whose first row is at position `start-pos`.
`(apply-rope-rotation Q K rope pos)` rotates a single query and key of
shape `(n_heads, head_size)`.

### Multi-head self-attention

```scheme
(define attention (make-multihead-attention 64 4 rope: rope causal?: #t dtype: 'f32))
;;                                          d_model, n_heads
```

Attention lets every token collect information from other tokens. Each
token produces a *query* ("what am I looking for?"), a *key* ("what do I
contain?") and a *value* ("what do I pass on?"). A token's output is a
weighted average of the values, weighted by how well its query matches
each key.

1. Three dense layers project the input `(seq_len, d_model)` to Q, K
   and V.
2. RoPE rotates Q and K.
3. Head `h` uses columns `[h*hs, (h+1)*hs)` of Q, K and V, where
   `hs = d_model / n_heads`.
4. Each head computes `softmax(Q_h K_h^T / sqrt(hs) + M) V_h`. When
   `causal?` is true, the mask `M` adds -1e9 to the scores of later keys,
   which makes their weights zero.
5. The heads' outputs are put side by side, and the output projection is
   applied.

`(multihead-attention-core Q K V n-heads #!key (causal? #f))` does steps 3
to 5 without the output projection, on plain matrices.
`(scaled-dot-product-attention Q K V head-size seq-len #!key causal?)`
does the same on `(len, n_heads, head_size)` tensors.

### Feed-forward network (SwiGLU)

```scheme
(define ffn (make-ffn-block 64 256 dtype: 'f32))   ;; d_model, d_ff
```

After attention has mixed information between tokens, the feed-forward
network transforms each token's vector on its own:

```
FFN(x) = W2 (SiLU(W1 x) * W3 x)
```

Here `SiLU(x) = x * sigmoid(x)`, `*` is element-wise multiplication, and
W1, W2 and W3 are dense layers. The branch `SiLU(W1 x)` acts as a gate that
scales `W3 x` element by element. `d_ff` defaults to `4 * d_model` in
`make-transformer`.

### RMSNorm and the block

RMSNorm divides each row by `sqrt(mean(x^2) + eps)`, with `eps = 1e-5`,
and multiplies the result by a learned weight vector.

```scheme
(define block (make-transformer-block 64 4 256 rope: rope causal?: #t dtype: 'f32))
;;                                    d_model, n_heads, d_ff
```

### Complete model

```scheme
(make-transformer vocab-size max-seq-len d-model n-layers n-heads
                  #!key (d-ff #f) (use-rope? #t) (rope-base 10000.0)
                        (causal? #t) (dtype 'f32) (name "Transformer"))
```

The parameters are in this order:

1. the token embedding `(vocab_size, d_model)`;
2. the final norm weight;
3. for each block: its two norm weights, its attention parameters (WQ,
   WK, WV, WO, each a weight and a bias), and its feed-forward parameters
   (W1, W2, W3, each a weight and a bias).

`forward` on the model takes a list or a tensor of token ids. The
attention and block `forward` methods also accept `position:`,
`kv-cache:` and `layer-idx:` keywords.

The module also exports the helpers `causal-mask`, `embedding-lookup`,
`slice-columns`, `concat-columns`, `get-frequencies`, `rope-head-size`,
`cache-kv-pair!` and `get-cached-kv`. Its predicates are
`rope-embedding?`, `kv-cache?`, `multihead-attention?`, `ffn-block?`,
`transformer-block?` and `transformer?`.

## Training

```scheme
(import (srfi 1) (srfi 4)
        nanograd-autograd nanograd-layer nanograd-optimizer
        nanograd-transformer)

(define model (make-transformer vocab-size 128 64 4 4 dtype: 'f32))
(define optimizer (make-adam (parameters model) learning-rate: 0.001))

(define (ids->tensor ids)
  (make-tensor32 (list->f32vector (map exact->inexact ids))
                 (list (length ids)) requires-grad?: #f))

;; Each sequence predicts its own next tokens
(do ((epoch 1 (+ epoch 1)))
    ((> epoch num-epochs))
  (for-each
   (lambda (sequence)
     (let* ((logits (forward model (drop-right sequence 1)))   ; (seq_len, vocab)
            (loss (cross-entropy-loss logits (ids->tensor (cdr sequence))
                                      from-logits: #t)))
       (backward! loss)
       (step! optimizer)
       (zero-grad-layer! model)))
   training-data))
```

*Cross-entropy* measures how much probability the model gave to the
correct next token. Compute the loss from the model's logits tensor
itself, as shown above. Copying the numbers into a new tensor would cut
the link that `backward!` follows, and the model would not learn.

The strict model processes one sequence per `forward` call. Joining
sequences into one would let them attend to each other. To train on a
batch, average the losses of several sequences before calling `backward!`:

```scheme
(let ((losses (map sequence-loss batch)))
  (backward! (scale-op (fold add (car losses) (cdr losses))
                       (/ 1.0 (length losses)))))
```

Gradients add up until they are zeroed, so several `backward!` calls
before one `step!` also act like a larger batch. Use `'f64` when checking
gradients, and `'f32` otherwise: it is faster and uses half the memory.

## Generation with the KV cache

```scheme
(define cache (make-kv-cache 4 4 128 16 dtype: 'f32))
;;                           n_layers, n_heads, max_seq_len, head_size
```

When the model generates text one token at a time, the keys and values of
earlier tokens do not change. The KV cache stores them for every layer,
so each new token needs only its own keys and values. With a cache,
`forward` appends its tokens after those already cached and returns the
logits of its last token only, with shape `(vocab)`:

```scheme
(define (argmax vec)
  (let loop ((i 1) (best 0))
    (cond ((= i (f32vector-length vec)) best)
          ((> (f32vector-ref vec i) (f32vector-ref vec best)) (loop (+ i 1) i))
          (else (loop (+ i 1) best)))))

(define prompt-tokens '(1 2 3 4 5))

;; The first call fills the cache with the prompt; later calls feed one token
(let loop ((logits (forward model prompt-tokens kv-cache: cache))
           (position (length prompt-tokens)))
  (when (< position max-length)
    (let ((next-token (argmax (tensor-data logits))))    ; greedy choice
      (printf "~A " next-token)
      (loop (forward model (list next-token) kv-cache: cache)
            (+ position 1)))))
```

`kv-cache-length` gives the number of cached positions, and `clear-cache!`
empties the cache for a new text. A cache holds one text, so each
candidate text needs its own cache. Cached keys and values are stored as
constants, so use the cache only for generation, never for training.

In Example 4 of `examples/transformer_example.scm`, generating 30 tokens
after a 20-token context runs about 10 times faster with the cache. The
ratio depends on the machine.

## Gradient formulas

Scaled dot-product attention, with `O` the output and `*` element-wise:

```
S = Q K^T / sqrt(d_k),   P = softmax(S) row by row,   O = P V

dL/dV = P^T dL/dO
dL/dP = dL/dO V^T
dL/dS = P * (dL/dP - rowsum(dL/dP * P))
dL/dQ = dL/dS K / sqrt(d_k)
dL/dK = dL/dS^T Q / sqrt(d_k)
```

SwiGLU:

```
h1 = W1 x,  h3 = W3 x,  g = SiLU(h1),  FFN(x) = W2 (g * h3)

dL/dh    = W2^T dL/dFFN
dL/dx    = W1^T (SiLU'(h1) * h3 * dL/dh) + W3^T (g * dL/dh)
SiLU'(x) = sigmoid(x) (1 + x (1 - sigmoid(x)))
```

RoPE: `dL/dq = R^T dL/d(Rq)`, where `R^T` is the rotation by `-theta`.

## Lazy (array-morphisms) version

`nanograd-am-transformer` provides:

```scheme
(make-am-transformer vocab-size seq-len d-model n-layers n-heads
                     #!key (d-ff #f) (use-rope? #t) (rope-base 10000.0)
                           (causal? #t) (norm-eps 1e-5) (dtype 'f32)
                           (name "AM-Transformer"))
```

It computes the same function as `make-transformer`, and its parameters
are in the same order. `am-copy-parameters!` from `nanograd-am-common`
copies the weights of a strict model into a lazy one.

The lazy model's forward pass builds a description of the computation
(a graph). `am-training-step/ssa` compiles this graph once into a program
for the whole training step, then replays the program on every new batch.
`am-forward/ssa` does the same for inference. Both come from the core
module `nanograd-array-morphisms`.

Differences from the strict model:

- **Batches**: B sequences of length T form one `[B*T, vocab]` matrix of
  *one-hot* rows. A one-hot row is all zeros except for a 1 in the column
  of its token, so the embedding lookup becomes a matrix product with the
  embedding table. `(am-tokens->one-hot sequences vocab-size dtype)` and
  `(am-next-token-targets sequences vocab-size dtype)` build these
  matrices.
- **Heads**: a product with a constant 0/1 matrix selects each head's
  columns. A block-diagonal mask keeps the sequences of a batch apart.
- **Fixed shapes**: `seq-len` is the fixed length T, not a maximum, and
  a compiled program is fixed to B and T. Shorter sequences are padded to
  T.
- **No KV cache**: to generate each new token, the model runs the compiled
  forward program on a padded window. Because attention is causal, the
  padding does not change the logits of the real tokens.

The module also exports `make-am-attention`, `make-am-swiglu-ffn`,
`make-am-transformer-block` and `am-transformer?`. See
`examples/am_transformer_example.scm` and the header comment of
`am-transformer.scm`.

## Examples and tests

```bash
csi -s examples/transformer_example.scm
```

The examples are:

1. **Character-level language model**: trains on five short sentences and
   completes prompts such as "the c" using the KV cache.
2. **Copy task**: the input is `x1 .. xk SEP x1 .. xk`. After the
   separator, the model must find each token in the first half, which
   needs attention.
3. **RoPE comparison**: the model is trained with and without RoPE on a
   task that depends on token order.
4. **KV cache**: generation with and without the cache, which must give
   the same tokens.
5. **Gradient clipping**: an unstable SGD run, with and without clipping
   of the global gradient norm.

The training examples record gradient norms at every step with
`nanograd-diagnostics`, through `examples/gradient-monitoring.scm`.

```bash
csi -s tests/run.scm test-transformer test-gradcheck test-am-models
```

`test-gradcheck` uses f64. For a few sampled entries of each parameter,
it compares the analytic gradient with central finite differences of a
random linear function of the output. `test-am-models` checks that the
lazy model agrees with the strict one.

## Limitations

- Attention time and memory grow with the square of the sequence length,
  and the full attention weights are kept for the backward pass.
- The strict model processes one sequence per call.
- There is no grouped-query attention.

## References

1. **LLAMA2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023)
2. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
3. **Attention**: "Attention Is All You Need" (Vaswani et al., 2017)
4. **GLU variants**: "GLU Variants Improve Transformer" (Shazeer, 2020)
5. **Pre-norm**: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
