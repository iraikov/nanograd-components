# ResNet for Nanograd

The `nanograd-resnet` module provides trainable ResNet (Residual Network)
image classifiers. Given an image, a model returns one score (*logit*) per
class, and the highest score is the prediction. The module provides the
ImageNet-style ResNet-18, 34, 50, 101 and 152, and `make-resnet-cifar`, a
configurable variant for small images such as the 32x32 images of
CIFAR-10.

This note covers the *strict* models, which compute values eagerly on
nanograd tensors. The lazy version, `nanograd-am-resnet`, is described in
[Lazy (array-morphisms) version](#lazy-array-morphisms-version).

## Architecture

A ResNet is a stack of *residual blocks*. Each block adds its input to its
output through a *skip connection*:

```
x -> block layers F -> (+) -> ReLU -> y
|                       ^
+-----------------------+
     skip connection
```

The block therefore learns only the difference `F(x) = y - x`. A block
that is not useful can learn `F(x) = 0` and pass its input on unchanged.
The skip connection also passes gradients straight back, so they do not
fade away in deep networks.

When a block changes the number of channels or the image size, its input
cannot be added to its output directly. The skip connection then goes
through a *downsample* layer: a 1x1 convolution with the block's stride,
followed by batch normalization (BN).

### Block types

**Basic block** (ResNet-18, ResNet-34 and `make-resnet-cifar`):

```
3x3 conv, BN, ReLU -> 3x3 conv, BN -> (+ skip) -> ReLU
```

The first convolution carries the block's stride; stride 2 halves the
height and width.

**Bottleneck block** (ResNet-50, 101 and 152):

```
1x1 conv (reduce), BN, ReLU -> 3x3 conv, BN, ReLU -> 1x1 conv (expand), BN -> (+ skip) -> ReLU
```

The middle 3x3 convolution carries the stride and works on 4 times fewer
channels than the block's output. For 256 channels in and out, its weights
total 69,632, against 589,824 for a single 3x3 convolution. This keeps
the deep models affordable.

### ImageNet-style models

```
Input (3, H, W)
  -> 7x7 conv, stride 2, 64 channels -> BN -> ReLU
  -> 3x3 max pooling, stride 2
  -> Layer 1:  64 channels, stride 1
  -> Layer 2: 128 channels, stride 2
  -> Layer 3: 256 channels, stride 2
  -> Layer 4: 512 channels, stride 2
  -> global average pooling -> dense layer (num_classes) -> logits
```

Each layer is a sequence of blocks. With bottleneck blocks, the output
channel counts of layers 1 to 4 are 4 times larger: 256, 512, 1024 and
2048. *Global average pooling* averages each channel over the image, so
the classifier sees one number per channel, whatever the image size.

The stem and layers shrink the image by a factor of 32. These models
therefore reject inputs smaller than 32x32, and print a warning when the
height is below 64. The standard ImageNet size is 224x224.

### CIFAR-style model

```
Input (C, H, W)
  -> 3x3 conv, stride 1 -> BN -> ReLU
  -> one layer of basic blocks per entry of STAGES, with (list-ref WIDTHS i)
     channels; every layer after the first starts with stride 2
  -> global average pooling -> dense layer (num_classes) -> logits
```

There is no max pooling, so the image shrinks only by
`2^(length STAGES - 1)`.

## API

```scheme
(import nanograd-autograd nanograd-layer nanograd-resnet)
```

| Procedure | Description |
|---|---|
| `(make-resnet-18 num-classes #!key (dtype 'f32))`, and likewise `-34`, `-50`, `-101`, `-152` | ImageNet-style models |
| `(make-resnet-cifar stages widths num-classes #!key (in-channels 3) (dtype 'f32) (name "ResNet-CIFAR"))` | CIFAR-style model |
| `(make-resnet block-fn layers num-classes #!key (dtype 'f32) (name "ResNet"))` | ImageNet-style model from a block constructor and four block counts |
| `(make-residual-layer block-fn in-channels out-channels num-blocks #!key (stride 1) (dtype 'f32) (name "Layer"))` | A layer of blocks. The first block gets the stride, and also a downsample layer when the stride or channel count changes |
| `(make-residual-block in out #!key (stride 1) (downsample #f) (dtype 'f32) (name "BasicBlock"))` | Basic block |
| `(make-bottleneck-block in out #!key (stride 1) (downsample #f) (dtype 'f32) (name "Bottleneck"))` | Bottleneck block; `out` is the channel count after expansion |
| `(make-downsample in out stride dtype)` | 1x1 convolution and BN for the skip connection; `dtype` is positional |

The predicates are `resnet?`, `residual-block?` and `bottleneck-block?`.
All the models and blocks follow nanograd's layer protocol (`forward`,
`parameters`, `zero-grad-layer!`, `set-training-mode!`,
`set-eval-mode!`). The models accept a single image `(C, H, W)` or a batch
`(N, C, H, W)`.

The convolutions, batch normalization, pooling, activations and the dense
classifier come from `nanograd-layer` and `nanograd-autograd`.

### Parameter counts

These are the counts as built by this module. Every convolution here has a
bias vector, which the standard ResNet omits because the following BN
cancels it. That is why the counts differ slightly from the published
ones. The Top-1 column gives published ImageNet accuracies for these
architectures; this implementation has not reproduced them.

| Model | Blocks per layer | Parameters (1000 classes) | Parameters (10 classes) | Published Top-1 |
|---|---|---|---|---|
| ResNet-18 | 2, 2, 2, 2 | 11,694,312 | 11,186,442 | 69.8% |
| ResNet-34 | 3, 4, 6, 3 | 21,806,184 | 21,298,314 | 73.3% |
| ResNet-50 | 3, 4, 6, 3 | 25,583,592 | 23,555,082 | 76.1% |
| ResNet-101 | 3, 4, 23, 3 | 44,601,832 | 42,573,322 | 77.4% |
| ResNet-152 | 3, 8, 36, 3 | 60,268,520 | 58,240,010 | 78.3% |

`make-resnet-cifar` has 20,122 parameters with stages `(1 1 1)`, widths
`(8 16 32)` and 10 classes. With widths `(16 32 64)` it has 78,378. In
f32, the parameters of ResNet-18 take 47 MB and those of ResNet-152
241 MB. Training needs several times more: one gradient per parameter,
the optimizer's state, and the intermediate results of the forward pass.

The strict models run one operation at a time, so they suit small images
and the smaller models. For training, the lazy models are much faster; see
`examples/bench-results.md`.

## Batch normalization

Batch normalization (`make-batch-norm-2d` from `nanograd-layer`, with
defaults `epsilon: 1e-5` and `momentum: 0.1`) keeps the values in each
channel at a similar scale. It has two modes:

- **Training mode** (`(set-training-mode! model #t)`): normalizes each
  channel with the mean and variance of the current batch over all
  pixels. It then scales and shifts the result by the learned gamma and
  beta, and updates the running statistics:
  `running <- (1 - momentum) running + momentum batch`.
- **Evaluation mode** (`(set-eval-mode! model)`): uses the running
  statistics instead, so the output for an image does not depend on the
  other images in the batch. The output records no gradients.

After a short training run, the running statistics may not yet match the
batch statistics. Accuracy in evaluation mode can then lag behind
accuracy in training mode.

## Training and evaluation

The following sketch shows the training loop. It assumes the caller
provides `num-epochs`, `training-batches` and the batch accessors. See
`examples/resnet_example.scm` for complete, runnable code.

```scheme
(define model (make-resnet-cifar '(1 1 1) '(16 32 64) 10 dtype: 'f32))
(define optimizer (make-adam (parameters model) learning-rate: 0.001))

(set-training-mode! model #t)
(do ((epoch 1 (+ epoch 1)))
    ((> epoch num-epochs))
  (for-each
   (lambda (batch)
     ;; images in (N, 3, 32, 32) order; targets one-hot, (N, 10)
     (let* ((x (make-tensor32 (batch-images batch) (list n 3 32 32)
                              requires-grad?: #f))
            (t (make-tensor32 (batch-targets batch) (list n 10)
                              requires-grad?: #f))
            (logits (forward model x))                          ; (N, 10)
            (loss (cross-entropy-loss logits t from-logits: #t)))
       (backward! loss)            ; gradients of the mean loss over the batch
       (step! optimizer)
       (zero-grad-layer! model)))
   training-batches))
```

With `from-logits: #t`, the loss applies softmax internally, in a
numerically stable way. To change the learning rate between epochs, call
`(set-learning-rate! optimizer rate)`. For evaluation, call
`(set-eval-mode! model)`, run `forward`, and take the index of the largest
logit as the predicted class.

### Transfer learning

A model trained on one task can give a head start on another task. Copy
every parameter except the final classifier (the last two parameters: its
weight and bias) into a new model, then train with a small learning rate:

```scheme
(define (copy-backbone! src dst)          ; needs (srfi 1) for take
  (let ((n (- (length (parameters src)) 2)))
    (for-each (lambda (s d)
                (let ((sv (tensor-data s)) (dv (tensor-data d)))
                  (do ((i 0 (+ i 1))) ((= i (f32vector-length sv)))
                    (f32vector-set! dv i (f32vector-ref sv i)))))
              (take (parameters src) n) (take (parameters dst) n))))
```

Example 5 of `examples/resnet_example.scm` does this and compares the
result with training from scratch.

`forward` returns only the logits. To get the pooled features that feed
the classifier, build the network from its parts (`make-residual-layer`,
`global-avg-pool2d`) without the final dense layer.

## Lazy (array-morphisms) version

`nanograd-am-resnet` provides `make-am-resnet`, built from basic blocks.
With its default `'cifar` stem, it computes the same function as
`make-resnet-cifar` in training mode. Its parameters are in the same order
and have the same sizes, so `am-copy-parameters!` from
`nanograd-am-common` can copy the weights of a strict model into a lazy
one.

The lazy forward pass builds a graph of the computation, which
`am-training-step/ssa` compiles once into a program for the whole training
step and then replays on every new batch. `am-forward/ssa` does the same
for inference.

| Procedure | Description |
|---|---|
| `(make-am-resnet stages widths num-classes #!key (in-channels 3) (stem 'cifar) (dtype 'f32) (name "AM-ResNet"))` | Model; `stem: 'strided` makes the 3x3 stem use stride 2 |
| `(make-am-resnet-8 num-classes #!key (dtype 'f32))` | Stages `(1 1 1)`, widths 16/32/64 |
| `(make-am-resnet-18-narrow num-classes #!key (dtype 'f32))` | Stages `(2 2 2 2)`, widths 16/32/64/128 |
| `make-am-residual-block`, `make-am-residual-stage`, `make-am-downsample` | Building blocks; `make-am-downsample` takes `dtype:` as a keyword |
| `(am-nchw->nhwc data n c h w dtype)` | Converts image data from the strict layout |

Differences from the strict models:

- **Layout**: images are NHWC, `[N, H, W, C]`, with the channel last.
- **Batch normalization** (`make-am-batch-norm-nhwc`, eps 1e-5) always
  uses the statistics of the current batch. There are no running
  statistics, `set-training-mode!` and `set-eval-mode!` do nothing, and
  inference should run on batches.
- **No max pooling and no 7x7 stem**: the max-pooling gradient needs a
  reduce-max operation that the SSA engine does not support.
- **Pooling**: global average pooling is a product with a constant
  averaging matrix.
- **Fixed shapes**: a compiled program is fixed to its input shape,
  including the batch size.
- **No saving**: `layer->serializable` returns `#f`.

See `examples/am_resnet_example.scm` and the header comment of
`am-resnet.scm`.

## Examples and tests

```bash
csi -s examples/resnet_example.scm
```

The examples cover:

1. classification of synthetic striped images with a small CIFAR-style
   ResNet;
2. training with a step learning-rate schedule;
3. parameter counts of ResNet-18, 34 and 50, and a forward pass;
4. output statistics of an untrained ResNet-18;
5. transfer learning to new classes.

`examples/am_resnet_example.scm` trains the lazy model, and
`examples/bench-am-resnet-crunch.scm` benchmarks it.

```bash
csi -s tests/run.scm test-resnet test-resnet-numerical test-gradcheck test-am-models
```

## References

1. **ResNet**: "Deep Residual Learning for Image Recognition" (He et al., 2015)
2. **Identity mappings**: "Identity Mappings in Deep Residual Networks" (He et al., 2016)
3. **Batch normalization**: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (Ioffe and Szegedy, 2015)
