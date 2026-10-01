;;; am-resnet.scm
;;;
;;; ResNet with basic residual blocks on lazy (array-morphisms) tensors,
;;; for SSA training.
;;;
;;; Activations are NHWC tensors [N, H, W, C]: make-am-conv2d-layer reads
;;; NHWC input and writes NHWC output, so layers chain without layout
;;; changes.  Each block is
;;;
;;;   conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> (+ shortcut) -> ReLU
;;;
;;; where the shortcut is the identity, or a strided 1x1 convolution
;;; followed by BN when the shape changes.  Batch normalization always
;;; uses the statistics of the current batch (see make-am-batch-norm-nhwc).
;;;
;;; There is no max pooling: its gradient needs the reduce-max VJP, which
;;; the SSA engine does not have.  The CIFAR-style stem is a 3x3 stride-1
;;; convolution; the 'strided stem uses stride 2 instead.
;;;
;;; The model computes the same function as nanograd-resnet's
;;; make-resnet-cifar (for the 'cifar stem, in training mode), with the
;;; parameters in the same order and with the same element counts, so
;;; am-copy-parameters! can transfer weights between the two.  Strict
;;; conv weights [out, in, kh, kw] and lazy ones [out, in*kh*kw] share
;;; the same row-major element order.

(module nanograd-am-resnet

  (make-am-residual-block
   make-am-downsample
   make-am-residual-stage
   make-am-resnet
   am-resnet?
   make-am-resnet-8
   make-am-resnet-18-narrow
   am-nchw->nhwc)

  (import scheme (chicken base) (chicken format))
  (import (only srfi-1 fold iota append-map last))
  (import srfi-4)
  (import yasos)
  (import array-morphisms-core)
  (import (prefix array-morphisms-grad am:))
  (import nanograd-autograd)
  (import nanograd-layer)
  (import nanograd-array-morphisms)
  (import nanograd-am-common)

  (define-predicate am-resnet?)

  (define (conv in out k stride padding dtype)
    (make-am-conv2d-layer in out k stride: stride padding: padding dtype: dtype))

  (define (relu-lt x) (lazy (am:var-relu (mv x))))

  (define (make-am-downsample in-channels out-channels stride #!key (dtype 'f32))
    "Shortcut for a block that changes shape: 1x1 conv (stride STRIDE)
     then BN.  Parameters: conv W b, BN gamma beta."
    (let ((c (conv in-channels out-channels 1 stride 0 dtype))
          (bn (make-am-batch-norm-nhwc out-channels dtype: dtype)))
      (object
       ((layer? self) #t)
       ((layer-name self) "AM-Downsample")
       ((parameters self) (append (parameters c) (parameters bn)))
       ((zero-grad-layer! self) (zero-grad-layer! c) (zero-grad-layer! bn))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x) (forward bn (forward c x)))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-residual-block in-channels out-channels
                                  #!key (stride 1) (downsample #f) (dtype 'f32)
                                  (name "AM-BasicBlock"))
    "Basic residual block.  Parameters: conv1, bn1, conv2, bn2, then the
     downsample layer if there is one."
    (let* ((conv1 (conv in-channels out-channels 3 stride 1 dtype))
           (bn1 (make-am-batch-norm-nhwc out-channels dtype: dtype))
           (conv2 (conv out-channels out-channels 3 1 1 dtype))
           (bn2 (make-am-batch-norm-nhwc out-channels dtype: dtype))
           (parts (append (list conv1 bn1 conv2 bn2)
                          (if downsample (list downsample) '()))))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((parameters self) (append-map parameters parts))
       ((zero-grad-layer! self) (for-each zero-grad-layer! parts))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x)
        (let* ((h (relu-lt (forward bn1 (forward conv1 x))))
               (r (forward bn2 (forward conv2 h)))
               (shortcut (if downsample (forward downsample x) x)))
          (lazy (am:var-relu (am:var+ (mv r) (mv shortcut))))))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-residual-stage in-channels out-channels n-blocks
                                  #!key (stride 1) (dtype 'f32) (name "AM-Stage"))
    "N-BLOCKS basic blocks; the first has stride STRIDE and a downsample
     shortcut when the stride or the channel count changes."
    (let* ((ds (and (or (not (= stride 1)) (not (= in-channels out-channels)))
                    (make-am-downsample in-channels out-channels stride dtype: dtype)))
           (blocks (cons (make-am-residual-block in-channels out-channels
                                                 stride: stride downsample: ds
                                                 dtype: dtype
                                                 name: (sprintf "~A-Block0" name))
                         (map (lambda (i)
                                (make-am-residual-block out-channels out-channels
                                                        dtype: dtype
                                                        name: (sprintf "~A-Block~A" name (+ i 1))))
                              (iota (- n-blocks 1))))))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((parameters self) (append-map parameters blocks))
       ((zero-grad-layer! self) (for-each zero-grad-layer! blocks))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x) (fold (lambda (b h) (forward b h)) x blocks))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  ;; Global average pooling of an NHWC morph-variable [N, H, W, C] as the
  ;; product A @ X of the [N*H*W, C] view X with the constant averaging
  ;; matrix A [N, N*H*W] (A[n, n*H*W + k] = 1/(H*W)).  Both the product
  ;; and its gradient A^T G are matrix products, which the BLAS backends
  ;; run; the equivalent mean over axes 1 and 2 would broadcast its
  ;; gradient over the middle axes, which has no fast replay path.
  (define pooling-matrices '())   ; alist (n hw dtype) -> constant

  (define (global-average-pool x)
    (let* ((shape (vector->list (morph-shape (am:var-value x))))
           (n (car shape)) (hw (* (cadr shape) (caddr shape))) (c (cadddr shape))
           (dtype (morph-dtype (am:var-value x)))
           (key (list n hw dtype))
           (a (or (let ((e (assoc key pooling-matrices))) (and e (cdr e)))
                  (let ((m (am-const-fill (list n (* n hw)) dtype
                                          (lambda (i)
                                            (if (= (quotient (modulo i (* n hw)) hw)
                                                   (quotient i (* n hw)))
                                                (/ 1.0 hw)
                                                0.0)))))
                    (set! pooling-matrices (cons (cons key m) pooling-matrices))
                    m))))
      (am:var-matmul a (am:var-reshape x (list (* n hw) c)))))

  (define (make-am-resnet stages widths num-classes
                          #!key (in-channels 3) (stem 'cifar) (dtype 'f32)
                          (name "AM-ResNet"))
    "ResNet on NHWC input [N, H, W, IN-CHANNELS].

     Stem: 3x3 conv (stride 1 for 'cifar, 2 for 'strided), BN, ReLU.
     Then one stage per entry of STAGES, with (list-ref WIDTHS i)
     channels; every stage after the first starts with stride 2.
     Finally global average pooling over H and W (as a matrix product,
     see global-average-pool) and a dense head.
     Returns [N, NUM-CLASSES] logits."
    (let* ((w0 (car widths))
           (stem-conv (conv in-channels w0 3 (if (eq? stem 'strided) 2 1) 1 dtype))
           (stem-bn (make-am-batch-norm-nhwc w0 dtype: dtype))
           (stage-list (let loop ((ss stages) (ws widths) (in w0) (i 0) (acc '()))
                         (if (null? ss)
                             (reverse acc)
                             (loop (cdr ss) (cdr ws) (car ws) (+ i 1)
                                   (cons (make-am-residual-stage
                                          in (car ws) (car ss)
                                          stride: (if (= i 0) 1 2)
                                          dtype: dtype
                                          name: (sprintf "~A-Stage~A" name (+ i 1)))
                                         acc)))))
           (head (make-am-dense-layer (last widths) num-classes dtype: dtype))
           (parts (append (list stem-conv stem-bn) stage-list (list head))))
      (object
       ((layer? self) #t)
       ((am-resnet? self) #t)
       ((layer-name self) name)
       ((layer-input-size self) in-channels)
       ((layer-output-size self) num-classes)
       ((parameters self) (append-map parameters parts))
       ((zero-grad-layer! self) (for-each zero-grad-layer! parts))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x)
        (let* ((h (relu-lt (forward stem-bn (forward stem-conv x))))
               (h (fold (lambda (s h) (forward s h)) h stage-list)))
          (forward head (lazy (global-average-pool (mv h))))))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-resnet-8 num-classes #!key (dtype 'f32))
    "Three stages of one block, widths 16/32/64 (CIFAR-style ResNet-8)."
    (make-am-resnet '(1 1 1) '(16 32 64) num-classes dtype: dtype name: "AM-ResNet-8"))

  (define (make-am-resnet-18-narrow num-classes #!key (dtype 'f32))
    "ResNet-18 layout (2 2 2 2) with widths 16/32/64/128."
    (make-am-resnet '(2 2 2 2) '(16 32 64 128) num-classes dtype: dtype
                    name: "AM-ResNet-18-narrow"))

  (define (am-nchw->nhwc data n c h w dtype)
    "Copy an NCHW typed vector into a new NHWC typed vector."
    (let ((out (allocate-typed-vector dtype (* n c h w))))
      (do ((b 0 (+ b 1))) ((= b n) out)
        (do ((ch 0 (+ ch 1))) ((= ch c))
          (do ((y 0 (+ y 1))) ((= y h))
            (do ((x 0 (+ x 1))) ((= x w))
              (typed-vector-set! out dtype (+ (* (+ (* (+ (* b h) y) w) x) c) ch)
                                 (typed-vector-ref data dtype
                                                   (+ (* (+ (* (+ (* b c) ch) h) y) w) x)))))))))

  ) ;; end module
