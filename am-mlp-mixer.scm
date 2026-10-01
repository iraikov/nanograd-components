;;; am-mlp-mixer.scm
;;;
;;; MLP-Mixer on lazy (array-morphisms) tensors, for SSA training.
;;;
;;; The model computes the same function as nanograd-mlp-mixer, applied
;;; to a whole batch at once.  A batch of B samples, each a (T, D) matrix
;;; of T tokens with D features, is stored token-major as one [T*B, D]
;;; matrix: row t*B + b holds token t of sample b.  With this layout
;;;   - channel mixing is a pair of dense layers on the [T*B, D] rows;
;;;   - token mixing is a left matrix product on the [T, B*D] view of
;;;     the same data, W1 [Ht, T] @ X [T, B*D], so no transposes of the
;;;     activations are needed.
;;;
;;; Parameters are listed in the same order and with the same element
;;; counts as the strict model, so am-copy-parameters! can transfer
;;; weights between the two.

(module nanograd-am-mlp-mixer

  (make-am-token-mixing-mlp
   make-am-channel-mixing-mlp
   make-am-mlp-mixer-block
   make-am-mlp-mixer
   am-mlp-mixer?
   am-mixer-pack-batch
   am-mixer-unpack-batch)

  (import scheme (chicken base) (chicken format))
  (import (only srfi-1 fold iota append-map))
  (import srfi-4)
  (import yasos)
  (import array-morphisms-core)
  (import (prefix array-morphisms-grad am:))
  (import nanograd-autograd)
  (import nanograd-layer)
  (import nanograd-array-morphisms)
  (import nanograd-am-common)

  (define-predicate am-mlp-mixer?)

  (define (rows-of x) (vector-ref (morph-shape (am:var-value x)) 0))

  (define (am-gelu x)
    "GELU (tanh approximation) of morph-variable X."
    (mv (activation-forward-am (make-gelu) (lazy x))))

  (define (make-am-token-mixing-mlp seq-len #!key (expansion-factor 4) (dtype 'f32)
                                    (name "AM-TokenMixing"))
    "Token mixing for token-major [T*B, D] input, T = SEQ-LEN.

     For every sample and feature, the column of T token values x is
     mapped to W2 gelu(W1 x + b1) + b2, with W1 [Ht, T], W2 [T, Ht] and
     Ht = EXPANSION-FACTOR * T.  Parameters: W1 b1 W2 b2."
    (let* ((ht (* expansion-factor seq-len))
           (w1 (am-param-normal (list ht seq-len) (sqrt (/ 2.0 seq-len)) dtype))
           (b1 (am-param-fill (list ht) 0.0 dtype))
           (w2 (am-param-normal (list seq-len ht) (sqrt (/ 2.0 ht)) dtype))
           (b2 (am-param-fill (list seq-len) 0.0 dtype))
           (params (list w1 b1 w2 b2)))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((parameters self) params)
       ((zero-grad-layer! self) (for-each (lambda (p) (am:zero-grad! (mv p))) params))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x-lt)
        (let* ((x (mv x-lt))
               (shape (vector->list (morph-shape (am:var-value x))))
               (cols (quotient (apply * shape) seq-len))
               (xt (am:var-reshape x (list seq-len cols)))       ; [T, B*D]
               (h (am-gelu (am:var+ (am:var-matmul (mv w1) xt)
                                 (am:var-reshape (mv b1) (list ht 1)))))
               (o (am:var+ (am:var-matmul (mv w2) h)
                           (am:var-reshape (mv b2) (list seq-len 1)))))
          (lazy (am:var-reshape o shape))))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-channel-mixing-mlp hidden-dim #!key (expansion-factor 4) (dtype 'f32)
                                      (name "AM-ChannelMixing"))
    "Channel mixing: dense(D -> 4D) -> GELU -> dense(4D -> D) on each row."
    (let* ((hd (* expansion-factor hidden-dim))
           (d1 (make-am-dense-layer hidden-dim hd activation: (make-gelu) dtype: dtype))
           (d2 (make-am-dense-layer hd hidden-dim dtype: dtype)))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((parameters self) (append (parameters d1) (parameters d2)))
       ((zero-grad-layer! self) (zero-grad-layer! d1) (zero-grad-layer! d2))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x) (forward d2 (forward d1 x)))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-mlp-mixer-block seq-len hidden-dim
                                   #!key
                                   (token-expansion 4)
                                   (channel-expansion 4)
                                   (use-rmsnorm #t)
                                   (norm-eps 1e-6)
                                   (dtype 'f32)
                                   (name "AM-MixerBlock"))
    "Pre-norm mixer block:
       h = x + TokenMix(Norm1(x));  y = h + ChannelMix(Norm2(h)).
     Parameters: token mixing, channel mixing, norm1, norm2."
    (let* ((norm (lambda (suffix)
                   (if use-rmsnorm
                       (make-am-rmsnorm hidden-dim eps: norm-eps dtype: dtype
                                        name: (sprintf "~A-~A" name suffix))
                       (make-am-layernorm hidden-dim eps: norm-eps dtype: dtype
                                          name: (sprintf "~A-~A" name suffix)))))
           (token (make-am-token-mixing-mlp seq-len expansion-factor: token-expansion
                                            dtype: dtype))
           (channel (make-am-channel-mixing-mlp hidden-dim
                                                expansion-factor: channel-expansion
                                                dtype: dtype))
           (norm1 (norm "Norm1"))
           (norm2 (norm "Norm2"))
           (parts (list token channel norm1 norm2)))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((parameters self) (append-map parameters parts))
       ((zero-grad-layer! self) (for-each zero-grad-layer! parts))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x)
        (let* ((h (am:var+ (mv x) (mv (forward token (forward norm1 x)))))
               (y (am:var+ h (mv (forward channel (forward norm2 (lazy h)))))))
          (lazy y)))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-mlp-mixer seq-len hidden-dim n-blocks
                             #!key
                             (token-expansion 4)
                             (channel-expansion 4)
                             (use-rmsnorm #t)
                             (norm-eps 1e-6)
                             (num-classes #f)
                             (dtype 'f32)
                             (name "AM-MLPMixer"))
    "A stack of N-BLOCKS mixer blocks on token-major [T*B, D] input.

     Without NUM-CLASSES the output is the [T*B, D] feature matrix, as in
     the strict model.  With NUM-CLASSES the features are averaged over
     tokens and a dense head gives [B, NUM-CLASSES] logits; the head's
     parameters come last."
    (let* ((blocks (map (lambda (i)
                          (make-am-mlp-mixer-block seq-len hidden-dim
                                                   token-expansion: token-expansion
                                                   channel-expansion: channel-expansion
                                                   use-rmsnorm: use-rmsnorm
                                                   norm-eps: norm-eps
                                                   dtype: dtype
                                                   name: (sprintf "~A-Block~A" name i)))
                        (iota n-blocks)))
           (head (and num-classes
                      (make-am-dense-layer hidden-dim num-classes dtype: dtype)))
           (parts (if head (append blocks (list head)) blocks)))
      (object
       ((layer? self) #t)
       ((am-mlp-mixer? self) #t)
       ((layer-name self) name)
       ((layer-input-size self) hidden-dim)
       ((layer-output-size self) (or num-classes hidden-dim))
       ((parameters self) (append-map parameters parts))
       ((zero-grad-layer! self) (for-each zero-grad-layer! parts))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x)
        (let ((features (fold (lambda (b h) (forward b h)) x blocks)))
          (if head
              (let* ((f (mv features))
                     (batch (quotient (rows-of f) seq-len))
                     (pooled (am:var-reshape
                              (am:var-mean (am:var-reshape f (list seq-len (* batch hidden-dim)))
                                           '(0))
                              (list batch hidden-dim))))
                (forward head (lazy pooled)))
              features)))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  ;;;; ================================================================
  ;;;; Batch layout helpers
  ;;;; ================================================================

  (define (am-mixer-pack-batch samples seq-len hidden-dim dtype)
    "Pack a list of per-sample row-major (T, D) vectors into one
     token-major [T*B, D] typed vector."
    (let* ((batch (length samples))
           (out (allocate-typed-vector dtype (* seq-len batch hidden-dim))))
      (let loop ((ss samples) (b 0))
        (unless (null? ss)
          (do ((t 0 (+ t 1))) ((= t seq-len))
            (do ((d 0 (+ d 1))) ((= d hidden-dim))
              (typed-vector-set! out dtype
                                 (+ (* (+ (* t batch) b) hidden-dim) d)
                                 (typed-vector-ref (car ss) dtype
                                                   (+ (* t hidden-dim) d)))))
          (loop (cdr ss) (+ b 1))))
      out))

  (define (am-mixer-unpack-batch data batch seq-len hidden-dim dtype)
    "Inverse of am-mixer-pack-batch: a list of BATCH (T, D) vectors."
    (map (lambda (b)
           (let ((v (allocate-typed-vector dtype (* seq-len hidden-dim))))
             (do ((t 0 (+ t 1))) ((= t seq-len) v)
               (do ((d 0 (+ d 1))) ((= d hidden-dim))
                 (typed-vector-set! v dtype (+ (* t hidden-dim) d)
                                    (typed-vector-ref data dtype
                                                      (+ (* (+ (* t batch) b) hidden-dim) d)))))))
         (iota batch)))

  ) ;; end module
