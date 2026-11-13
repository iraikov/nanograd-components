;;; am-transformer.scm
;;;
;;; LLAMA-style Transformer on lazy (array-morphisms) tensors, for SSA
;;; training.
;;;
;;; A batch of B token sequences of length T is processed as one matrix
;;; with B*T rows; row b*T + t holds position t of sequence b.  The
;;; input is the one-hot matrix X [B*T, V] of the tokens, so the
;;; embedding lookup is the product X @ E with the embedding E [V, D],
;;; and its gradient is X^T G.
;;;
;;; Every step is built from matrix products and element-wise
;;; operations, which the SSA engine compiles and the registered
;;; backends run:
;;;   - attention head h uses the columns [h*hs, (h+1)*hs) of Q, K and V.
;;;     Q, K and V [B*T, D] are regrouped into [H, B*T, hs] arrays, and
;;;     the attention of one head is written once as a function of its
;;;     [B*T, hs] cells and lifted over the H heads (var-lift): the
;;;     products of all heads are one batched matrix product, and the
;;;     softmax reductions of all heads are one reduction each.  The head
;;;     outputs are regrouped back into [B*T, D];
;;;   - the scores of all B*T rows of a head are one [B*T, B*T] product;
;;;     a constant additive mask sets -1e9 between different sequences
;;;     and, when causal, for keys after the query;
;;;   - RoPE is x*C + (x@P)*S with constant cos/sin tables C, S [B*T, D]
;;;     and a constant pair-rotation matrix P [D, D].
;;; The attention cost is O((B*T)^2 * D), so this layout suits the small
;;; batches of a benchmark; B*T of a few hundred is practical.
;;;
;;; The model computes the same function as nanograd-transformer's
;;; make-transformer, with the parameters in the same order, so
;;; am-copy-parameters! can transfer weights between the two.  The
;;; output projection is tied to the token embedding.
;;;
;;; The gradients of the lifted attention are derived by the SSA engine,
;;; so the model is trained with am-training-step/ssa; backward! on its
;;; output signals an error.

(module nanograd-am-transformer

  (make-am-attention
   make-am-swiglu-ffn
   make-am-transformer-block
   make-am-transformer
   am-transformer?
   am-tokens->one-hot
   am-next-token-targets)

  (import scheme (chicken base) (chicken format))
  (import (only srfi-1 fold iota append-map take drop last))
  (import srfi-4)
  (import (only srfi-69 make-hash-table hash-table-ref/default hash-table-set!))
  (import yasos)
  (import array-morphisms-core)
  (import (only array-morphisms-basic-ops
                morph-cast morph-exp morph-reduce morph+ morph- morph* morph/))
  (import (only array-morphisms-structural-ops morph-reshape morph-transpose))
  (import (only array-morphisms-blas-exec morph-matmul))
  (import (prefix array-morphisms-grad am:))
  (import nanograd-autograd)
  (import nanograd-layer)
  (import nanograd-array-morphisms)
  (import nanograd-am-common)

  (define-predicate am-transformer?)

  (define (rows-of x) (vector-ref (morph-shape (am:var-value x)) 0))

  ;;;; ================================================================
  ;;;; Constant matrices
  ;;;; ================================================================

  (define (attention-mask batch seq-len causal? dtype)
    "[B*T, B*T]: 0 where query row may attend to key column, else -1e9."
    (let ((n (* batch seq-len)))
      (am-const-fill (list n n) dtype
                     (lambda (i)
                       (let* ((q (quotient i n)) (k (modulo i n))
                              (same (= (quotient q seq-len) (quotient k seq-len)))
                              (ok (and same (or (not causal?)
                                                (<= (modulo k seq-len) (modulo q seq-len))))))
                         (if ok 0.0 -1e9))))))

  (define (rope-tables batch seq-len d hs base dtype)
    "(values C S) [B*T, D]: cos and sin of position(row) * f_i, where i
     is the pair index (j mod hs)/2 of column j and f_i = base^(-2i/hs)."
    (let* ((n (* batch seq-len))
           (angle (lambda (idx)
                    (let* ((r (quotient idx d)) (j (modulo idx d))
                           (i (quotient (modulo j hs) 2))
                           (freq (/ 1.0 (expt base (/ (* 2.0 i) hs)))))
                      (* (exact->inexact (modulo r seq-len)) freq)))))
      (values (am-const-fill (list n d) dtype (lambda (idx) (cos (angle idx))))
              (am-const-fill (list n d) dtype (lambda (idx) (sin (angle idx)))))))

  (define (rope-rotation d dtype)
    "P [D, D] with (x@P)[2i] = -x[2i+1] and (x@P)[2i+1] = x[2i]."
    (am-const-fill (list d d) dtype
                   (lambda (idx)
                     (let ((r (quotient idx d)) (c (modulo idx d)))
                       (cond ((and (even? c) (= r (+ c 1))) -1.0)
                             ((and (odd? c) (= r (- c 1))) 1.0)
                             (else 0.0))))))

  (define (make-constants d n-heads use-rope? rope-base causal? dtype)
    "Constants shared by all layers, built once per batch size."
    (let ((hs (quotient d n-heads))
          (cache (make-hash-table)))
      (lambda (batch seq-len)
        (let ((key (cons batch seq-len)))
          (or (hash-table-ref/default cache key #f)
              (let ((c (list (attention-mask batch seq-len causal? dtype)
                             (and use-rope?
                                  (let-values (((C S) (rope-tables batch seq-len d hs
                                                                   rope-base dtype)))
                                    (list C S (rope-rotation d dtype)))))))
                (hash-table-set! cache key c)
                c))))))

  ;;;; ================================================================
  ;;;; Layers
  ;;;; ================================================================

  (define (softmax-rows-morph s)
    "Softmax over the last axis of the morphism S, with the row maximum
     subtracted first.  The SSA gradient passes nothing through the
     maximum, which does not change the gradient because softmax is
     invariant under a shift of each row (as in am-softmax-rows)."
    (let* ((m (morph-reduce 'max s '(1) #t))
           (e (morph-exp (morph- s m))))
      (morph/ e (morph-reduce 'sum e '(1) #t))))

  (define (make-am-attention d-model n-heads #!key (dtype 'f32) (name "AM-Attention"))
    "Multi-head self-attention.  Forward takes the [B*T, D] input and the
     constants (mask rope) of make-constants.  Parameters:
     WQ b, WK b, WV b, WO b.

     Q, K and V [B*T, D] are regrouped into [H, B*T, hs] arrays, copied
     so that each head is stored contiguously.  One head computes

       softmax(qh kh^T * s + mask) vh,   s = 1/sqrt(hs),

     on its [B*T, hs] cells qh, kh, vh; this function is lifted over the
     H heads with var-lift, so all heads are computed together and each
     head has the values it would have on its own.  The [H, B*T, hs]
     result is regrouped into [B*T, D] and projected by WO.  The
     gradients of the lifted part come from am-training-step/ssa."
    (let* ((hs (quotient d-model n-heads))
           (scale (/ 1.0 (sqrt (exact->inexact hs))))
           (proj (lambda () (make-am-dense-layer d-model d-model dtype: dtype)))
           (wq (proj)) (wk (proj)) (wv (proj)) (wo (proj))
           (parts (list wq wk wv wo)))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((parameters self) (append-map parameters parts))
       ((zero-grad-layer! self) (for-each zero-grad-layer! parts))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x consts)
        (let* ((mask (am:var-value (car consts)))
               (rope (cadr consts))
               (rot (lambda (v)
                      (if rope
                          (am:var+ (am:var* v (car rope))
                                   (am:var* (am:var-matmul v (caddr rope)) (cadr rope)))
                          v)))
               (q (rot (mv (forward wq x))))
               (k (rot (mv (forward wk x))))
               (v (mv (forward wv x)))
               (n (rows-of q))
               (s (am:var-value (am-scalar scale dtype)))
               ;; [B*T, D] -> [H, B*T, hs], one contiguous copy.  The
               ;; SSA program differentiates this morphism directly, so
               ;; the variable needs no gradient links of its own.
               (heads-of
                (lambda (m)
                  (am:make-var
                   (morph-cast (morph-transpose (morph-reshape (am:var-value m)
                                                               (vector n n-heads hs))
                                                '(1 0 2))
                               dtype)
                   #f)))
               (head
                (lambda (qh kh vh)
                  (morph-matmul
                   (softmax-rows-morph
                    (morph+ (morph* (morph-matmul qh (morph-transpose kh '(1 0))) s)
                            mask))
                   vh)))
               (o3 (am:var-lift head 2 (heads-of q) (heads-of k) (heads-of v)))
               (merged (am:var-reshape (am:var-transpose o3 '(1 0 2))
                                       (vector n d-model))))
          (forward wo (lazy merged))))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-swiglu-ffn d-model d-ff #!key (dtype 'f32) (name "AM-FFN"))
    "(SiLU(W1 x) * W3 x) W2.  Parameters: W1 b, W2 b, W3 b."
    (let* ((w1 (make-am-dense-layer d-model d-ff dtype: dtype))
           (w2 (make-am-dense-layer d-ff d-model dtype: dtype))
           (w3 (make-am-dense-layer d-model d-ff dtype: dtype))
           (parts (list w1 w2 w3)))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((parameters self) (append-map parameters parts))
       ((zero-grad-layer! self) (for-each zero-grad-layer! parts))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x)
        (let ((g (mv (activation-forward-am (make-silu) (forward w1 x))))
              (u (mv (forward w3 x))))
          (forward w2 (lazy (am:var* g u)))))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-transformer-block d-model n-heads d-ff
                                     #!key (norm-eps 1e-5) (dtype 'f32)
                                     (name "AM-TransformerBlock"))
    "Pre-norm block: h = x + Attn(RMSNorm(x)); y = h + FFN(RMSNorm(h)).
     Parameters: norm1, norm2, attention, FFN."
    (let* ((norm1 (make-am-rmsnorm d-model eps: norm-eps dtype: dtype))
           (norm2 (make-am-rmsnorm d-model eps: norm-eps dtype: dtype))
           (attn (make-am-attention d-model n-heads dtype: dtype))
           (ffn (make-am-swiglu-ffn d-model d-ff dtype: dtype))
           (parts (list norm1 norm2 attn ffn)))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((parameters self) (append-map parameters parts))
       ((zero-grad-layer! self) (for-each zero-grad-layer! parts))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x consts)
        (let* ((h (am:var+ (mv x) (mv (forward attn (forward norm1 x) consts))))
               (y (am:var+ h (mv (forward ffn (forward norm2 (lazy h)))))))
          (lazy y)))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-transformer vocab-size seq-len d-model n-layers n-heads
                               #!key
                               (d-ff #f)
                               (use-rope? #t)
                               (rope-base 10000.0)
                               (causal? #t)
                               (norm-eps 1e-5)
                               (dtype 'f32)
                               (name "AM-Transformer"))
    "Transformer language model on one-hot input [B*T, VOCAB-SIZE],
     with T = SEQ-LEN.  Returns logits [B*T, VOCAB-SIZE].

     Parameters, in order: token embedding [V, D], final norm weight,
     then for each block its norms, attention and FFN parameters."
    (let* ((d-ff (or d-ff (* 4 d-model)))
           (embedding (am-param-uniform (list vocab-size d-model) 0.02 dtype))
           (final-norm (make-am-rmsnorm d-model eps: norm-eps dtype: dtype))
           (blocks (map (lambda (i)
                          (make-am-transformer-block d-model n-heads d-ff
                                                     norm-eps: norm-eps dtype: dtype
                                                     name: (sprintf "~A-Block~A" name i)))
                        (iota n-layers)))
           (constants (make-constants d-model n-heads use-rope? rope-base causal? dtype)))
      (object
       ((layer? self) #t)
       ((am-transformer? self) #t)
       ((layer-name self) name)
       ((layer-input-size self) vocab-size)
       ((layer-output-size self) vocab-size)
       ((parameters self)
        (append (list embedding) (parameters final-norm) (append-map parameters blocks)))
       ((zero-grad-layer! self)
        (am:zero-grad! (mv embedding))
        (zero-grad-layer! final-norm)
        (for-each zero-grad-layer! blocks))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x-lt)
        (let* ((x (mv x-lt))
               (batch (quotient (rows-of x) seq-len))
               (consts (constants batch seq-len))
               (e (mv embedding))
               (h (fold (lambda (blk h) (forward blk h consts))
                        (lazy (am:var-matmul x e))
                        blocks))
               (hn (mv (forward final-norm h))))
          (lazy (am:var-matmul hn (am:var-transpose e '(1 0))))))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  ;;;; ================================================================
  ;;;; Inputs and targets
  ;;;; ================================================================

  (define (am-tokens->one-hot sequences vocab-size dtype)
    "One-hot typed vector [B*T, VOCAB-SIZE] for a list of B token lists
     of equal length T, row b*T + t for token t of sequence b."
    (let* ((rows (apply append sequences))
           (v (allocate-typed-vector dtype (* (length rows) vocab-size))))
      (do ((i 0 (+ i 1))) ((= i (* (length rows) vocab-size)))
        (typed-vector-set! v dtype i 0.0))
      (let loop ((rs rows) (r 0))
        (unless (null? rs)
          (typed-vector-set! v dtype (+ (* r vocab-size) (car rs)) 1.0)
          (loop (cdr rs) (+ r 1))))
      v))

  (define (am-next-token-targets sequences vocab-size dtype)
    "For sequences of length T+1, the (inputs targets) pair of one-hot
     typed vectors [B*T, VOCAB-SIZE]: inputs are the first T tokens and
     targets the last T."
    (let ((inputs (map (lambda (s) (take s (- (length s) 1))) sequences))
          (targets (map cdr sequences)))
      (list (am-tokens->one-hot inputs vocab-size dtype)
            (am-tokens->one-hot targets vocab-size dtype))))

  ) ;; end module
