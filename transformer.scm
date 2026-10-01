;; Transformer with RoPE and KV caching for Nanograd.
;;
;; The architecture follows LLAMA2: token embeddings, a stack of
;; pre-norm blocks (RMSNorm -> multi-head self-attention -> residual,
;; RMSNorm -> SwiGLU feed-forward -> residual), a final RMSNorm, and an
;; output projection tied to the token embedding matrix.
;;
;; Every step of the forward pass is differentiable: the embedding
;; lookup, the rotary position embedding, the attention heads and the
;; output logits all propagate gradients to the parameters.  Sequences
;; are processed as (seq_len, d_model) matrices; attention heads are
;; column blocks of width head_size.
;;
;; The KV cache supports incremental generation.  Each forward call
;; with a cache appends the keys and values of its tokens after the
;; ones already cached and attends over the whole cached prefix.

(module nanograd-transformer
  (
   ;; RoPE operations
   make-rope-embedding
   rope-embedding?
   apply-rope-rotation
   rope-rotate-rows
   get-frequencies
   rope-head-size

   ;; KV Cache
   make-kv-cache
   kv-cache?
   cache-kv-pair!
   get-cached-kv
   kv-cache-length
   clear-cache!

   ;; Attention
   make-multihead-attention
   multihead-attention?
   scaled-dot-product-attention
   multihead-attention-core
   causal-mask

   ;; Differentiable helpers
   embedding-lookup
   slice-columns
   concat-columns

   ;; Feed-Forward Network
   make-ffn-block
   ffn-block?

   ;; Transformer Block
   make-transformer-block
   transformer-block?

   ;; Full Transformer
   make-transformer
   transformer?
   )

  (import
   scheme
   (chicken base)
   (chicken format)
   (chicken random)
   (srfi 1)
   (srfi 4)
   yasos
   blas
   nanograd-autograd
   nanograd-layer
   )

  ;; Hygienic macro for dtype-based operation dispatch
  (include "with-dtype.scm")

  ;;; ==================================================================
  ;;; Small tensor helpers
  ;;; ==================================================================

  (define (make-constant-tensor data shape dtype)
    "A tensor that does not require gradients."
    (make-base-tensor data shape dtype #f))

  (define (make-ones-parameter n dtype)
    (case dtype
      ((f32) (make-tensor32 (make-f32vector n 1.0) (list n)))
      ((f64) (make-tensor64 (make-f64vector n 1.0) (list n)))))

  (define (token->index t)
    "Token ids may arrive as flonums (from tensor->list)."
    (if (exact? t) t (inexact->exact (round t))))

  (define (slice-columns x start width)
    "Columns [START, START+WIDTH) of the matrix X (rows, cols).
     The gradient is scattered back into the same columns of X."
    (let* ((dtype (tensor-dtype x))
           (shape (tensor-shape x))
           (rows (car shape))
           (cols (cadr shape))
           (src (tensor-data x))
           (out (with-dtype dtype (vec (* rows width) 0.0)))
           (requires-grad? (tensor-requires-grad? x)))
      (unless (<= (+ start width) cols)
        (error 'slice-columns "column range out of bounds" start width cols))
      (do ((r 0 (+ r 1))) ((= r rows))
        (with-dtype dtype
          (copy-to out src Xoffset: (+ (* r cols) start)
                   Yoffset: (* r width) size: width)))
      (let ((result (make-base-tensor out (list rows width) dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn!
           result
           (lambda ()
             (let ((g (tensor-grad result))
                   (gx (with-dtype dtype (vec (* rows cols) 0.0))))
               (do ((r 0 (+ r 1))) ((= r rows))
                 (with-dtype dtype
                   (copy-to gx g Xoffset: (* r width)
                            Yoffset: (+ (* r cols) start) size: width)))
               (add-to-grad! x gx)))
           (list x)))
        result)))

  (define (concat-columns xs)
    "Concatenate matrices with equal row counts along the column axis.
     The gradient of each input is its column block of the output gradient."
    (let* ((x0 (car xs))
           (dtype (tensor-dtype x0))
           (rows (car (tensor-shape x0)))
           (widths (map (lambda (x) (cadr (tensor-shape x))) xs))
           (cols (apply + widths))
           (out (with-dtype dtype (vec (* rows cols) 0.0)))
           (requires-grad? (any tensor-requires-grad? xs)))
      (let loop ((xs xs) (widths widths) (start 0))
        (unless (null? xs)
          (let ((src (tensor-data (car xs))) (w (car widths)))
            (do ((r 0 (+ r 1))) ((= r rows))
              (with-dtype dtype
                (copy-to out src Xoffset: (* r w)
                         Yoffset: (+ (* r cols) start) size: w)))
            (loop (cdr xs) (cdr widths) (+ start w)))))
      (let ((result (make-base-tensor out (list rows cols) dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn!
           result
           (lambda ()
             (let ((g (tensor-grad result)))
               (let loop ((xs xs) (widths widths) (start 0))
                 (unless (null? xs)
                   (let ((x (car xs)) (w (car widths)))
                     (when (tensor-requires-grad? x)
                       (let ((gx (with-dtype dtype (vec (* rows w) 0.0))))
                         (do ((r 0 (+ r 1))) ((= r rows))
                           (with-dtype dtype
                             (copy-to gx g Xoffset: (+ (* r cols) start)
                                      Yoffset: (* r w) size: w)))
                         (add-to-grad! x gx)))
                     (loop (cdr xs) (cdr widths) (+ start w)))))))
           xs))
        result)))

  (define (embedding-lookup table tokens)
    "Rows of TABLE (vocab, d) selected by the token ids in TOKENS.
     Returns a (length TOKENS, d) tensor.  The backward pass adds each
     output row gradient into the table row it was copied from."
    (let* ((dtype (tensor-dtype table))
           (d (cadr (tensor-shape table)))
           (vocab (car (tensor-shape table)))
           (ids (map token->index tokens))
           (n (length ids))
           (src (tensor-data table))
           (out (with-dtype dtype (vec (* n d) 0.0)))
           (requires-grad? (tensor-requires-grad? table)))
      (let loop ((ids ids) (i 0))
        (unless (null? ids)
          (let ((t (car ids)))
            (unless (and (>= t 0) (< t vocab))
              (error 'embedding-lookup "token id out of range" t vocab))
            (with-dtype dtype
              (copy-to out src Xoffset: (* t d) Yoffset: (* i d) size: d))
            (loop (cdr ids) (+ i 1)))))
      (let ((result (make-base-tensor out (list n d) dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn!
           result
           (lambda ()
             (let ((g (tensor-grad result))
                   (gt (with-dtype dtype (vec (* vocab d) 0.0))))
               (let loop ((ids ids) (i 0))
                 (unless (null? ids)
                   (let ((t (car ids)))
                     (do ((j 0 (+ j 1))) ((= j d))
                       (with-dtype dtype
                         (elt-set! gt (+ (* t d) j)
                                   (+ (elt-ref gt (+ (* t d) j))
                                      (elt-ref g (+ (* i d) j)))))))
                   (loop (cdr ids) (+ i 1))))
               (add-to-grad! table gt)))
           (list table)))
        result)))

  ;;; ==================================================================
  ;;; RoPE (Rotary Position Embedding)
  ;;; ==================================================================

  (define-predicate rope-embedding?)
  (define-operation (get-frequencies rope pos))
  (define-operation (rope-head-size rope))
  (define-operation (rope-max-seq-len rope))

  (define (make-rope-embedding max-seq-len head-size #!key (base 10000.0) (dtype 'f32))
    "Create a RoPE lookup table.

     Precomputes cos(p * f_i) and sin(p * f_i) for every position p below
     MAX-SEQ-LEN and every pair index i below head_size/2, where
     f_i = BASE^(-2i / head_size)."

    (let* ((half-dim (quotient head-size 2))
           (freqs-real (with-dtype dtype (vec (* max-seq-len half-dim) 0.0)))
           (freqs-imag (with-dtype dtype (vec (* max-seq-len half-dim) 0.0))))

      (unless (even? head-size)
        (error 'make-rope-embedding "head size must be even" head-size))

      (with-dtype dtype
                  (do ((pos 0 (+ pos 1)))
                      ((= pos max-seq-len))
                    (do ((i 0 (+ i 1)))
                        ((= i half-dim))
                      (let* ((freq (/ 1.0 (expt base (/ (* 2.0 i) head-size))))
                             (angle (* pos freq))
                             (idx (+ (* pos half-dim) i)))
                        (elt-set! freqs-real idx (cos angle))
                        (elt-set! freqs-imag idx (sin angle))))))

      (object
       ((rope-embedding? self) #t)
       ((layer? self) #t)
       ((rope-head-size self) head-size)
       ((rope-max-seq-len self) max-seq-len)

       ;; Cos/sin vectors (length head_size/2) for one position
       ((get-frequencies self pos)
        (with-dtype
         dtype
         (let ((start (* pos half-dim)))
           (values
            (subvec freqs-real start half-dim)
            (subvec freqs-imag start half-dim)))))

       ((parameters self) '())
       ((zero-grad-layer! self) (void)))))

  (define (rope-rotate-rows x rope start-pos)
    "Rotate the rows of X (seq_len, d) with RoPE.

     Row t is taken to sit at position START-POS + t.  Each row is split
     into heads of width (rope-head-size ROPE); inside a head the pair
     (x_2i, x_2i+1) is rotated by the angle for pair index i.  The
     backward pass applies the inverse rotation to the gradient."
    (let* ((dtype (tensor-dtype x))
           (shape (tensor-shape x))
           (rows (car shape))
           (d (cadr shape))
           (hs (rope-head-size rope))
           (src (tensor-data x))
           (out (with-dtype dtype (vec (* rows d) 0.0)))
           (requires-grad? (tensor-requires-grad? x)))
      ;; ROTATE! writes R(sign * angle) v into DST for every pair.
      (define (rotate! dst v sign)
        (do ((t 0 (+ t 1))) ((= t rows))
          (let-values (((cs sn) (get-frequencies rope (+ start-pos t))))
            (let ((off (* t d)))
              (do ((j 0 (+ j 2))) ((>= j d))
                (let* ((i (quotient (modulo j hs) 2))
                       (c (with-dtype dtype (elt-ref cs i)))
                       (s (* sign (with-dtype dtype (elt-ref sn i))))
                       (a (with-dtype dtype (elt-ref v (+ off j))))
                       (b (with-dtype dtype (elt-ref v (+ off j 1)))))
                  (with-dtype dtype
                    (elt-set! dst (+ off j) (- (* a c) (* b s)))
                    (elt-set! dst (+ off j 1) (+ (* a s) (* b c))))))))))

      (unless (zero? (modulo d hs))
        (error 'rope-rotate-rows "row width is not a multiple of the head size" d hs))
      (unless (<= (+ start-pos rows) (rope-max-seq-len rope))
        (error 'rope-rotate-rows "position exceeds RoPE table" (+ start-pos rows)))

      (rotate! out src 1.0)
      (let ((result (make-base-tensor out shape dtype requires-grad?)))
        (when requires-grad?
          (set-backward-fn!
           result
           (lambda ()
             (let ((gx (with-dtype dtype (vec (* rows d) 0.0))))
               (rotate! gx (tensor-grad result) -1.0)
               (add-to-grad! x gx)))
           (list x)))
        result)))

  (define (apply-rope-rotation Q K rope pos)
    "Apply RoPE at position POS to one query and one key.

     Q and K have shape (n_heads, head_size).
     Returns (values Q-rotated K-rotated), both differentiable."
    (let* ((shape (tensor-shape Q))
           (flat (list 1 (apply * shape)))
           (rot (lambda (t)
                  (reshape (rope-rotate-rows (reshape t flat) rope pos) shape))))
      (values (rot Q) (rot K))))

  ;;; ==================================================================
  ;;; KV Cache for Generation
  ;;; ==================================================================

  (define-predicate kv-cache?)
  (define-operation (cache-kv-pair! cache layer-idx k v pos))
  (define-operation (get-cached-kv cache layer-idx pos))
  (define-operation (kv-cache-length cache))
  (define-operation (clear-cache! cache))

  (define (make-kv-cache n-layers n-heads max-seq-len head-size #!key (dtype 'f32))
    "Create a key/value cache for incremental generation.

     For each layer the cache holds up to MAX-SEQ-LEN rows of keys and
     values, each row of width n_heads * head_size.  The cache is used
     for inference only and records no gradients."

    (let* ((row (* n-heads head-size))
           (layer-size (* max-seq-len row))
           (key-cache (with-dtype dtype (vec (* n-layers layer-size) 0.0)))
           (value-cache (with-dtype dtype (vec (* n-layers layer-size) 0.0)))
           (cached-length 0))

      (object
       ((kv-cache? self) #t)

       ;; Number of positions stored so far
       ((kv-cache-length self) cached-length)

       ;; Store the rows of K and V (shape (T, d) or (T, n_heads,
       ;; head_size)) at positions POS .. POS+T-1 of layer LAYER-IDX.
       ((cache-kv-pair! self layer-idx k v pos)
        (let* ((n (apply * (tensor-shape k)))
               (t (quotient n row))
               (offset (+ (* layer-idx layer-size) (* pos row))))
          (unless (<= (+ pos t) max-seq-len)
            (error 'cache-kv-pair! "KV cache is full" (+ pos t) max-seq-len))
          (with-dtype dtype
            (copy-to key-cache (tensor-data k) Yoffset: offset size: n)
            (copy-to value-cache (tensor-data v) Yoffset: offset size: n))
          (set! cached-length (max cached-length (+ pos t)))))

       ;; Cached keys and values for positions 0 .. POS of a layer, as
       ;; (values K V) with shape (POS+1, n_heads, head_size).
       ((get-cached-kv self layer-idx pos)
        (let* ((layer-offset (* layer-idx layer-size))
               (seq-len (+ pos 1))
               (kv-size (* seq-len row)))
          (values
           (make-constant-tensor
            (with-dtype dtype (subvec key-cache layer-offset kv-size))
            (list seq-len n-heads head-size) dtype)
           (make-constant-tensor
            (with-dtype dtype (subvec value-cache layer-offset kv-size))
            (list seq-len n-heads head-size) dtype))))

       ;; Clear cache for a new sequence
       ((clear-cache! self)
        (set! cached-length 0)
        (with-dtype dtype
          (do ((i 0 (+ i 1)))
              ((= i (vector-length-for-dtype key-cache dtype)))
            (elt-set! key-cache i 0.0)
            (elt-set! value-cache i 0.0)))))))

  ;;; ==================================================================
  ;;; Scaled dot-product attention
  ;;; ==================================================================

  (define (causal-mask q-len kv-len dtype)
    "Additive mask (Q-LEN, KV-LEN).  Query row i sits at absolute
     position (KV-LEN - Q-LEN) + i and may attend to keys at positions
     up to and including its own; later keys get -1e9."
    (let ((m (with-dtype dtype (vec (* q-len kv-len) 0.0)))
          (offset (- kv-len q-len)))
      (do ((i 0 (+ i 1))) ((= i q-len))
        (do ((j (+ i offset 1) (+ j 1))) ((>= j kv-len))
          (with-dtype dtype (elt-set! m (+ (* i kv-len) j) -1e9))))
      (make-constant-tensor m (list q-len kv-len) dtype)))

  (define (multihead-attention-core Q K V n-heads #!key (causal? #f))
    "Multi-head attention on matrices.

     Q is (q_len, d); K and V are (kv_len, d); head h uses columns
     [h*hs, (h+1)*hs) with hs = d / N-HEADS.  For each head
       O_h = softmax(Q_h K_h^T / sqrt(hs) + M) V_h
     where M is the causal mask when CAUSAL? is true.  Returns the
     heads concatenated back into a (q_len, d) matrix."
    (let* ((dtype (tensor-dtype Q))
           (q-len (car (tensor-shape Q)))
           (kv-len (car (tensor-shape K)))
           (d (cadr (tensor-shape Q)))
           (hs (quotient d n-heads))
           (scale (/ 1.0 (sqrt (exact->inexact hs))))
           (mask (and causal? (causal-mask q-len kv-len dtype))))
      (concat-columns
       (map (lambda (h)
              (let* ((qh (slice-columns Q (* h hs) hs))
                     (kh (slice-columns K (* h hs) hs))
                     (vh (slice-columns V (* h hs) hs))
                     (scores (scale-op (matmul-op qh (transpose-tensor kh '(1 0)))
                                       scale))
                     (masked (if mask (add scores mask) scores))
                     (probs (softmax masked axis: -1)))
                (matmul-op probs vh)))
            (iota n-heads)))))

  (define (scaled-dot-product-attention Q K V head-size seq-len #!key (causal? #f))
    "Scaled dot-product attention on head-shaped tensors.

     Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V, per head.

     Args:
       Q: (query_len, n_heads, head_size)
       K: (kv_len, n_heads, head_size)
       V: (kv_len, n_heads, head_size)
       head-size: Dimension of each head
       seq-len: Query length (must equal query_len)
       causal?: Mask keys that come after each query

     Returns:
       (query_len, n_heads, head_size), differentiable in Q, K and V."
    (let* ((q-shape (tensor-shape Q))
           (q-len (car q-shape))
           (n-heads (cadr q-shape))
           (d (* n-heads head-size))
           (kv-len (car (tensor-shape K))))
      (unless (= seq-len q-len)
        (error 'scaled-dot-product-attention "seq-len differs from query length"
               seq-len q-len))
      (reshape (multihead-attention-core (reshape Q (list q-len d))
                                         (reshape K (list kv-len d))
                                         (reshape V (list kv-len d))
                                         n-heads causal?: causal?)
               q-shape)))

  ;;; ==================================================================
  ;;; Multi-Head Self-Attention with RoPE
  ;;; ==================================================================

  (define-predicate multihead-attention?)

  (define (make-multihead-attention d-model n-heads
                                   #!key
                                   (rope #f)
                                   (causal? #f)
                                   (dtype 'f32)
                                   (name "MultiHeadAttention"))
    "Multi-head self-attention with optional RoPE and causal masking.

     Args:
       d-model: Model dimension
       n-heads: Number of attention heads
       rope: Optional RoPE embedding object
       causal?: Restrict each position to itself and earlier positions
       dtype: Data type

     Forward args:
       input: (seq_len, d_model) tensor
       position: Absolute position of the first input row
       kv-cache: Optional KV cache object
       layer-idx: Layer index (for caching)"

  (unless (zero? (modulo d-model n-heads))
    (error 'make-multihead-attention
           (format #f "d_model (~A) must be divisible by n_heads (~A)"
                   d-model n-heads)))

  (let* ((head-size (quotient d-model n-heads))
         (proj (lambda (suffix)
                 (make-dense-layer d-model d-model
                                   activation: (make-identity)
                                   dtype: dtype
                                   name: (sprintf "~A-~A" name suffix))))
         (wq (proj "WQ"))
         (wk (proj "WK"))
         (wv (proj "WV"))
         (wo (proj "WO")))

    (object
     ((layer? self) #t)
     ((multihead-attention? self) #t)
     ((layer-name self) name)

     ((forward self input #!key (position 0) (kv-cache #f) (layer-idx 0))
      (let* ((q (forward wq input))
             (k (forward wk input))
             (v (forward wv input))
             (q-rot (if rope (rope-rotate-rows q rope position) q))
             (k-rot (if rope (rope-rotate-rows k rope position) k)))
        (let-values (((k-all v-all)
                      (if kv-cache
                          (let ((t (car (tensor-shape input))))
                            (cache-kv-pair! kv-cache layer-idx k-rot v position)
                            (let-values (((kc vc) (get-cached-kv kv-cache layer-idx
                                                                 (+ position t -1))))
                              (values (reshape kc (list (+ position t) d-model))
                                      (reshape vc (list (+ position t) d-model)))))
                          (values k-rot v))))
          (forward wo (multihead-attention-core q-rot k-all v-all n-heads
                                                causal?: causal?)))))

     ((parameters self)
      (append (parameters wq)
              (parameters wk)
              (parameters wv)
              (parameters wo)))

     ((zero-grad-layer! self)
      (zero-grad-layer! wq)
      (zero-grad-layer! wk)
      (zero-grad-layer! wv)
      (zero-grad-layer! wo))

     ((set-training-mode! self train?) (void))
     ((set-eval-mode! self) (void)))))

  ;;; ==================================================================
  ;;; Feed-Forward Network (SwiGLU)
  ;;; ==================================================================

  (define-predicate ffn-block?)

  (define (make-ffn-block d-model d-ff #!key (dtype 'f32) (name "FFN"))
    "Feed-forward network with SwiGLU activation.

     FFN(x) = (SiLU(W1(x)) . W3(x)) W2

     where . is element-wise multiplication."

    (let ((w1 (make-dense-layer d-model d-ff
                                activation: (make-identity)
                                dtype: dtype
                                name: (sprintf "~A-W1" name)))
          (w2 (make-dense-layer d-ff d-model
                                activation: (make-identity)
                                dtype: dtype
                                name: (sprintf "~A-W2" name)))
          (w3 (make-dense-layer d-model d-ff
                                activation: (make-identity)
                                dtype: dtype
                                name: (sprintf "~A-W3" name))))

      (object
       ((layer? self) #t)
       ((ffn-block? self) #t)
       ((layer-name self) name)

       ((forward self x)
        (forward w2 (mul (silu (forward w1 x)) (forward w3 x))))

       ((parameters self)
        (append (parameters w1)
                (parameters w2)
                (parameters w3)))

       ((zero-grad-layer! self)
        (zero-grad-layer! w1)
        (zero-grad-layer! w2)
        (zero-grad-layer! w3))

       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void)))))

  ;;; ==================================================================
  ;;; Transformer Block
  ;;; ==================================================================

  (define-predicate transformer-block?)

  (define (make-transformer-block d-model n-heads d-ff
                                 #!key
                                 (rope #f)
                                 (causal? #f)
                                 (dtype 'f32)
                                 (name "TransformerBlock"))
    "Single transformer block with attention + FFN.

     Uses pre-normalization (RMSNorm before each sublayer)."

    (let* ((norm1-weight (make-ones-parameter d-model dtype))
           (norm2-weight (make-ones-parameter d-model dtype))
           (attention (make-multihead-attention d-model n-heads
                                                rope: rope
                                                causal?: causal?
                                                dtype: dtype
                                                name: (sprintf "~A-Attn" name)))
           (ffn (make-ffn-block d-model d-ff
                                dtype: dtype
                                name: (sprintf "~A-FFN" name))))

      (object
       ((layer? self) #t)
       ((transformer-block? self) #t)
       ((layer-name self) name)

       ((forward self x #!key (position 0) (kv-cache #f) (layer-idx 0))
        (let* ((attn-out (forward attention (rmsnorm x norm1-weight)
                                  position: position
                                  kv-cache: kv-cache
                                  layer-idx: layer-idx))
               (h (add x attn-out)))
          (add h (forward ffn (rmsnorm h norm2-weight)))))

       ((parameters self)
        (append (list norm1-weight norm2-weight)
                (parameters attention)
                (parameters ffn)))

       ((zero-grad-layer! self)
        (zero-grad! norm1-weight)
        (zero-grad! norm2-weight)
        (zero-grad-layer! attention)
        (zero-grad-layer! ffn))

       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void)))))

  ;;; ==================================================================
  ;;; Complete Transformer
  ;;; ==================================================================

  (define-predicate transformer?)

  (define (make-transformer vocab-size max-seq-len d-model n-layers n-heads
                            #!key
                            (d-ff #f)
                            (use-rope? #t)
                            (rope-base 10000.0)
                            (causal? #t)
                            (dtype 'f32)
                            (name "Transformer"))
    "Complete Transformer model with token embeddings and stacked blocks.

     Args:
       vocab-size: Size of vocabulary
       max-seq-len: Maximum sequence length
       d-model: Model dimension
       n-layers: Number of transformer blocks
       n-heads: Number of attention heads
       d-ff: Feed-forward hidden dimension (default: 4 * d_model)
       use-rope?: Whether to use RoPE (default: #t)
       rope-base: Base for RoPE frequencies
       causal?: Causal (autoregressive) attention (default: #t)
       dtype: Data type

     Parameters, in order: token embedding (vocab, d_model), final norm
     weight, then for each block its norm weights, attention and FFN
     parameters.  The output projection reuses the token embedding."

    (let* ((d-ff-actual (or d-ff (* 4 d-model)))
           (head-size (quotient d-model n-heads))

           (token-embedding
            (let* ((emb-size (* vocab-size d-model))
                   (emb-data (with-dtype dtype
                               (let ((v (vec emb-size 0.0)))
                                 (do ((i 0 (+ i 1)))
                                     ((= i emb-size) v)
                                   (elt-set! v i
                                            (* 0.02 (- (pseudo-random-real) 0.5))))))))
              (case dtype
                ((f32) (make-tensor32 emb-data (list vocab-size d-model)))
                ((f64) (make-tensor64 emb-data (list vocab-size d-model))))))

           (rope (and use-rope?
                      (make-rope-embedding max-seq-len head-size
                                           base: rope-base
                                           dtype: dtype)))

           (blocks (map (lambda (i)
                          (make-transformer-block d-model n-heads d-ff-actual
                                                  rope: rope
                                                  causal?: causal?
                                                  dtype: dtype
                                                  name: (sprintf "~A-Block~A"
                                                                 name i)))
                        (iota n-layers)))

           (final-norm-weight (make-ones-parameter d-model dtype)))

      (object
       ((layer? self) #t)
       ((transformer? self) #t)
       ((layer-name self) name)

       ((forward self tokens #!key (kv-cache #f))
        "Forward pass through the transformer.

         Args:
           tokens: List of token ids, or a tensor of ids
           kv-cache: Optional KV cache for generation.  The tokens are
             placed after the positions already in the cache.

         Returns:
           Without a cache, logits of shape (seq_len, vocab_size).
           With a cache, logits of the last token, shape (vocab_size)."
        (let* ((token-list (if (tensor? tokens) (tensor->list tokens) tokens))
               (seq-len (length token-list))
               (start (if kv-cache (kv-cache-length kv-cache) 0))
               (hidden (let loop ((bs blocks) (i 0)
                                  (x (embedding-lookup token-embedding token-list)))
                         (if (null? bs)
                             x
                             (loop (cdr bs) (+ i 1)
                                   (forward (car bs) x
                                            position: start
                                            kv-cache: kv-cache
                                            layer-idx: i)))))
               (normed (rmsnorm hidden final-norm-weight)))
          (if kv-cache
              (matmul-op token-embedding
                         (reshape (slice-tensor normed (- seq-len 1) 1)
                                  (list d-model)))
              (matmul-op normed (transpose-tensor token-embedding '(1 0))))))

       ((parameters self)
        (append (list token-embedding final-norm-weight)
                (append-map parameters blocks)))

       ((zero-grad-layer! self)
        (zero-grad! token-embedding)
        (zero-grad! final-norm-weight)
        (for-each zero-grad-layer! blocks))

       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void)))))

) ;; end module
