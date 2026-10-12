;; Examples of training and generating with the Transformer on lazy
;; (array-morphisms) tensors.
;;
;; This is the array-morphisms counterpart of transformer_example.scm.
;; The model is nanograd-am-transformer's make-am-transformer, trained
;; with am-training-step/ssa: the first step compiles forward, backward
;; and the gradients into one SSA program, and later steps replay it on
;; new batches.  Inference uses am-forward/ssa in the same way.
;;
;; 1. Character-level language model with greedy generation
;; 2. Copy task: reproduce a sequence after a separator (needs attention)
;; 3. Training with and without RoPE on a periodic sequence task
;; 4. Agreement with the strict model, and generation speed against the
;;    strict model with its KV cache
;;
;; Differences from the strict examples:
;;   - a batch of B sequences of length T is one [B*T, V] matrix of
;;     one-hot rows (row b*T + t is token t of sequence b), and targets
;;     are one-hot rows of the same shape;
;;   - a compiled program is fixed to B and T, so sequences are padded
;;     to a common length;
;;   - there is no KV cache: generation runs the compiled forward program
;;     on a window of T tokens, padded at the end.  Attention is causal,
;;     so the padding does not change the logits of the real tokens.
;;
;; Gradient diagnostics (nanograd-diagnostics, through
;; gradient-monitoring.scm) work as with strict tensors: the optimizer
;; is wrapped, and am-training-step/ssa calls its step! after installing
;; the gradients of the compiled program.  Examples 1 and 2 monitor
;; training and clip the global gradient norm.
;;
;; Run with: csi -s examples/am_transformer_example.scm
;; See am-example-backend.scm for the AM_EXAMPLE_BACKEND variable.

(import scheme
        (chicken base)
        (chicken blob)
        (chicken format)
        (chicken random)
        (chicken time)
        (srfi 1)
        (srfi 4)
        (srfi 69)
        yasos
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-transformer
        nanograd-am-transformer)

(include-relative "am-example-backend.scm")
(include-relative "gradient-monitoring.scm")

;;; ==================================================================
;;; Utilities
;;; ==================================================================

(define (one-hot-lazy sequences vocab-size)
  "Lazy tensor [B*T, VOCAB-SIZE] of the one-hot rows of B token lists
   of length T."
  (vector->lazy (am-tokens->one-hot sequences vocab-size 'f32)
                (list (fold + 0 (map length sequences)) vocab-size)))

(define (weighted-targets sequences weights vocab-size)
  "Like one-hot-lazy, but row t of each sequence holds (list-ref
   WEIGHTS t) instead of 1.  A weight of 0 leaves a position out of the
   cross-entropy, which averages over all B*T rows."
  (let ((v (am-tokens->one-hot sequences vocab-size 'f32))
        (t (length weights)))
    (for-each (lambda (seq b)
                (for-each (lambda (tok w pos)
                            (f32vector-set! v (+ (* (+ (* b t) pos) vocab-size) tok) w))
                          seq weights (iota t)))
              sequences (iota (length sequences)))
    (vector->lazy v (list (* t (length sequences)) vocab-size))))

;; A trainer couples a model, its optimizer and a morphism context.
;; Each call takes the input and target lazy tensors of one batch, whose
;; shapes must be the same at every call.
(define (make-trainer model optimizer)
  (let ((ctx (make-morphism-context)))
    (lambda (x-lt t-lt)
      (lazy-scalar
       (am-training-step/ssa ctx optimizer model
                             (lambda (logits) (am-cross-entropy-loss logits t-lt))
                             x-lt t-lt)))))

;; An evaluator returns the logits (an f32vector [B*T, V]) of a batch of
;; token lists, through a compiled forward program.
(define (make-evaluator model vocab-size)
  (let ((ctx (make-morphism-context)))
    (lambda (sequences)
      (tensor-data (am-forward/ssa ctx model (one-hot-lazy sequences vocab-size))))))

(define (argmax-row vec row width)
  "Column index of the largest element of row ROW of a row-major matrix."
  (let ((start (* row width)))
    (let loop ((i 1) (max-i 0) (max-val (f32vector-ref vec start)))
      (if (= i width)
          max-i
          (let ((val (f32vector-ref vec (+ start i))))
            (if (> val max-val)
                (loop (+ i 1) i val)
                (loop (+ i 1) max-i max-val)))))))

(define (greedy-generate evaluate prompt seq-len vocab-size n-tokens #!key (pad 0) (stop #f))
  "Generate up to N-TOKENS tokens after PROMPT by running EVALUATE on a
   window of the last SEQ-LEN tokens, padded with PAD at the end.
   Stops after the token STOP if it is given."
  (let loop ((seq prompt) (out '()) (i 0))
    (if (= i n-tokens)
        (reverse out)
        (let* ((window (if (> (length seq) seq-len) (take-right seq seq-len) seq))
               (len (length window))
               (logits (evaluate (list (append window (make-list (- seq-len len) pad)))))
               (next (argmax-row logits (- len 1) vocab-size)))
          (if (and stop (= next stop))
              (reverse (cons next out))
              (loop (append seq (list next)) (cons next out) (+ i 1)))))))

(define (random-tokens n lo hi)
  "N tokens drawn uniformly from [LO, HI)."
  (list-tabulate n (lambda (_) (+ lo (pseudo-random-integer (- hi lo))))))

(define (now-ms) (current-process-milliseconds))

;; A gradient monitor for the Transformer.  As explained in
;; transformer_example.scm, the tied token embedding gets gradients
;; thousands of times larger than the query and key projections, so the
;; imbalance check is set to 1e5 rather than the default 1000.
(define (make-transformer-monitor)
  (make-gradient-monitor exploding-threshold: 10.0
                         vanishing-threshold: 1e-7
                         imbalance-threshold: 1e5
                         history-size: 1000))

;;; ==================================================================
;;; Example 1: Character-level language model
;;; ==================================================================

(define (example-char-lm)
  (printf "\n=== Example 1: Character-Level Language Model ===\n")

  (define vocab '(#\space #\a #\b #\c #\d #\e #\f #\g #\h #\i #\j #\k #\l #\m
                  #\n #\o #\p #\q #\r #\s #\t #\u #\v #\w #\x #\y #\z #\.))
  (define vocab-size (length vocab))
  (define char->id
    (let ((ht (make-hash-table)))
      (for-each (lambda (c i) (hash-table-set! ht c i)) vocab (iota vocab-size))
      ht))
  (define id->char (list->vector vocab))
  (define period (hash-table-ref char->id #\.))

  (define training-texts
    '("the cat sat."
      "the dog ran."
      "a bird flew."
      "the sun set."
      "dogs and cats."))

  (define (text->ids text)
    (map (lambda (c) (hash-table-ref char->id c)) (string->list text)))
  (define (ids->text ids)
    (list->string (map (lambda (i) (vector-ref id->char i)) ids)))

  ;; All texts padded with periods to T+1 tokens; the whole set is one batch
  (define seq-len 14)
  (define padded
    (map (lambda (text)
           (let ((ids (text->ids text)))
             (append ids (make-list (- (+ seq-len 1) (length ids)) period))))
         training-texts))
  (define x-lt (one-hot-lazy (map (lambda (s) (drop-right s 1)) padded) vocab-size))
  (define t-lt (one-hot-lazy (map cdr padded) vocab-size))

  (define model (make-am-transformer vocab-size seq-len 64 2 4 use-rope?: #t dtype: 'f32))
  (printf "Model created with ~A parameters; batch ~A x ~A tokens\n"
          (am-parameter-count model) (length padded) seq-len)

  ;; The global gradient norm is clipped to 1, and the per-parameter
  ;; table is printed at the first (trace) step and at the last step
  (define monitor (make-transformer-monitor))
  (define optimizer
    (make-monitored-optimizer (make-adam (parameters model) learning-rate: 0.005)
                              (parameters model) monitor
                              clip-norm: 1.0 layer-report: '(1 120)))
  (define train! (make-trainer model optimizer))

  (printf "\nTraining...\n")
  (let ((t0 (now-ms)))
    (do ((step 1 (+ step 1))) ((> step 120))
      (let ((loss (train! x-lt t-lt)))
        (when (zero? (modulo step 20))
          (printf "Step ~A: loss = ~A, gradient norm = ~A\n"
                  step loss (fmt-norm (recent-mean-norm monitor 1))))))
    (printf "120 steps (the first compiles the program), with monitoring: ~A ms\n"
            (- (now-ms) t0)))
  (print-norm-trace monitor)
  (print-training-diagnosis optimizer "lazy character-level language model")

  (printf "\nGenerating text...\n")
  (let ((evaluate (make-evaluator model vocab-size)))
    (for-each
     (lambda (prompt)
       (printf "  \"~A\" -> \"~A~A\"\n" prompt prompt
               (ids->text (greedy-generate evaluate (text->ids prompt) seq-len vocab-size
                                           (- seq-len (string-length prompt))
                                           stop: period))))
     '("the c" "the d" "a b" "the s" "dogs"))))

;;; ==================================================================
;;; Example 2: Copy task
;;; ==================================================================

(define (example-copy-task)
  (printf "\n=== Example 2: Copy Task ===\n")

  ;; Sequences are  x1 .. xk SEP x1 .. xk  with SEP = 0.  After the
  ;; separator, predicting the next token means finding the matching
  ;; token in the first half, which needs attention.  Only the k
  ;; predictions after the separator count: their target rows have
  ;; weight T/k and the others weight 0, so the loss is the mean
  ;; cross-entropy over the copied tokens.
  (define vocab-size 10)
  (define k 5)
  (define sep 0)
  (define seq-len (* 2 k))
  (define batch-size 8)

  (define (make-example)
    (let ((xs (random-tokens k 1 vocab-size)))
      (append xs (list sep) xs)))

  (define weights
    (map (lambda (pos) (if (>= pos k) (exact->inexact (/ seq-len k)) 0.0)) (iota seq-len)))

  (define model (make-am-transformer vocab-size seq-len 32 2 4 use-rope?: #t dtype: 'f32))
  (define monitor (make-transformer-monitor))
  (define optimizer
    (make-monitored-optimizer (make-adam (parameters model) learning-rate: 0.005)
                              (parameters model) monitor clip-norm: 1.0))
  (define train! (make-trainer model optimizer))
  (printf "Model created with ~A parameters; batches of ~A sequences\n"
          (am-parameter-count model) batch-size)

  (printf "Training copy task...\n")
  (do ((epoch 1 (+ epoch 1))) ((> epoch 40))
    (let ((total
           (fold (lambda (_ acc)
                   (let ((batch (list-tabulate batch-size (lambda (_) (make-example)))))
                     (+ acc (train! (one-hot-lazy (map (lambda (s) (drop-right s 1)) batch)
                                                  vocab-size)
                                    (weighted-targets (map cdr batch) weights vocab-size)))))
                 0.0 (iota 10))))
      (when (zero? (modulo epoch 10))
        (printf "Epoch ~A: avg loss = ~A, mean gradient norm = ~A\n"
                epoch (/ total 10) (fmt-norm (recent-mean-norm monitor 100))))))
  (print-training-diagnosis optimizer "lazy copy task")

  (printf "\nTesting copy ability on new sequences...\n")
  (let* ((evaluate (make-evaluator model vocab-size))
         (tests (list-tabulate batch-size (lambda (_) (make-example))))
         (logits (evaluate (map (lambda (s) (drop-right s 1)) tests)))
         (copies (map (lambda (b)
                        (map (lambda (pos) (argmax-row logits (+ (* b seq-len) pos) vocab-size))
                             (iota k k)))
                      (iota batch-size)))
         (correct (fold + 0 (map (lambda (seq copy) (count = copy (drop seq (+ k 1))))
                                 tests copies))))
    (for-each (lambda (seq copy) (printf "  Input: ~A  copy: ~A\n" (take seq k) copy))
              (take tests 3) (take copies 3))
    (printf "Accuracy: ~A/~A tokens\n" correct (* batch-size k))))

;;; ==================================================================
;;; Example 3: With and without RoPE
;;; ==================================================================

(define (example-rope-comparison)
  (printf "\n=== Example 3: RoPE Comparison ===\n")

  ;; Each sequence repeats a random pattern of 3 distinct tokens, e.g.
  ;; 4 7 2 4 7 2 ...  From the third token on, the next token is the one
  ;; two positions back, so the model must know the order of the tokens.
  ;; Without RoPE the model has no position information at all.
  (define vocab-size 10)
  (define seq-len 12)
  (define batch-size 8)

  (define (make-sequence)
    (let* ((a (pseudo-random-integer vocab-size))
           (b (let loop () (let ((t (pseudo-random-integer vocab-size))) (if (= t a) (loop) t))))
           (c (let loop () (let ((t (pseudo-random-integer vocab-size)))
                             (if (or (= t a) (= t b)) (loop) t))))
           (pattern (vector a b c)))
      (list-tabulate (+ seq-len 1) (lambda (i) (vector-ref pattern (modulo i 3))))))

  (define (train-model use-rope?)
    (let* ((model (make-am-transformer vocab-size seq-len 32 2 4
                                       use-rope?: use-rope? dtype: 'f32))
           (train! (make-trainer model (make-adam (parameters model) learning-rate: 0.005))))
      (do ((step 0 (+ step 1))) ((= step 150))
        (let ((batch (list-tabulate batch-size (lambda (_) (make-sequence)))))
          (train! (one-hot-lazy (map (lambda (s) (drop-right s 1)) batch) vocab-size)
                  (one-hot-lazy (map cdr batch) vocab-size))))
      ;; Accuracy on the predictable positions 2 .. seq-len-1 of new sequences
      (let* ((tests (list-tabulate batch-size (lambda (_) (make-sequence))))
             (logits ((make-evaluator model vocab-size) (map (lambda (s) (drop-right s 1)) tests)))
             (correct
              (fold + 0
                    (map (lambda (seq b)
                           (count (lambda (pos)
                                    (= (argmax-row logits (+ (* b seq-len) pos) vocab-size)
                                       (list-ref seq (+ pos 1))))
                                  (iota (- seq-len 2) 2)))
                         tests (iota batch-size)))))
        (/ correct (* batch-size (- seq-len 2))))))

  (printf "Training without RoPE...\n")
  (printf "Accuracy without RoPE: ~A%\n" (* 100.0 (train-model #f)))
  (printf "\nTraining with RoPE...\n")
  (printf "Accuracy with RoPE: ~A%\n" (* 100.0 (train-model #t))))

;;; ==================================================================
;;; Example 4: Agreement with the strict model and generation speed
;;; ==================================================================

(define (example-strict-agreement)
  (printf "\n=== Example 4: Agreement with the Strict Model ===\n")

  ;; The lazy Transformer has the parameters of make-transformer in the
  ;; same order, so am-copy-parameters! can transfer them.
  (define vocab-size 100)
  (define seq-len 50)
  (define d-model 64)
  (define n-layers 4)
  (define n-heads 8)
  (define strict (make-transformer vocab-size seq-len d-model n-layers n-heads
                                   use-rope?: #t dtype: 'f32))
  (define model (make-am-transformer vocab-size seq-len d-model n-layers n-heads
                                     use-rope?: #t dtype: 'f32))
  (am-copy-parameters! strict model)
  (define evaluate (make-evaluator model vocab-size))

  (define tokens (random-tokens seq-len 0 vocab-size))
  (let ((strict-logits (tensor-data (forward strict tokens)))
        (lazy-logits (evaluate (list tokens))))
    (printf "Largest logit difference over ~A positions: ~A\n" seq-len
            (fold max 0.0 (map (lambda (i) (abs (- (f32vector-ref strict-logits i)
                                                    (f32vector-ref lazy-logits i))))
                               (iota (* seq-len vocab-size))))))

  ;; Generate 30 tokens after a 20-token context with both models
  (define context (take tokens 20))
  (define gen-length 30)

  (let ((t0 (now-ms)))
    (define cache (make-kv-cache n-layers n-heads seq-len (quotient d-model n-heads)
                                 dtype: 'f32))
    (define strict-tokens
      (let loop ((pos 0) (logits (forward strict context kv-cache: cache)) (out '()))
        (let ((next-id (argmax-row (tensor-data logits) 0 vocab-size)))
          (if (= (+ pos 1) gen-length)
              (reverse (cons next-id out))
              (loop (+ pos 1) (forward strict (list next-id) kv-cache: cache)
                    (cons next-id out))))))
    (define t1 (now-ms))
    ;; The first call compiled the program above; this times replays only
    (define lazy-tokens (greedy-generate evaluate context seq-len vocab-size gen-length))
    (define t2 (now-ms))
    (printf "\nGenerating ~A tokens after a ~A-token context:\n" gen-length (length context))
    (printf "Strict model with KV cache: ~A ms\n" (- t1 t0))
    (printf "Lazy model, compiled forward on the whole window per token: ~A ms\n" (- t2 t1))
    (printf "Same tokens generated: ~A\n" (equal? strict-tokens lazy-tokens))))

;;; ==================================================================
;;; Run all examples
;;; ==================================================================

(define (run-all-examples)
  (printf "\n")
  (printf "========================================\n")
  (printf "LAZY (ARRAY-MORPHISMS) TRANSFORMER EXAMPLES\n")
  (printf "backend: ~A\n" am-backend)
  (printf "========================================\n")

  (set-pseudo-random-seed! (string->blob "42"))

  (example-char-lm)
  (example-copy-task)
  (example-rope-comparison)
  (example-strict-agreement)

  (printf "\n")
  (printf "========================================\n")
  (printf "ALL EXAMPLES COMPLETE\n")
  (printf "========================================\n\n"))

(run-all-examples)
