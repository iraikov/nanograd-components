;; Examples of training and generating with the Transformer, using the
;; strict (eagerly evaluated) nanograd tensors.
;;
;; 1. Character-level language model with greedy generation
;; 2. Copy task: reproduce a sequence after a separator (needs attention)
;; 3. Training with and without RoPE on a periodic sequence task
;; 4. Generation with and without the KV cache
;; 5. Diagnosing an unstable run: SGD with a too-large learning rate,
;;    with and without gradient clipping
;;
;; The training examples use the gradient diagnostics of
;; nanograd-diagnostics through gradient-monitoring.scm.  The optimizer
;; is wrapped so that every step records the gradient norms in a
;; monitor, which flags exploding, vanishing or unbalanced gradients,
;; and can clip the global gradient norm before the update.
;;
;; Each training step computes the cross-entropy of the logits of a
;; whole sequence, (seq_len, vocab_size), against its target token ids,
;; so the gradient flows through the model.
;;
;; The array-morphisms version is examples/am_transformer_example.scm.
;; Run with: csi -s examples/transformer_example.scm

(import scheme
        (chicken base)
        (chicken blob)
        (chicken format)
        (chicken random)
        (chicken time)
        (srfi 1)
        (srfi 4)
        (srfi 69)
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-transformer)

(include-relative "gradient-monitoring.scm")

;;; ==================================================================
;;; Utilities
;;; ==================================================================

(define (ids->tensor ids)
  "Token ids as a (len) tensor of class indices, for cross-entropy-loss."
  (make-tensor32 (list->f32vector (map exact->inexact ids)) (list (length ids))
                 requires-grad?: #f))

(define (sequence-loss logits target-ids)
  "Mean cross-entropy of LOGITS (seq_len, vocab) against TARGET-IDS."
  (cross-entropy-loss logits (ids->tensor target-ids) from-logits: #t))

(define (train-step! model optimizer loss)
  (backward! loss)
  (step! optimizer)
  (zero-grad-layer! model)
  (f32vector-ref (tensor-data loss) 0))

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

(define (predictions logits vocab-size)
  "Greedy prediction for every row of LOGITS (seq_len, vocab)."
  (let ((data (tensor-data logits)))
    (map (lambda (pos) (argmax-row data pos vocab-size))
         (iota (car (tensor-shape logits))))))

(define (count-params model)
  (fold (lambda (p acc) (+ acc (f32vector-length (tensor-data p))))
        0 (parameters model)))

(define (random-tokens n lo hi)
  "N tokens drawn uniformly from [LO, HI)."
  (list-tabulate n (lambda (_) (+ lo (pseudo-random-integer (- hi lo))))))

;; A gradient monitor for the Transformer.  Its per-parameter gradient
;; norms are unbalanced by design: the token embedding (parameter 0) is
;; tied to the output projection and receives the largest gradient,
;; while at initialization attention is nearly uniform, so the query and
;; key projections receive gradients about 1000 times smaller.  The
;; imbalance check is therefore set to 1e5 rather than the default 1000.
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

  (define training-texts
    '("the cat sat."
      "the dog ran."
      "a bird flew."
      "the sun set."
      "dogs and cats."))

  (define (text->ids text)
    (map (lambda (c) (hash-table-ref char->id c)) (string->list text)))

  (define max-seq-len 32)
  (define d-model 64)
  (define n-layers 2)
  (define n-heads 4)

  (define model
    (make-transformer vocab-size max-seq-len d-model n-layers n-heads
                      use-rope?: #t dtype: 'f32))
  (printf "Model created with ~A parameters\n" (count-params model))

  ;; Clipping the global gradient norm to 1 is common for Transformers.
  ;; The per-parameter table is printed at the first step, where the
  ;; query and key projections (parameters 4-7 of the first layer)
  ;; receive the smallest gradients.
  (define monitor (make-transformer-monitor))
  (define optimizer
    (make-monitored-optimizer (make-adam (parameters model) learning-rate: 0.005)
                              (parameters model) monitor
                              clip-norm: 1.0 layer-report: '(1)))

  (printf "\nTraining...\n")
  (do ((epoch 1 (+ epoch 1))) ((> epoch 60))
    (let ((total-loss
           (fold (lambda (text acc)
                   (let ((ids (text->ids text)))
                     (+ acc (train-step! model optimizer
                                         (sequence-loss (forward model (drop-right ids 1))
                                                        (cdr ids))))))
                 0.0 training-texts)))
      (when (zero? (modulo epoch 10))
        (printf "Epoch ~A: avg loss = ~A, mean gradient norm = ~A\n"
                epoch (/ total-loss (length training-texts))
                (fmt-norm (recent-mean-norm monitor (* 10 (length training-texts))))))))
  (print-norm-trace monitor)
  (print-training-diagnosis optimizer "character-level language model")

  ;; Greedy generation with the KV cache: the prompt is processed in one
  ;; call, then each generated token is fed back one at a time.
  (printf "\nGenerating text...\n")
  (for-each
   (lambda (prompt)
     (let* ((cache (make-kv-cache n-layers n-heads max-seq-len (quotient d-model n-heads)
                                  dtype: 'f32))
            (prompt-ids (text->ids prompt)))
       (let loop ((logits (forward model prompt-ids kv-cache: cache))
                  (out (reverse (string->list prompt)))
                  (len (length prompt-ids)))
         (let* ((next-id (argmax-row (tensor-data logits) 0 vocab-size))
                (next-char (vector-ref id->char next-id))
                (out (cons next-char out)))
           (if (or (char=? next-char #\.) (>= (+ len 1) max-seq-len))
               (printf "  \"~A\" -> \"~A\"\n" prompt (list->string (reverse out)))
               (loop (forward model (list next-id) kv-cache: cache) out (+ len 1)))))))
   '("the c" "the d" "a b" "the s" "dogs")))

;;; ==================================================================
;;; Example 2: Copy task
;;; ==================================================================

(define (example-copy-task)
  (printf "\n=== Example 2: Copy Task ===\n")

  ;; Sequences are  x1 .. xk SEP x1 .. xk  with SEP = 0.  After the
  ;; separator, predicting the next token means finding the matching
  ;; token in the first half, which needs attention.  The loss covers
  ;; only the k predictions after the separator.
  (define vocab-size 10)
  (define k 5)
  (define sep 0)

  (define (make-example)
    (let ((xs (random-tokens k 1 vocab-size)))
      (append xs (list sep) xs)))

  (define model (make-transformer vocab-size 16 32 2 4 use-rope?: #t dtype: 'f32))
  (define monitor (make-transformer-monitor))
  (define optimizer
    (make-monitored-optimizer (make-adam (parameters model) learning-rate: 0.005)
                              (parameters model) monitor clip-norm: 1.0))
  (printf "Model created with ~A parameters\n" (count-params model))

  (define (copy-loss seq)
    (let ((logits (forward model (drop-right seq 1))))
      ;; rows k .. 2k-1 predict the tokens after the separator
      (sequence-loss (slice-tensor logits k k) (drop seq (+ k 1)))))

  ;; Each step averages the loss of a few sequences: single-sequence
  ;; updates are too noisy for the model to settle on copying.
  (define batch-size 4)
  (define (batch-loss)
    (let ((ls (list-tabulate batch-size (lambda (_) (copy-loss (make-example))))))
      (scale-op (fold (lambda (l acc) (add acc l)) (car ls) (cdr ls))
                (/ 1.0 batch-size))))

  (printf "Training copy task (~A sequences per step)...\n" batch-size)
  (do ((epoch 1 (+ epoch 1))) ((> epoch 50))
    (let ((total-loss
           (fold (lambda (_ acc) (+ acc (train-step! model optimizer (batch-loss))))
                 0.0 (iota 10))))
      (when (zero? (modulo epoch 10))
        (printf "Epoch ~A: avg loss = ~A, mean gradient norm = ~A\n"
                epoch (/ total-loss 10) (fmt-norm (recent-mean-norm monitor 100))))))
  (print-training-diagnosis optimizer "copy task")

  (printf "\nTesting copy ability on new sequences...\n")
  (let ((correct
         (fold (lambda (i acc)
                 (let* ((seq (make-example))
                        (preds (drop (predictions (forward model (drop-right seq 1)) vocab-size) k))
                        (targets (drop seq (+ k 1))))
                   (when (< i 3)
                     (printf "  Input: ~A  copy: ~A\n" (take seq k) preds))
                   (+ acc (count = preds targets))))
               0 (iota 20))))
    (printf "Accuracy: ~A/~A tokens\n" correct (* 20 k))))

;;; ==================================================================
;;; Example 3: With and without RoPE
;;; ==================================================================

(define (example-rope-comparison)
  (printf "\n=== Example 3: RoPE Comparison ===\n")

  ;; Each sequence repeats a random pattern of 3 distinct tokens, e.g.
  ;; 4 7 2 4 7 2 ...  From the third token on, the next token is the one two
  ;; positions back, so the model must know the order of the tokens.
  ;; Without RoPE the model has no position information at all.
  (define vocab-size 10)
  (define seq-len 12)

  (define (make-sequence)
    (let* ((a (pseudo-random-integer vocab-size))
           (b (let loop () (let ((t (pseudo-random-integer vocab-size))) (if (= t a) (loop) t))))
           (c (let loop () (let ((t (pseudo-random-integer vocab-size)))
                             (if (or (= t a) (= t b)) (loop) t))))
           (pattern (vector a b c)))
      (list-tabulate (+ seq-len 1) (lambda (i) (vector-ref pattern (modulo i 3))))))

  (define (train-model use-rope?)
    (let* ((model (make-transformer vocab-size 32 32 2 4 use-rope?: use-rope? dtype: 'f32))
           (opt (make-adam (parameters model) learning-rate: 0.005)))
      (do ((step 0 (+ step 1))) ((= step 300))
        (let ((seq (make-sequence)))
          (train-step! model opt (sequence-loss (forward model (drop-right seq 1)) (cdr seq)))))
      ;; Accuracy on the predictable positions 2 .. seq-len-1 of new sequences
      (let ((correct
             (fold (lambda (_ acc)
                     (let* ((seq (make-sequence))
                            (preds (drop (predictions (forward model (drop-right seq 1)) vocab-size) 2)))
                       (+ acc (count = preds (drop seq 3)))))
                   0 (iota 20))))
        (/ correct (* 20 (- seq-len 2))))))

  (printf "Training without RoPE...\n")
  (printf "Accuracy without RoPE: ~A%\n" (* 100.0 (train-model #f)))
  (printf "\nTraining with RoPE...\n")
  (printf "Accuracy with RoPE: ~A%\n" (* 100.0 (train-model #t))))

;;; ==================================================================
;;; Example 4: KV cache
;;; ==================================================================

(define (example-kv-cache-speed)
  (printf "\n=== Example 4: KV Cache Speed Test ===\n")

  (define vocab-size 100)
  (define max-seq-len 128)
  (define d-model 64)
  (define n-layers 4)
  (define n-heads 8)
  (define model (make-transformer vocab-size max-seq-len d-model n-layers n-heads
                                  use-rope?: #t dtype: 'f32))

  (define context (random-tokens 20 0 vocab-size))
  (define gen-length 30)

  ;; Without the cache, every step runs the whole sequence again
  (printf "Generating ~A tokens without cache...\n" gen-length)
  (define t0 (current-process-milliseconds))
  (define generated-no-cache
    (let loop ((pos 0) (seq context) (out '()))
      (if (= pos gen-length)
          (reverse out)
          (let* ((logits (forward model seq))
                 (next-id (argmax-row (tensor-data logits) (- (length seq) 1) vocab-size)))
            (loop (+ pos 1) (append seq (list next-id)) (cons next-id out))))))
  (define time-no-cache (- (current-process-milliseconds) t0))
  (printf "Time without cache: ~A ms\n" time-no-cache)

  ;; With the cache, the context is processed once and each new token
  ;; attends to the cached keys and values
  (printf "\nGenerating ~A tokens with cache...\n" gen-length)
  (define t1 (current-process-milliseconds))
  (define generated-cache
    (let ((cache (make-kv-cache n-layers n-heads max-seq-len (quotient d-model n-heads)
                                dtype: 'f32)))
      (let loop ((pos 0) (logits (forward model context kv-cache: cache)) (out '()))
        (let ((next-id (argmax-row (tensor-data logits) 0 vocab-size)))
          (if (= (+ pos 1) gen-length)
              (reverse (cons next-id out))
              (loop (+ pos 1) (forward model (list next-id) kv-cache: cache)
                    (cons next-id out)))))))
  (define time-cache (- (current-process-milliseconds) t1))
  (printf "Time with cache: ~A ms\n" time-cache)
  (printf "Speedup: ~Ax\n" (/ (round (* 10.0 (/ time-no-cache (max 1 time-cache)))) 10.0))
  (printf "Same tokens generated: ~A\n" (equal? generated-no-cache generated-cache)))

;;; ==================================================================
;;; Example 5: Diagnosing an unstable run
;;; ==================================================================

(define (example-unstable-training)
  (printf "\n=== Example 5: Diagnosing an Unstable Run ===\n")

  ;; The copy task of example 2, one sequence per step, trained with
  ;; plain SGD at a learning rate of 1, which is far too large.  Both runs
  ;; start from the same weights and see the same sequences; the second
  ;; clips the global gradient norm to 1 before every update.
  (define vocab-size 10)
  (define k 5)

  (define (run clip-norm)
    (set-pseudo-random-seed! (string->blob "unstable"))
    (let* ((model (make-transformer vocab-size 16 32 2 4 use-rope?: #t dtype: 'f32))
           (monitor (make-transformer-monitor))
           (opt (make-monitored-optimizer (make-sgd (parameters model) learning-rate: 1.0)
                                          (parameters model) monitor
                                          clip-norm: clip-norm max-warnings: 2)))
      (printf "\n~A\n" (if clip-norm
                            (sprintf "SGD, learning rate 1, gradient norm clipped to ~A:" clip-norm)
                            "SGD, learning rate 1, no clipping:"))
      (do ((step 1 (+ step 1))) ((> step 60))
        (let* ((xs (random-tokens k 1 vocab-size))
               (seq (append xs (list 0) xs))
               (loss (train-step! model opt (sequence-loss (forward model (drop-right seq 1))
                                                           (cdr seq)))))
          (when (zero? (modulo step 15))
            (printf "  step ~A: loss = ~A\n" step loss))))
      (print-norm-trace monitor buckets: 6)
      (print-training-diagnosis opt (if clip-norm "with clipping" "without clipping"))))

  ;; Without clipping the gradient norm grows by orders of magnitude
  ;; within a few dozen steps, and the loss with it.  The run then ends
  ;; with zero gradients and a loss of ln 10 = 2.3026: the network
  ;; predicts the uniform distribution and has stopped learning.  The
  ;; diagnosis of the last ten steps reports vanishing gradients; the
  ;; norm trace shows that they followed an explosion.
  (run #f)
  (run 1.0))

;;; ==================================================================
;;; Run all examples
;;; ==================================================================

(define (run-all-examples)
  (printf "\n")
  (printf "========================================\n")
  (printf "TRANSFORMER EXAMPLES\n")
  (printf "========================================\n")

  (set-pseudo-random-seed! (string->blob "42"))

  (example-char-lm)
  (example-copy-task)
  (example-rope-comparison)
  (example-kv-cache-speed)
  (example-unstable-training)

  (printf "\n")
  (printf "========================================\n")
  (printf "ALL EXAMPLES COMPLETE\n")
  (printf "========================================\n\n"))

(run-all-examples)
