;; Examples of ResNet training and evaluation with lazy
;; (array-morphisms) tensors.
;;
;; This is the array-morphisms counterpart of resnet_example.scm.  The
;; model is nanograd-am-resnet's make-am-resnet: its forward pass builds
;; a morphism graph, am-training-step/ssa compiles forward, backward and
;; the gradients of a whole training step into one SSA program on the
;; first call, and later calls replay that program on new batches.
;; am-forward/ssa does the same for inference.
;;
;; 1. Classification of synthetic striped images
;; 2. Training with a step learning-rate schedule
;; 3. Agreement with the strict model (make-resnet-cifar) after copying
;;    its weights
;; 4. Compiled inference: trace time, replay time and strict forward time
;; 5. Transfer learning: reuse a trained backbone for new classes
;;
;; Differences from the strict examples:
;;   - images are NHWC, [N, H, W, C];
;;   - a compiled program is fixed to the shapes of its first call, so
;;     every batch of a training context has the same size, and
;;     inference uses one context per batch size;
;;   - batch normalization always uses the statistics of the current
;;     batch (there are no running statistics), so inference should be
;;     done on batches, not single images;
;;   - there is no max pooling and no ImageNet-style 7x7 stem.
;;
;; Gradient diagnostics (nanograd-diagnostics, through
;; gradient-monitoring.scm) work the same way as with strict tensors.
;; am-training-step/ssa installs the gradients of the compiled program in
;; the parameters and then calls the optimizer's step!, so wrapping the
;; optimizer gives the monitor the gradients of every replayed step.
;; Example 1 monitors training, and example 3 checks that the strict and
;; the lazy model have the same per-parameter gradient statistics.
;;
;; Run with: csi -s examples/am_resnet_example.scm
;; See am-example-backend.scm for the AM_EXAMPLE_BACKEND variable.

(import scheme
        (chicken base)
        (chicken blob)
        (chicken format)
        (chicken random)
        (chicken time)
        (srfi 1)
        (srfi 4)
        yasos
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-resnet
        nanograd-am-resnet)

(include-relative "am-example-backend.scm")
(include-relative "gradient-monitoring.scm")

;;; ==================================================================
;;; Data and utilities
;;; ==================================================================

(define img-size 16)
(define channels 3)

(define (make-image label noise)
  "A CHANNELS x IMG-SIZE x IMG-SIZE image (NCHW element order) of
   stripes whose orientation and frequency depend on LABEL, with
   uniform noise of amplitude NOISE.  Same images as resnet_example.scm."
  (let* ((data (make-f32vector (* channels img-size img-size) 0.0))
         (angle (* label 0.314159))
         (freq (+ 0.4 (* 0.15 (modulo label 3))))
         (phase (* 6.283 (pseudo-random-real))))
    (do ((c 0 (+ c 1))) ((= c channels) data)
      (do ((i 0 (+ i 1))) ((= i img-size))
        (do ((j 0 (+ j 1))) ((= j img-size))
          (f32vector-set! data (+ (* c img-size img-size) (* i img-size) j)
                          (+ 0.5
                             (* 0.3 (sin (+ phase
                                            (* freq (+ (* i (cos angle))
                                                       (* j (sin angle))))
                                            (* 0.5 c))))
                             (* noise (- (pseudo-random-real) 0.5)))))))))

(define (make-dataset labels per-class noise)
  "A list of (image-data . label) pairs, PER-CLASS for each of LABELS."
  (append-map (lambda (label)
                (list-tabulate per-class (lambda (_) (cons (make-image label noise) label))))
              labels))

(define (batch->vectors batch num-classes #!key (label-offset 0))
  "The NCHW input and one-hot [N, K] target f32vectors of a list of
   (image-data . label) pairs; labels are shifted down by LABEL-OFFSET."
  (let* ((n (length batch))
         (size (* channels img-size img-size))
         (x (make-f32vector (* n size)))
         (t (make-f32vector (* n num-classes) 0.0)))
    (for-each
     (lambda (sample b)
       (let ((img (car sample)))
         (do ((i 0 (+ i 1))) ((= i size))
           (f32vector-set! x (+ (* b size) i) (f32vector-ref img i))))
       (f32vector-set! t (+ (* b num-classes) (- (cdr sample) label-offset)) 1.0))
     batch (iota n))
    (values x t)))

(define (batch->lazy batch num-classes #!key (label-offset 0))
  "The NHWC input [N, H, W, C] and one-hot target [N, K] lazy tensors."
  (let-values (((x t) (batch->vectors batch num-classes label-offset: label-offset)))
    (let ((n (length batch)))
      (values (vector->lazy (am-nchw->nhwc x n channels img-size img-size 'f32)
                            (list n img-size img-size channels))
              (vector->lazy t (list n num-classes))))))

(define (list-shuffle lst)
  "Fisher-Yates shuffle."
  (let ((vec (list->vector lst)))
    (do ((i (- (vector-length vec) 1) (- i 1)))
        ((< i 1) (vector->list vec))
      (let* ((j (pseudo-random-integer (+ i 1)))
             (temp (vector-ref vec i)))
        (vector-set! vec i (vector-ref vec j))
        (vector-set! vec j temp)))))

(define (full-batches lst size)
  "Split LST into consecutive lists of exactly SIZE elements; a shorter
   remainder is dropped, because a compiled program has a fixed batch size."
  (if (< (length lst) size)
      '()
      (cons (take lst size) (full-batches (drop lst size) size))))

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

(define (now-ms) (current-process-milliseconds))

;;; ------------------------------------------------------------------
;;; Training and evaluation
;;; ------------------------------------------------------------------

;; A trainer couples a model, its optimizer and a morphism context.  The
;; first step compiles the SSA program; later steps replay it.
(define (make-trainer model optimizer num-classes #!key (label-offset 0))
  (let ((ctx (make-morphism-context)))
    (lambda (batch)
      (let-values (((x-lt t-lt) (batch->lazy batch num-classes label-offset: label-offset)))
        (am-training-step/ssa ctx optimizer model
                              (lambda (logits) (am-cross-entropy-loss logits t-lt))
                              x-lt t-lt)))))

(define (train-epoch! trainer data batch-size)
  "One pass over DATA in shuffled batches of BATCH-SIZE.  Returns the
   mean loss."
  (let ((bs (full-batches (list-shuffle data) batch-size)))
    (/ (fold (lambda (b acc) (+ acc (lazy-scalar (trainer b)))) 0.0 bs)
       (length bs))))

;; An evaluator runs the compiled forward program of MODEL on whole
;; datasets; it keeps one context per dataset size.
(define (make-evaluator model num-classes #!key (label-offset 0))
  (let ((contexts '()))
    (lambda (data)
      (let* ((n (length data))
             (ctx (or (alist-ref n contexts)
                      (let ((c (make-morphism-context)))
                        (set! contexts (cons (cons n c) contexts))
                        c))))
        (let-values (((x-lt t-lt) (batch->lazy data num-classes label-offset: label-offset)))
          (tensor-data (am-forward/ssa ctx model x-lt)))))))

(define (accuracy evaluate data num-classes #!key (label-offset 0))
  (let ((logits (evaluate data)))
    (/ (count (lambda (b sample)
                (= (argmax-row logits b num-classes) (- (cdr sample) label-offset)))
              (iota (length data)) data)
       (length data))))

;;; ==================================================================
;;; Example 1: Synthetic image classification
;;; ==================================================================

(define (example-classification)
  (printf "\n=== Example 1: Synthetic Image Classification ===\n")

  (define num-classes 10)
  (define train-data (make-dataset (iota num-classes) 16 0.2))
  (define test-data (make-dataset (iota num-classes) 4 0.2))

  (printf "~A training and ~A test images of size ~Ax~Ax~A (NHWC)\n"
          (length train-data) (length test-data) img-size img-size channels)

  (define model (make-am-resnet '(1 1 1) '(8 16 32) num-classes dtype: 'f32))
  (printf "Lazy CIFAR-style ResNet with ~A parameters\n" (am-parameter-count model))

  ;; As in resnet_example.scm, the convolution biases followed by batch
  ;; normalization get no gradient; they are reported at the first step
  ;; and left out of the monitor.  The first step is the trace step.
  (define monitor (make-gradient-monitor exploding-threshold: 10.0
                                         vanishing-threshold: 1e-6
                                         history-size: 1000))
  (define optimizer
    (make-monitored-optimizer (make-adam (parameters model) learning-rate: 0.01)
                              (parameters model) monitor
                              layer-report: '(1) exclude-inactive?: #t))
  (define trainer (make-trainer model optimizer num-classes))

  (printf "\nTraining (the first step compiles the SSA program)...\n")
  (let ((t0 (now-ms)))
    (trainer (take train-data 16))
    (printf "First step: ~A ms\n" (- (now-ms) t0)))
  (let ((t0 (now-ms)))
    (do ((epoch 1 (+ epoch 1))) ((> epoch 6))
      (printf "Epoch ~A: loss=~A, mean gradient norm=~A\n"
              epoch (train-epoch! trainer train-data 16) (fmt-norm (recent-mean-norm monitor 10))))
    (printf "Six epochs (60 replayed steps, with monitoring): ~A ms\n" (- (now-ms) t0)))
  (print-norm-trace monitor)
  (print-training-diagnosis optimizer "lazy ResNet classification")

  (define evaluate (make-evaluator model num-classes))
  (printf "\nTest accuracy: ~A%\n" (* 100.0 (accuracy evaluate test-data num-classes)))

  (printf "\nPredictions for one image per class (evaluated as one batch):\n")
  (let* ((samples (make-dataset (iota num-classes) 1 0.2))
         (logits (evaluate samples)))
    (for-each (lambda (sample b)
                (let ((pred (argmax-row logits b num-classes)))
                  (printf "  True: ~A, Pred: ~A ~A\n" (cdr sample) pred
                          (if (= pred (cdr sample)) "(correct)" "(wrong)"))))
              samples (iota num-classes))))

;;; ==================================================================
;;; Example 2: Learning-rate schedule
;;; ==================================================================

(define (example-lr-schedule)
  (printf "\n=== Example 2: Training with a Learning-Rate Schedule ===\n")

  (define num-classes 4)
  (define model (make-am-resnet '(1 1) '(8 16) num-classes dtype: 'f32))
  (define optimizer (make-adam (parameters model) learning-rate: 0.02))
  (define trainer (make-trainer model optimizer num-classes))

  ;; Step schedule: divide the rate by 10 at epochs 4 and 7.  The
  ;; optimizer reads its rate at every step, so changing it does not
  ;; require recompiling the program.
  (define (get-lr epoch)
    (cond ((< epoch 4) 0.02)
          ((< epoch 7) 0.002)
          (else 0.0002)))

  (printf "Each epoch draws 4 fresh batches of 16 noisy images\n")
  (do ((epoch 1 (+ epoch 1))) ((> epoch 8))
    (set-learning-rate! optimizer (get-lr epoch))
    (printf "Epoch ~A: LR=~A, loss=~A\n"
            epoch (get-learning-rate optimizer)
            (train-epoch! trainer (make-dataset (iota num-classes) 16 0.6) 16)))
  (printf "Training complete!\n"))

;;; ==================================================================
;;; Example 3: Agreement with the strict model
;;; ==================================================================

(define (example-strict-agreement)
  (printf "\n=== Example 3: Agreement with the Strict Model ===\n")

  ;; The lazy ResNet has the parameters of make-resnet-cifar in the same
  ;; order and element order, so am-copy-parameters! can transfer them.
  (define num-classes 10)
  (define stages '(1 1 1))
  (define widths '(8 16 32))
  (define strict (make-resnet-cifar stages widths num-classes dtype: 'f32))
  (define model (make-am-resnet stages widths num-classes dtype: 'f32))
  (am-copy-parameters! strict model)
  (set-training-mode! strict #t)

  (printf "Parameters: strict ~A, lazy ~A\n"
          (fold + 0 (map (lambda (p) (apply * (tensor-shape p))) (parameters strict)))
          (am-parameter-count model))

  (define batch (make-dataset (iota num-classes) 1 0.2))
  (define n (length batch))

  (let*-values (((x t) (batch->vectors batch num-classes))
                ((strict-logits)
                 (tensor-data (forward strict (make-tensor32 x (list n channels img-size img-size)
                                                             requires-grad?: #f))))
                ((lazy-logits) ((make-evaluator model num-classes) batch)))
    (printf "Largest logit difference over a batch of ~A: ~A\n" n
            (fold max 0.0 (map (lambda (i) (abs (- (f32vector-ref strict-logits i)
                                                    (f32vector-ref lazy-logits i))))
                               (iota (* n num-classes)))))
    (printf "Same predictions: ~A\n"
            (every (lambda (b) (= (argmax-row strict-logits b num-classes)
                                  (argmax-row lazy-logits b num-classes)))
                   (iota n))))

  ;; One training step of each with the same batch.  The strict
  ;; gradients come from backward!; the lazy ones from the compiled
  ;; program, read by an optimizer whose step! only takes their
  ;; statistics (gradient-stats) and leaves the parameters unchanged.
  (let*-values (((x t) (batch->vectors batch num-classes))
                ((strict-loss strict-stats)
                 (let ((l (cross-entropy-loss
                           (forward strict (make-tensor32 x (list n channels img-size img-size)
                                                          requires-grad?: #f))
                           (make-tensor32 t (list n num-classes) requires-grad?: #f)
                           from-logits: #t)))
                   (backward! l)
                   (values (f32vector-ref (tensor-data l) 0) (gradient-stats (parameters strict)))))
                ((lazy-stats) #f)
                ((lazy-loss)
                 (lazy-scalar ((make-trainer model
                                             (object ((optimizer? self) #t)
                                                     ((step! self)
                                                      (set! lazy-stats
                                                            (gradient-stats (parameters model)))))
                                             num-classes)
                               batch))))
    (printf "Training loss: strict ~A, lazy ~A\n" strict-loss lazy-loss)
    ;; The conv biases before batch normalization have gradients that are
    ;; zero up to rounding, so the comparison skips parameters whose norm
    ;; is below 1e-6 times the global norm.
    (let* ((norms (lambda (st) (map (lambda (ls) (cdr (assq 'l2-norm ls)))
                                    (cdr (assq 'layer-stats st)))))
           (global (lambda (st) (cdr (assq 'global-l2-norm st))))
           (pairs (filter (lambda (p) (> (car p) (* 1e-6 (global strict-stats))))
                          (map cons (norms strict-stats) (norms lazy-stats)))))
      (printf "Global gradient L2 norm: strict ~A, lazy ~A\n"
              (global strict-stats) (global lazy-stats))
      (printf "Largest relative difference of the per-parameter gradient norms (~A parameters): ~A\n"
              (length pairs)
              (fold max 0.0 (map (lambda (p) (/ (abs (- (car p) (cdr p))) (car p))) pairs))))))

;;; ==================================================================
;;; Example 4: Compiled inference
;;; ==================================================================

(define (example-inference-speed)
  (printf "\n=== Example 4: Compiled Inference ===\n")

  (define num-classes 10)
  (define stages '(2 2 2 2))
  (define widths '(16 32 64 128))
  (define strict (make-resnet-cifar stages widths num-classes dtype: 'f32))
  (define model (make-am-resnet-18-narrow num-classes dtype: 'f32))
  (am-copy-parameters! strict model)
  (set-training-mode! strict #t)

  (printf "ResNet-18 layout, widths ~A, ~A parameters, backend ~A\n"
          widths (am-parameter-count model) am-backend)

  (define batch (make-dataset (iota num-classes) 2 0.2))
  (define n (length batch))
  (define evaluate (make-evaluator model num-classes))

  (let ((t0 (now-ms)))
    (evaluate batch)
    (printf "Lazy forward, batch ~A, first call (trace + compile): ~A ms\n" n (- (now-ms) t0)))
  (let ((t0 (now-ms)))
    (do ((i 0 (+ i 1))) ((= i 5)) (evaluate batch))
    (printf "Lazy forward, replay: ~A ms per batch\n" (/ (- (now-ms) t0) 5.0)))
  (let-values (((x t) (batch->vectors batch num-classes)))
    (let ((t0 (now-ms)))
      (forward strict (make-tensor32 x (list n channels img-size img-size) requires-grad?: #f))
      (printf "Strict forward: ~A ms per batch\n" (- (now-ms) t0)))))

;;; ==================================================================
;;; Example 5: Transfer learning
;;; ==================================================================

(define (copy-backbone! src dst)
  "Copy all parameters of SRC into DST except the dense head (the last
   two parameters, weight and bias).  A lazy parameter's tensor-data is
   its storage, so the copy changes the model in place."
  (let* ((ps (parameters src))
         (pd (parameters dst))
         (n (- (length ps) 2)))
    (for-each (lambda (s d)
                (let ((sv (tensor-data s)) (dv (tensor-data d)))
                  (do ((i 0 (+ i 1))) ((= i (f32vector-length sv)))
                    (f32vector-set! dv i (f32vector-ref sv i)))))
              (take ps n) (take pd n))
    n))

(define (example-transfer-learning)
  (printf "\n=== Example 5: Transfer Learning ===\n")

  ;; Source task: classes 0-4.  Target task: classes 5-9, with little data.
  (define stages '(1 1 1))
  (define widths '(8 16 32))
  (define source-data (make-dataset '(0 1 2 3 4) 16 0.2))
  (define target-train (make-dataset '(5 6 7 8 9) 4 0.2))
  (define target-test (make-dataset '(5 6 7 8 9) 6 0.2))

  (printf "Source task: classes 0-4, ~A images\n" (length source-data))
  (printf "Target task: classes 5-9, ~A training images\n" (length target-train))

  (define source (make-am-resnet stages widths 5 dtype: 'f32))
  (define source-trainer
    (make-trainer source (make-adam (parameters source) learning-rate: 0.01) 5))
  (printf "\nTraining on the source task...\n")
  (do ((epoch 1 (+ epoch 1))) ((> epoch 4))
    (printf "  Epoch ~A: loss=~A\n" epoch (train-epoch! source-trainer source-data 16)))

  (define (fine-tune label pretrained?)
    (let* ((model (make-am-resnet stages widths 5 dtype: 'f32))
           (copied (if pretrained? (copy-backbone! source model) 0))
           (trainer (make-trainer model (make-adam (parameters model) learning-rate: 0.005)
                                  5 label-offset: 5)))
      (do ((epoch 1 (+ epoch 1))) ((> epoch 4))
        (train-epoch! trainer target-train 10))
      (printf "  ~A (~A parameter tensors copied): test acc=~A%\n"
              label copied
              (* 100.0 (accuracy (make-evaluator model 5 label-offset: 5)
                                 target-test 5 label-offset: 5)))))

  (printf "\nFine-tuning on the target task for 4 epochs:\n")
  (fine-tune "From scratch   " #f)
  (fine-tune "From pretrained" #t))

;;; ==================================================================
;;; Run all examples
;;; ==================================================================

(define (run-all-am-resnet-examples)
  (printf "\n")
  (printf "========================================\n")
  (printf "LAZY (ARRAY-MORPHISMS) RESNET EXAMPLES\n")
  (printf "backend: ~A\n" am-backend)
  (printf "========================================\n")

  (set-pseudo-random-seed! (string->blob "resnet42"))

  (example-classification)
  (example-lr-schedule)
  (example-strict-agreement)
  (example-inference-speed)
  (example-transfer-learning)

  (printf "\n")
  (printf "========================================\n")
  (printf "ALL EXAMPLES COMPLETE\n")
  (printf "========================================\n\n"))

(run-all-am-resnet-examples)
