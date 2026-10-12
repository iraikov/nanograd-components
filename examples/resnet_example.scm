;; Examples of ResNet training and evaluation with the strict
;; (eagerly evaluated) nanograd tensors.
;;
;; 1. Classification of synthetic striped images with a small
;;    CIFAR-style ResNet, trained on mini-batches
;; 2. Training with a step learning-rate schedule
;; 3. Parameter counts of ResNet-18, 34 and 50, and a forward pass
;; 4. Output statistics of ResNet-18 for a few images
;; 5. Transfer learning: reuse a trained backbone for new classes
;;
;; The training examples use the gradient diagnostics of
;; nanograd-diagnostics through gradient-monitoring.scm: the optimizer is
;; wrapped so that each step records the gradient norms in a monitor
;; and flags exploding, vanishing or unbalanced gradients.  Example 1
;; prints per-parameter gradient statistics and a diagnosis of the run,
;; and example 2 follows the gradient norm through a learning-rate
;; schedule.
;;
;; The first check finds parameters that do not learn: the bias of every
;; convolution that is followed by batch normalization.  Batch
;; normalization subtracts the per-channel mean, which cancels a
;; per-channel bias, so the gradient of that bias is zero up to rounding.
;; These parameters are reported and left out of the monitor, since
;; their tiny norms would otherwise flag every step as unbalanced.
;;
;; The ImageNet-style models (make-resnet-18 and larger) are only built
;; and run forward here: training them one image at a time would take
;; hours.  Training uses make-resnet-cifar, which has a 3x3 stem, no
;; max pooling and configurable widths.
;;
;; Run with: csi -s examples/resnet_example.scm
;; The array-morphisms version is examples/am_resnet_example.scm.

(import scheme
        (chicken base)
        (chicken blob)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-resnet)

(include-relative "gradient-monitoring.scm")

;;; ==================================================================
;;; Data and utilities
;;; ==================================================================

(define img-size 16)
(define channels 3)

(define (make-image label noise)
  "A CHANNELS x IMG-SIZE x IMG-SIZE image (NCHW element order) of
   stripes whose orientation and frequency depend on LABEL, with
   uniform noise of amplitude NOISE."
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

(define (batch->tensors batch num-classes #!key (label-offset 0))
  "The input tensor (N, C, H, W) and one-hot target tensor (N, K) of a
   list of (image-data . label) pairs; labels are shifted down by
   LABEL-OFFSET."
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
    (values (make-tensor32 x (list n channels img-size img-size) requires-grad?: #f)
            (make-tensor32 t (list n num-classes) requires-grad?: #f))))

(define (list-shuffle lst)
  "Fisher-Yates shuffle."
  (let ((vec (list->vector lst)))
    (do ((i (- (vector-length vec) 1) (- i 1)))
        ((< i 1) (vector->list vec))
      (let* ((j (pseudo-random-integer (+ i 1)))
             (temp (vector-ref vec i)))
        (vector-set! vec i (vector-ref vec j))
        (vector-set! vec j temp)))))

(define (batches lst size)
  "Split LST into consecutive lists of SIZE elements (the last may be shorter)."
  (if (null? lst)
      '()
      (let ((k (min size (length lst))))
        (cons (take lst k) (batches (drop lst k) size)))))

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

(define (count-params model)
  (fold (lambda (p acc) (+ acc (f32vector-length (tensor-data p))))
        0 (parameters model)))

(define (scalar t) (f32vector-ref (tensor-data t) 0))

(define (pad-right s n)
  (let ((s (if (string? s) s (sprintf "~A" s))))
    (if (>= (string-length s) n) s (string-append s (make-string (- n (string-length s)) #\space)))))

(define (pad-left s n)
  (let ((s (if (string? s) s (sprintf "~A" s))))
    (if (>= (string-length s) n) s (string-append (make-string (- n (string-length s)) #\space) s))))

(define (train-epoch! model optimizer data num-classes batch-size #!key (label-offset 0))
  "One pass over DATA in shuffled mini-batches.  Returns the mean loss
   and the training accuracy."
  (let loop ((bs (batches (list-shuffle data) batch-size))
             (total-loss 0.0) (correct 0) (seen 0))
    (if (null? bs)
        (values (/ total-loss (max 1 seen)) (/ correct (max 1 seen)))
        (let*-values (((x t) (batch->tensors (car bs) num-classes label-offset: label-offset))
                      ((n) (length (car bs)))
                      ((logits) (forward model x))
                      ((loss) (cross-entropy-loss logits t from-logits: #t)))
          (backward! loss)
          (step! optimizer)
          (zero-grad-layer! model)
          (loop (cdr bs)
                (+ total-loss (* n (scalar loss)))
                (+ correct (count (lambda (b sample)
                                    (= (argmax-row (tensor-data logits) b num-classes)
                                       (- (cdr sample) label-offset)))
                                  (iota n) (car bs)))
                (+ seen n))))))

(define (accuracy model data num-classes #!key (label-offset 0))
  "Fraction of DATA classified correctly, evaluated as one batch."
  (let-values (((x t) (batch->tensors data num-classes label-offset: label-offset)))
    (let ((logits (tensor-data (forward model x))))
      (/ (count (lambda (b sample)
                  (= (argmax-row logits b num-classes) (- (cdr sample) label-offset)))
                (iota (length data)) data)
         (length data)))))

;;; ==================================================================
;;; Example 1: Synthetic image classification
;;; ==================================================================

(define (example-classification)
  (printf "\n=== Example 1: Synthetic Image Classification ===\n")

  (define num-classes 10)
  (define train-data (make-dataset (iota num-classes) 12 0.2))
  (define test-data (make-dataset (iota num-classes) 4 0.2))

  (printf "~A training and ~A test images of size ~Ax~Ax~A\n"
          (length train-data) (length test-data) channels img-size img-size)

  ;; Three stages of one basic block, widths 8/16/32
  (define model (make-resnet-cifar '(1 1 1) '(8 16 32) num-classes dtype: 'f32))
  (printf "CIFAR-style ResNet with ~A parameters\n" (count-params model))

  ;; The monitor flags a step whose global gradient norm is above 10 or
  ;; below 1e-6, or whose largest and smallest per-parameter norms differ
  ;; by more than 1000x.  Parameters with no gradient (the convolution
  ;; biases before batch normalization) are excluded at the first step.  The per-parameter table is printed at the first
  ;; step and at the last of the 48 steps (8 batches per epoch).
  (define monitor (make-gradient-monitor exploding-threshold: 10.0
                                         vanishing-threshold: 1e-6
                                         history-size: 1000))
  (define optimizer
    (make-monitored-optimizer (make-adam (parameters model) learning-rate: 0.01)
                              (parameters model) monitor
                              layer-report: '(1 48) exclude-inactive?: #t))

  (printf "\nTraining...\n")
  (set-training-mode! model #t)
  (do ((epoch 1 (+ epoch 1))) ((> epoch 6))
    (let-values (((loss acc) (train-epoch! model optimizer train-data num-classes 16)))
      (printf "Epoch ~A: loss=~A, train acc=~A%, mean gradient norm=~A\n"
              epoch loss (* 100.0 acc) (fmt-norm (recent-mean-norm monitor 8)))))
  (print-norm-trace monitor)
  (print-training-diagnosis optimizer "classification")

  ;; Batch statistics in training mode, running statistics in eval mode
  (printf "\nTest accuracy (batch statistics): ~A%\n"
          (* 100.0 (accuracy model test-data num-classes)))
  (set-eval-mode! model)
  (printf "Test accuracy (running statistics): ~A%\n"
          (* 100.0 (accuracy model test-data num-classes)))

  (printf "\nPredictions for one image per class:\n")
  (let* ((samples (make-dataset (iota num-classes) 1 0.2)))
    (let-values (((x t) (batch->tensors samples num-classes)))
      (let ((logits (tensor-data (forward model x))))
        (for-each (lambda (sample b)
                    (let ((pred (argmax-row logits b num-classes)))
                      (printf "  True: ~A, Pred: ~A ~A\n" (cdr sample) pred
                              (if (= pred (cdr sample)) "(correct)" "(wrong)"))))
                  samples (iota num-classes)))))
  model)

;;; ==================================================================
;;; Example 2: Learning-rate schedule
;;; ==================================================================

(define (example-lr-schedule)
  (printf "\n=== Example 2: Training with a Learning-Rate Schedule ===\n")

  (define num-classes 4)
  (define model (make-resnet-cifar '(1 1) '(8 16) num-classes dtype: 'f32))
  (define monitor (make-gradient-monitor history-size: 1000))
  (define optimizer
    (make-monitored-optimizer (make-adam (parameters model) learning-rate: 0.02)
                              (parameters model) monitor exclude-inactive?: #t))

  ;; Step schedule: divide the rate by 10 at epochs 4 and 7
  (define (get-lr epoch)
    (cond ((< epoch 4) 0.02)
          ((< epoch 7) 0.002)
          (else 0.0002)))

  (printf "Each epoch draws 4 fresh batches of 16 noisy images\n")
  (set-training-mode! model #t)
  (do ((epoch 1 (+ epoch 1))) ((> epoch 8))
    (set-learning-rate! optimizer (get-lr epoch))
    (let-values (((loss acc)
                  (train-epoch! model optimizer
                                (make-dataset (iota num-classes) 16 0.6)
                                num-classes 16)))
      (printf "Epoch ~A: LR=~A, loss=~A, acc=~A%, mean gradient norm=~A\n"
              epoch (get-learning-rate optimizer) loss (* 100.0 acc)
              (fmt-norm (recent-mean-norm monitor 4)))))
  (printf "Training complete!\n")

  ;; suggest-learning-rate maps a gradient norm to a rate adjustment: it
  ;; suggests a smaller rate above a norm of 5 and a larger one below
  ;; 1e-4.
  (printf "\nLearning-rate check from the mean gradient norm of the last epoch:\n")
  (suggest-learning-rate (parameters model) (get-learning-rate optimizer)
                         (recent-mean-norm monitor 4)))

;;; ==================================================================
;;; Example 3: ResNet architecture comparison
;;; ==================================================================

(define (example-architecture-comparison)
  (printf "\n=== Example 3: Architecture Comparison ===\n")

  (define num-classes 10)

  (printf "\nResNet architectures (~A classes):\n" num-classes)
  (printf "~A~A~A\n" (pad-right "Model" 15) (pad-left "Parameters" 15) (pad-left "Depth" 10))
  (printf "~A~A~A\n" (pad-right "-----" 15) (pad-left "----------" 15) (pad-left "-----" 10))
  (for-each
   (lambda (name make depth)
     (printf "~A~A~A\n" (pad-right name 15)
             (pad-left (count-params (make num-classes)) 15)
             (pad-left depth 10)))
   '("ResNet-18" "ResNet-34" "ResNet-50")
   (list make-resnet-18 make-resnet-34 make-resnet-50)
   '(18 34 50))

  (printf "\nTesting forward pass (64x64 input)...\n")
  (let ((model (make-resnet-18 num-classes))
        (test-image (make-tensor32 (make-f32vector (* 3 64 64) 0.5) (list 3 64 64))))
    (set-eval-mode! model)
    (let* ((output (forward model test-image))
           (data (tensor-data output)))
      (printf "Input shape: ~A\n" (tensor-shape test-image))
      (printf "Output shape: ~A\n" (tensor-shape output))
      (printf "Output range: [~A, ~A]\n"
              (f32vector-fold min +inf.0 data)
              (f32vector-fold max -inf.0 data)))))

;;; ==================================================================
;;; Example 4: Output statistics
;;; ==================================================================

(define (example-output-statistics)
  (printf "\n=== Example 4: Output Statistics of ResNet-18 ===\n")

  (define num-classes 10)
  (define model (make-resnet-18 num-classes dtype: 'f32))

  ;; Five images of different constant brightness plus a stripe pattern
  (define test-images
    (map (lambda (i)
           (let ((data (make-f32vector (* 3 64 64))))
             (do ((k 0 (+ k 1))) ((= k (f32vector-length data)))
               (f32vector-set! data k (+ 0.5 (* 0.2 (sin (* i 0.5)))
                                         (* 0.1 (sin (* 0.3 k))))))
             (make-tensor32 data (list 3 64 64))))
         (iota 5)))

  (set-eval-mode! model)
  (printf "Untrained model, ~A images of size 64x64:\n" (length test-images))
  (for-each
   (lambda (img i)
     (let* ((data (tensor-data (forward model img)))
            (n (f32vector-length data))
            (mean (/ (f32vector-fold + 0.0 data) n))
            (var (/ (f32vector-fold (lambda (x acc) (+ acc (expt (- x mean) 2))) 0.0 data) n)))
       (printf "  Image ~A: ~A outputs, mean=~A, std=~A\n" i n mean (sqrt var))))
   test-images
   (iota (length test-images))))

;;; ==================================================================
;;; Example 5: Transfer learning
;;; ==================================================================

(define (copy-backbone! src dst)
  "Copy all parameters of SRC into DST except the final dense layer
   (the last two parameters, weight and bias).  The two models must
   have the same architecture apart from the number of classes."
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
  (define target-train (make-dataset '(5 6 7 8 9) 3 0.2))
  (define target-test (make-dataset '(5 6 7 8 9) 6 0.2))

  (printf "Source task: classes 0-4, ~A images\n" (length source-data))
  (printf "Target task: classes 5-9, ~A training images\n" (length target-train))

  (define source (make-resnet-cifar stages widths 5 dtype: 'f32))
  (define source-opt (make-adam (parameters source) learning-rate: 0.01))
  (set-training-mode! source #t)
  (printf "\nTraining on the source task...\n")
  (do ((epoch 1 (+ epoch 1))) ((> epoch 4))
    (let-values (((loss acc) (train-epoch! source source-opt source-data 5 16)))
      (printf "  Epoch ~A: loss=~A, acc=~A%\n" epoch loss (* 100.0 acc))))

  (define (fine-tune label pretrained?)
    (let* ((model (make-resnet-cifar stages widths 5 dtype: 'f32))
           (copied (if pretrained? (copy-backbone! source model) 0))
           (opt (make-adam (parameters model) learning-rate: 0.005)))
      (set-training-mode! model #t)
      (do ((epoch 1 (+ epoch 1))) ((> epoch 4))
        (train-epoch! model opt target-train 5 8 label-offset: 5))
      (printf "  ~A (~A parameter tensors copied): test acc=~A%\n"
              label copied
              (* 100.0 (accuracy model target-test 5 label-offset: 5)))))

  (printf "\nFine-tuning on the target task for 4 epochs:\n")
  (fine-tune "From scratch   " #f)
  (fine-tune "From pretrained" #t))

;;; ==================================================================
;;; Run all examples
;;; ==================================================================

(define (run-all-resnet-examples)
  (printf "\n")
  (printf "========================================\n")
  (printf "RESNET EXAMPLES\n")
  (printf "========================================\n")

  (set-pseudo-random-seed! (string->blob "resnet42"))

  (example-classification)
  (example-lr-schedule)
  (example-architecture-comparison)
  (example-output-statistics)
  (example-transfer-learning)

  (printf "\n")
  (printf "========================================\n")
  (printf "ALL EXAMPLES COMPLETE\n")
  (printf "========================================\n\n"))

(run-all-resnet-examples)
