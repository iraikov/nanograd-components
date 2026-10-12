;;; bench-am-resnet-crunch.scm -- ResNet training step on the lazy (SSA)
;;; path with the array-morphisms-crunch backend.
;;;
;;; The lazy model (nanograd-am-resnet) trains on an NHWC batch with
;;; softmax cross-entropy against one-hot labels.  The strict model
;;; (nanograd-resnet's make-resnet-cifar) with the same weights runs the
;;; same NCHW batch in training mode.  Both use a 3x3 stride-1 stem, no
;;; max pooling, batch statistics in batch normalization, global average
;;; pooling and a dense head.
;;;
;;;   config   stages     widths          input     batch  classes
;;;   small    (1 1 1)    (16 32 64)      32x32x3   8      10
;;;   medium   (2 2 2 2)  (16 32 64 128)  32x32x3   16     10
;;;
;;; See bench-am-common.scm for the environment variables.
;;; Compile with: csc -O3 bench-am-resnet-crunch.scm

(include-relative "bench-am-common.scm")
(import nanograd-resnet nanograd-am-resnet)

(define-values (stages widths N)
  (case bench-config
    ((small) (values '(1 1 1) '(16 32 64) 8))
    ((medium) (values '(2 2 2 2) '(16 32 64 128) 16))))

(define C 3) (define H 32) (define W 32) (define K 10)

(define strict (make-resnet-cifar stages widths K in-channels: C dtype: 'f32))
(define lazy-model (make-am-resnet stages widths K in-channels: C dtype: 'f32))
(set-training-mode! strict #t)
(am-copy-parameters! strict lazy-model)

(define x-nchw (random-f32vector (* N C H W)))
(define labels
  (let ((v (make-f32vector (* N K) 0.0)))
    (do ((i 0 (+ i 1))) ((= i N) v)
      (f32vector-set! v (+ (* i K) (pseudo-random-integer K)) 1.0))))

(define x-lt (vector->lazy (am-nchw->nhwc x-nchw N C H W 'f32) (list N H W C)))
(define t-lt (vector->lazy labels (list N K)))

(define lazy-opt (make-adam (parameters lazy-model) learning-rate: 0.001))
(define strict-opt (make-adam (parameters strict) learning-rate: 0.001))
(define ctx (make-morphism-context))

;; OPT defaults to lazy-opt; run-benchmark passes a monitored wrapper of
;; it to measure the cost of gradient monitoring.
(define (lazy-step #!optional (opt lazy-opt))
  (am-training-step/ssa ctx opt lazy-model
                        (lambda (logits) (am-cross-entropy-loss logits t-lt))
                        x-lt t-lt))

(define (strict-loss)
  (cross-entropy-loss (forward strict (make-tensor32 x-nchw (list N C H W) requires-grad?: #f))
                      (make-tensor32 labels (list N K) requires-grad?: #f)
                      from-logits: #t))

(define (strict-step)
  (let ((l (strict-loss)))
    (backward! l)
    (step! strict-opt)
    (zero-grad-layer! strict)
    l))

(run-benchmark "ResNet SSA training step"
               (list (sprintf "stages ~A, widths ~A, input ~Ax~Ax~A, batch ~A, ~A classes"
                              stages widths H W C N K))
               lazy-model lazy-opt lazy-step strict-loss strict-step)
