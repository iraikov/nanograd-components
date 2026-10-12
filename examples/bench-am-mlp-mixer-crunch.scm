;;; bench-am-mlp-mixer-crunch.scm -- MLP-Mixer training step on the lazy
;;; (SSA) path with the array-morphisms-crunch backend.
;;;
;;; The lazy model (nanograd-am-mlp-mixer) trains on a token-major batch
;;; of B samples, each T tokens by D features, with the 0.5 * MSE loss of
;;; nanograd's strict mse-loss against a random target.  The strict model
;;; (nanograd-mlp-mixer) with the same weights processes the samples one
;;; at a time; its loss is the mean over the batch.
;;;
;;;   config   T    D   blocks  B
;;;   small    16   64  2       8
;;;   medium   64  128  4       16
;;;
;;; See bench-am-common.scm for the environment variables.
;;; Compile with: csc -O3 bench-am-mlp-mixer-crunch.scm

(include-relative "bench-am-common.scm")
(import nanograd-mlp-mixer nanograd-am-mlp-mixer)

(define-values (T D n-blocks B)
  (case bench-config
    ((small) (values 16 64 2 8))
    ((medium) (values 64 128 4 16))))

(define strict (make-mlp-mixer T D n-blocks dtype: 'f32))
(define lazy-model (make-am-mlp-mixer T D n-blocks dtype: 'f32))
(am-copy-parameters! strict lazy-model)

(define xs (map (lambda (b) (random-f32vector (* T D))) (iota B)))
(define ts (map (lambda (b) (random-f32vector (* T D))) (iota B)))

(define x-lt (vector->lazy (am-mixer-pack-batch xs T D 'f32) (list (* T B) D)))
(define t-lt (vector->lazy (am-mixer-pack-batch ts T D 'f32) (list (* T B) D)))

(define lazy-opt (make-adam (parameters lazy-model) learning-rate: 0.001))
(define strict-opt (make-adam (parameters strict) learning-rate: 0.001))
(define ctx (make-morphism-context))

;; OPT defaults to lazy-opt; run-benchmark passes a monitored wrapper of
;; it to measure the cost of gradient monitoring.
(define (lazy-step #!optional (opt lazy-opt))
  (am-training-step/ssa ctx opt lazy-model
                        (lambda (out) (am-half-mse-loss out t-lt))
                        x-lt t-lt))

(define (strict-loss)
  (let ((ls (map (lambda (x t)
                   (mse-loss (forward strict (make-tensor32 x (list T D) requires-grad?: #f))
                             (make-tensor32 t (list T D) requires-grad?: #f)))
                 xs ts)))
    (scale-op (fold (lambda (l acc) (add acc l)) (car ls) (cdr ls)) (/ 1.0 B))))

(define (strict-step)
  (let ((l (strict-loss)))
    (backward! l)
    (step! strict-opt)
    (zero-grad-layer! strict)
    l))

(run-benchmark "MLP-Mixer SSA training step"
               (list (sprintf "~A tokens x ~A features, ~A blocks, batch ~A" T D n-blocks B))
               lazy-model lazy-opt lazy-step strict-loss strict-step)
