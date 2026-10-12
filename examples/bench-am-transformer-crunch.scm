;;; bench-am-transformer-crunch.scm -- Transformer language-model training
;;; step on the lazy (SSA) path with the array-morphisms-crunch backend.
;;;
;;; The lazy model (nanograd-am-transformer) trains on a batch of B token
;;; sequences with next-token cross-entropy; attention is causal and uses
;;; RoPE.  The strict model (nanograd-transformer) with the same weights
;;; processes the sequences one at a time; its loss is the mean over the
;;; batch.
;;;
;;;   config   vocab  T    d    layers  heads  d-ff  B
;;;   small    32     32   64   2       4      256   4
;;;   medium   64     64   128  4       8      512   4
;;;
;;; See bench-am-common.scm for the environment variables.
;;; Compile with: csc -O3 bench-am-transformer-crunch.scm

(include-relative "bench-am-common.scm")
(import nanograd-transformer nanograd-am-transformer)

(define-values (V T D n-layers n-heads B)
  (case bench-config
    ((small) (values 32 32 64 2 4 4))
    ((medium) (values 64 64 128 4 8 4))))

(define strict (make-transformer V T D n-layers n-heads dtype: 'f32))
(define lazy-model (make-am-transformer V T D n-layers n-heads dtype: 'f32))
(am-copy-parameters! strict lazy-model)

;; B random sequences of T+1 tokens: inputs are the first T, targets the last T.
(define sequences
  (map (lambda (b) (map (lambda (t) (pseudo-random-integer V)) (iota (+ T 1))))
       (iota B)))

(define io (am-next-token-targets sequences V 'f32))
(define x-lt (vector->lazy (car io) (list (* B T) V)))
(define t-lt (vector->lazy (cadr io) (list (* B T) V)))

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
  (let ((ls (map (lambda (s)
                   (cross-entropy-loss
                    (forward strict (take s T))
                    (make-tensor32 (list->f32vector (map exact->inexact (cdr s))) (list T)
                                   requires-grad?: #f)
                    from-logits: #t))
                 sequences)))
    (scale-op (fold (lambda (l acc) (add acc l)) (car ls) (cdr ls)) (/ 1.0 B))))

(define (strict-step)
  (let ((l (strict-loss)))
    (backward! l)
    (step! strict-opt)
    (zero-grad-layer! strict)
    l))

(run-benchmark "Transformer SSA training step"
               (list (sprintf "vocab ~A, context ~A, d ~A, ~A layers, ~A heads, batch ~A"
                              V T D n-layers n-heads B)
                     "causal attention with RoPE, tied output projection")
               lazy-model lazy-opt lazy-step strict-loss strict-step
               ;; The tied embedding gets gradients thousands of times
               ;; larger than the attention projections at initialization
               ;; (see transformer_example.scm).
               imbalance-threshold: 1e5)
