;; Tests for the lazy (array-morphisms) models.
;;
;; Each lazy model is checked against its strict counterpart: the
;; strict model's weights are copied into the lazy model, both take one
;; SGD step (learning rate 1) on the same batch, and the losses and the
;; parameter updates, which equal the gradients, must agree.  The lazy
;; step goes through am-training-step/ssa, so this also checks the
;; compiled forward and backward programs.  All models use f64.

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        test
        yasos
        array-morphisms-core
        array-morphisms-context
        (prefix array-morphisms-grad am:)
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-array-morphisms
        nanograd-resnet
        nanograd-mlp-mixer
        nanograd-transformer
        nanograd-am-common
        nanograd-am-resnet
        nanograd-am-mlp-mixer
        nanograd-am-transformer)

(define tol 1e-12)

(define (smooth-vector n f)
  (let ((v (make-f64vector n)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f64vector-set! v i (f i)))))

(define (lazy-input data shape)
  (get-or-make-lazy (am:make-var (make-morphism data shape 'f64) #f)))

(define (snapshot model)
  (map (lambda (p) (f64vector->list (tensor-data p))) (parameters model)))

(define (max-abs xs) (fold max 0.0 (map abs xs)))

;; STRICT-LOSS computes the strict loss tensor; LAZY-STEP runs one
;; am-training-step/ssa with the given optimizer and returns the loss.
;; Returns (loss-difference update-difference largest-update).
(define (compare-one-step strict lz strict-loss lazy-step)
  (am-copy-parameters! strict lz)
  (let* ((s0 (snapshot strict))
         (l0 (snapshot lz))
         (s-opt (make-sgd (parameters strict) learning-rate: 1.0))
         (l-opt (make-sgd (parameters lz) learning-rate: 1.0))
         (s-loss (let ((l (strict-loss)))
                   (backward! l)
                   (step! s-opt)
                   (f64vector-ref (tensor-data l) 0)))
         (l-loss (f64vector-ref (tensor-data (lazy-step l-opt)) 0))
         (s-upd (append-map (lambda (a b) (map - a b)) s0 (snapshot strict)))
         (l-upd (append-map (lambda (a b) (map - a b)) l0 (snapshot lz))))
    (list (abs (- s-loss l-loss))
          (max-abs (map - s-upd l-upd))
          (max-abs s-upd))))

(define (check-agreement name result)
  (let ((loss-diff (car result))
        (update-diff (cadr result))
        (largest (caddr result)))
    (printf "  ~A: loss diff ~A, update diff ~A (largest update ~A)~%"
            name loss-diff update-diff largest)
    (test-assert (string-append name ": same loss") (< loss-diff tol))
    (test-assert (string-append name ": same gradients") (< update-diff tol))
    (test-assert (string-append name ": gradients are not zero") (> largest 1e-6))))

;;; ==================================================================

(test-group "MLP-Mixer, lazy vs strict"
  (let* ((T 4) (D 6) (B 3)
         (xs (map (lambda (b) (smooth-vector (* T D) (lambda (i) (sin (+ (* 0.7 i) b)))))
                  (iota B)))
         (ts (map (lambda (b) (smooth-vector (* T D) (lambda (i) (* 0.5 (cos (+ (* 0.3 i) b))))))
                  (iota B))))
    (for-each
     (lambda (rms?)
       (let* ((strict (make-mlp-mixer T D 2 token-expansion: 2 channel-expansion: 2
                                      use-rmsnorm: rms? dtype: 'f64))
              (lz (make-am-mlp-mixer T D 2 token-expansion: 2 channel-expansion: 2
                                     use-rmsnorm: rms? dtype: 'f64))
              (x-lt (lazy-input (am-mixer-pack-batch xs T D 'f64) (list (* T B) D)))
              (t-lt (lazy-input (am-mixer-pack-batch ts T D 'f64) (list (* T B) D))))
         (check-agreement
          (if rms? "RMSNorm mixer" "LayerNorm mixer")
          (compare-one-step
           strict lz
           (lambda ()
             (let ((ls (map (lambda (x t)
                              (mse-loss (forward strict (make-tensor64 x (list T D)
                                                                       requires-grad?: #f))
                                        (make-tensor64 t (list T D) requires-grad?: #f)))
                            xs ts)))
               (scale-op (fold (lambda (l acc) (add acc l)) (car ls) (cdr ls)) (/ 1.0 B))))
           (lambda (opt)
             (am-training-step/ssa (make-morphism-context) opt lz
                                   (lambda (out) (am-half-mse-loss out t-lt))
                                   x-lt t-lt))))))
     '(#t #f))))

(test-group "ResNet, lazy vs strict"
  (let* ((N 2) (C 2) (H 6) (W 6) (K 3)
         (xd (smooth-vector (* N C H W) (lambda (i) (sin (* 0.37 i)))))
         (onehot (f64vector 1 0 0 0 0 1))
         (strict (make-resnet-cifar '(1 1) '(3 4) K in-channels: C dtype: 'f64))
         (lz (make-am-resnet '(1 1) '(3 4) K in-channels: C dtype: 'f64))
         (t-lt (lazy-input onehot (list N K))))
    (set-training-mode! strict #t)
    (check-agreement
     "two-stage CIFAR ResNet"
     (compare-one-step
      strict lz
      (lambda ()
        (cross-entropy-loss (forward strict (make-tensor64 xd (list N C H W) requires-grad?: #f))
                            (make-tensor64 onehot (list N K) requires-grad?: #f)
                            from-logits: #t))
      (lambda (opt)
        (am-training-step/ssa (make-morphism-context) opt lz
                              (lambda (out) (am-cross-entropy-loss out t-lt))
                              (lazy-input (am-nchw->nhwc xd N C H W 'f64) (list N H W C))
                              t-lt))))))

(test-group "Transformer, lazy vs strict"
  (let* ((V 7) (T 5) (D 8) (B 3)
         (seqs '((1 2 3 4 5 6) (6 5 4 3 2 1) (0 3 0 3 6 2)))
         (io (am-next-token-targets seqs V 'f64)))
    (for-each
     (lambda (config)
       (let* ((causal? (car config))
              (rope? (cadr config))
              (strict (make-transformer V T D 2 2 d-ff: 16 causal?: causal? use-rope?: rope?
                                        dtype: 'f64))
              (lz (make-am-transformer V T D 2 2 d-ff: 16 causal?: causal? use-rope?: rope?
                                       dtype: 'f64))
              (x-lt (lazy-input (car io) (list (* B T) V)))
              (t-lt (lazy-input (cadr io) (list (* B T) V))))
         (check-agreement
          (sprintf "~A attention, ~A" (if causal? "causal" "bidirectional")
                   (if rope? "RoPE" "no RoPE"))
          (compare-one-step
           strict lz
           (lambda ()
             (let ((ls (map (lambda (s)
                              (cross-entropy-loss
                               (forward strict (take s T))
                               (make-tensor64 (list->f64vector (map exact->inexact (cdr s)))
                                              (list T) requires-grad?: #f)
                               from-logits: #t))
                            seqs)))
               (scale-op (fold (lambda (l acc) (add acc l)) (car ls) (cdr ls)) (/ 1.0 B))))
           (lambda (opt)
             (am-training-step/ssa (make-morphism-context) opt lz
                                   (lambda (out) (am-cross-entropy-loss out t-lt))
                                   x-lt t-lt))))))
     '((#t #t) (#f #f)))))

(test-group "Replay"
  ;; After the first (trace) step, later steps replay the compiled plan;
  ;; the loss must keep decreasing on a fixed batch.
  (let* ((V 7) (T 5) (B 3)
         (seqs '((1 2 3 4 5 6) (6 5 4 3 2 1) (0 3 0 3 6 2)))
         (io (am-next-token-targets seqs V 'f64))
         (lz (make-am-transformer V T 8 1 2 d-ff: 16 dtype: 'f64))
         (x-lt (lazy-input (car io) (list (* B T) V)))
         (t-lt (lazy-input (cadr io) (list (* B T) V)))
         (opt (make-adam (parameters lz) learning-rate: 0.01))
         (ctx (make-morphism-context))
         (losses (map (lambda (i)
                        (f64vector-ref
                         (tensor-data
                          (am-training-step/ssa ctx opt lz
                                                (lambda (out) (am-cross-entropy-loss out t-lt))
                                                x-lt t-lt))
                         0))
                      (iota 20))))
    (printf "  transformer losses: first ~A, last ~A~%" (car losses) (last losses))
    (test-assert "loss decreases over 20 replayed steps" (< (last losses) (* 0.9 (car losses))))))

(test-exit)
