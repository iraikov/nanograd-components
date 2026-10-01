;; Finite-difference gradient checks for the component models, and a
;; check that KV-cache generation reproduces the full forward pass.
;;
;; For a model M and a fixed input, the scalar loss is
;;   L = sum(W * M(input))
;; for a fixed random tensor W of the output's shape.  After one
;; backward! pass, a few entries of every parameter tensor are
;; perturbed by +-h and the central difference is compared with the
;; analytic gradient.  All models use f64.

(import scheme
        (chicken base)
        (chicken blob)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        test
        yasos
        nanograd-autograd
        nanograd-layer
        nanograd-resnet
        nanograd-mlp-mixer
        nanograd-transformer)

(set-pseudo-random-seed! (string->blob "component-gradcheck"))

(define (random-f64vector n #!key (scale 1.0))
  (let ((v (make-f64vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f64vector-set! v i (* scale (- (* 2.0 (pseudo-random-real)) 1.0))))))

(define (random-indices n k)
  "K distinct indices below N (all of them when N <= K)."
  (if (<= n k)
      (iota n)
      (let loop ((acc '()))
        (if (= (length acc) k)
            acc
            (let ((i (pseudo-random-integer n)))
              (loop (if (memv i acc) acc (cons i acc))))))))

;; RUN maps the model to its output tensor.  Returns, per parameter
;; tensor, the largest relative error over the sampled entries:
;;   |numeric - analytic| / max(1e-6, |numeric| + |analytic|)
(define (parameter-gradient-errors model run #!key (h 1e-5) (samples 4))
  (let* ((out (run))
         (w (make-tensor64 (random-f64vector (apply * (tensor-shape out)))
                           (tensor-shape out) requires-grad?: #f))
         (loss-value (lambda ()
                       (f64vector-ref (tensor-data (sum-tensor (mul (run) w))) 0))))
    (zero-grad-layer! model)
    (backward! (sum-tensor (mul out w)))
    (map (lambda (p)
           (let* ((d (tensor-data p))
                  (g (tensor-grad p))
                  (idx (random-indices (f64vector-length d) samples)))
             (fold max 0.0
                   (map (lambda (i)
                          (let* ((a (f64vector-ref g i))
                                 (x0 (f64vector-ref d i))
                                 (lp (begin (f64vector-set! d i (+ x0 h)) (loss-value)))
                                 (lm (begin (f64vector-set! d i (- x0 h)) (loss-value)))
                                 (num (/ (- lp lm) (* 2.0 h))))
                            (f64vector-set! d i x0)
                            (/ (abs (- num a))
                               (max 1e-6 (+ (abs num) (abs a))))))
                        idx))))
         (parameters model))))

(define (checked-errors model run)
  (parameter-gradient-errors model run))

(define (all-below? errs tol) (every (lambda (e) (< e tol)) errs))

(define (report name errs)
  (printf "  ~A: ~A parameter tensors, max relative error ~A~%"
          name (length errs) (fold max 0.0 errs)))

;;; ==================================================================

(test-group "MLP-Mixer gradients"
  (let* ((model (make-mlp-mixer 4 6 1 token-expansion: 2 channel-expansion: 2
                                dtype: 'f64))
         (x (make-tensor64 (random-f64vector 24) '(4 6) requires-grad?: #f))
         (errs (checked-errors model (lambda () (forward model x)))))
    (report "RMSNorm mixer" errs)
    (test-assert "every parameter, including token mixing" (all-below? errs 1e-4)))
  (let* ((model (make-mlp-mixer 4 6 1 token-expansion: 2 channel-expansion: 2
                                use-rmsnorm: #f dtype: 'f64))
         (x (make-tensor64 (random-f64vector 24) '(4 6) requires-grad?: #f))
         (errs (checked-errors model (lambda () (forward model x)))))
    (report "LayerNorm mixer" errs)
    (test-assert "LayerNorm variant" (all-below? errs 1e-4))))

(test-group "Transformer gradients"
  (let* ((model (make-transformer 7 8 8 1 2 d-ff: 16 dtype: 'f64))
         (tokens '(3 1 4 1 5))
         (errs (checked-errors model (lambda () (forward model tokens)))))
    (report "causal transformer" errs)
    (test-assert "embedding, norms, WQ/WK/WV/WO and FFN" (all-below? errs 1e-4)))
  (let* ((model (make-transformer 7 8 8 1 2 d-ff: 16 causal?: #f use-rope?: #f
                                  dtype: 'f64))
         (tokens '(2 6 0 6))
         (errs (checked-errors model (lambda () (forward model tokens)))))
    (report "bidirectional transformer without RoPE" errs)
    (test-assert "bidirectional, no RoPE" (all-below? errs 1e-4))))

(test-group "ResNet gradients"
  (let* ((model (make-resnet-cifar '(1 1) '(3 4) 3 in-channels: 2 dtype: 'f64))
         (x (make-tensor64 (random-f64vector (* 2 2 6 6)) '(2 2 6 6)
                           requires-grad?: #f))
         (_ (set-training-mode! model #t))
         (errs (checked-errors model (lambda () (forward model x)))))
    (report "CIFAR ResNet, batch 2" errs)
    (test-assert "conv, batch norm, downsample and FC" (all-below? errs 1e-4))))

(test-group "KV cache"
  (let* ((model (make-transformer 9 16 8 2 2 d-ff: 16 dtype: 'f64))
         (tokens '(3 1 4 1 5 8))
         (full (tensor-data (forward model tokens)))
         (vocab 9)
         (last-row (lambda (data rows)
                     (map (lambda (j) (f64vector-ref data (+ (* (- rows 1) vocab) j)))
                          (iota vocab))))
         (cache (make-kv-cache 2 2 16 4 dtype: 'f64))
         (prompt (forward model (take tokens 3) kv-cache: cache))
         (steps (map (lambda (t) (f64vector->list
                                  (tensor-data (forward model (list t) kv-cache: cache))))
                     (drop tokens 3)))
         (close? (lambda (a b)
                   (every (lambda (x y) (< (abs (- x y)) 1e-9)) a b))))
    (test "cache length after generation" 6 (kv-cache-length cache))
    (test-assert "prompt logits match the full pass"
      (close? (f64vector->list (tensor-data prompt))
              (last-row (tensor-data (forward model (take tokens 3))) 3)))
    (test-assert "last generated logits match the full pass"
      (close? (last steps) (last-row full 6)))
    (clear-cache! cache)
    (test "clear-cache! resets the length" 0 (kv-cache-length cache))))

(test-exit)
