;; MLP-Mixer usage examples for Nanograd.
;;
;; 1. Forward pass and mean pooling over tokens
;; 2. Recursive refinement in the style of TRM: many cheap iterations
;;    whose graph is cut (detached), followed by a few differentiable ones
;; 3. Deep supervision: a loss at every refinement step
;; 4. Parameter count
;; 5. A training loop with Adam
;;
;; Examples 2, 3 and 5 use the gradient diagnostics of
;; nanograd-diagnostics (see gradient-monitoring.scm): a health check of
;; the gradients after backward!, the built-in gradient report, and a
;; gradient monitor in the training loop.
;;
;; Run with: csi -s examples/mlp-mixer-examples.scm

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        yasos
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-mlp-mixer)

(include-relative "gradient-monitoring.scm")

(define seq-len 16)     ; a 4x4 grid of patches
(define hidden-dim 32)

(define (filled-tensor value #!key (requires-grad? #t))
  (make-tensor32 (make-f32vector (* seq-len hidden-dim) value)
                 (list seq-len hidden-dim)
                 requires-grad?: requires-grad?))

(define (detach t)
  "A new tensor that shares the data of T but has no graph history:
   gradients stop here."
  (make-base-tensor (tensor-data t) (tensor-shape t) (tensor-dtype t) #t))

(define (iterate f x n #!key (detach? #f))
  "Apply F to X N times.  With DETACH?, every result is detached, so no
   graph is kept between iterations."
  (let loop ((i 0) (x x))
    (if (= i n)
        x
        (let ((y (f x)))
          (loop (+ i 1) (if detach? (detach y) y))))))

(define (scalar t) (f32vector-ref (tensor-data t) 0))

(printf "MLP-Mixer usage examples~%~%")

;;; ==================================================================
;;; Example 1: forward pass and pooling
;;; ==================================================================

(printf "=== Example 1: forward pass ===~%")
(let* ((mixer (make-mlp-mixer seq-len hidden-dim 2))
       (x (filled-tensor 0.01))
       (features (forward mixer x))
       (pooled (mean-tensor features)))
  (printf "Input shape:    ~A~%" (tensor-shape x))
  (printf "Features shape: ~A~%" (tensor-shape features))
  (printf "Pooled value:   ~A~%~%" (scalar pooled)))

;;; ==================================================================
;;; Example 2: recursive refinement with detached iterations
;;; ==================================================================

(printf "=== Example 2: recursive refinement ===~%")
(let* ((mixer (make-mlp-mixer seq-len hidden-dim 2 name: "TRM-Mixer"))
       (x (filled-tensor 0.1 requires-grad?: #f))
       (step (lambda (z) (forward mixer (add x z))))
       (z-explored (iterate step (filled-tensor 0.0) 6 detach?: #t))
       (z-final (iterate step z-explored 2))
       (loss (mean-tensor (mul z-final z-final))))
  (backward! loss)
  (printf "6 detached iterations, then 2 with gradients~%")
  (printf "Loss: ~A~%" (scalar loss))
  ;; The detached iterations contribute no gradient, but the last two
  ;; reach every parameter.  check-gradient-health prints a warning for
  ;; each problem it finds and returns them as a list.
  (let* ((params (parameters mixer))
         (health (check-gradient-health params warn?: #t))
         (with-grad (count (lambda (p) (let ((g (tensor-grad p)))
                                         (and g (> (gradient-l2-norm g (tensor-dtype p)) 0))))
                           params)))
    (printf "Parameters with a nonzero gradient: ~A of ~A~%" with-grad (length params))
    (printf "Gradient health: ~A~%"
            (if (cdr (assq 'healthy? health)) "healthy" (cdr (assq 'issues health))))
    (print-layer-gradients params "after 2 differentiable iterations")
    (printf "~%")))

;;; ==================================================================
;;; Example 3: deep supervision
;;; ==================================================================

(printf "=== Example 3: deep supervision ===~%")
(let* ((mixer (make-mlp-mixer seq-len hidden-dim 2))
       (x (filled-tensor 0.1 requires-grad?: #f))
       (target (filled-tensor 1.0 requires-grad?: #f))
       (states (let loop ((i 0) (z (filled-tensor 0.0)) (acc '()))
                 (if (= i 4)
                     (reverse acc)
                     (let ((z2 (forward mixer (add x z))))
                       (loop (+ i 1) z2 (cons z2 acc))))))
       (losses (map (lambda (z) (mse-loss z target)) states))
       (total (fold (lambda (l acc) (add acc l)) (car losses) (cdr losses))))
  (backward! total)
  (printf "Losses at each step: ~A~%" (map scalar losses))
  (printf "Total loss: ~A~%" (scalar total))
  ;; The built-in report of nanograd-diagnostics: overall status, global
  ;; norms and one line per parameter
  (print-gradient-report (parameters mixer) "deep supervision, 4 steps")
  (printf "~%"))

;;; ==================================================================
;;; Example 4: parameter count
;;; ==================================================================

(printf "=== Example 4: parameter count ===~%")
(let* ((mixer (make-mlp-mixer seq-len hidden-dim 2))
       (count (apply + (map (lambda (p) (apply * (tensor-shape p)))
                            (parameters mixer)))))
  (printf "~A tokens, ~A features, 2 blocks: ~A parameters~%~%"
          seq-len hidden-dim count))

;;; ==================================================================
;;; Example 5: training loop with Adam
;;; ==================================================================

(printf "=== Example 5: training loop ===~%")
(let* ((model (make-mlp-mixer seq-len hidden-dim 2))
       (optimizer (make-adam (parameters model) learning-rate: 0.001))
       (input (make-tensor32
               (let ((v (make-f32vector (* seq-len hidden-dim))))
                 (do ((i 0 (+ i 1))) ((= i (f32vector-length v)) v)
                   (f32vector-set! v i (* 0.01 (exact->inexact (modulo i 7))))))
               (list seq-len hidden-dim) requires-grad?: #f))
       (target (filled-tensor 0.5 requires-grad?: #f))
       ;; The optimizer is wrapped so that each step records the
       ;; gradients in a monitor before the Adam update
       (monitor (make-gradient-monitor history-size: 1000))
       (monitored (make-monitored-optimizer optimizer (parameters model) monitor)))
  (do ((epoch 1 (+ epoch 1))) ((> epoch 30))
    (let ((loss (mse-loss (forward model input) target)))
      (backward! loss)
      (step! monitored)
      (zero-grad-layer! model)
      (when (or (= epoch 1) (zero? (modulo epoch 5)))
        (printf "  epoch ~A: loss ~A, gradient norm ~A~%" epoch (scalar loss)
                (fmt-norm (recent-mean-norm monitor 1))))))
  (print-norm-trace monitor buckets: 3)
  (print-training-diagnosis monitored "MLP-Mixer training loop")
  (printf "~%"))
