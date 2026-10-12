;;; gradient-monitoring.scm -- gradient diagnostics for the training examples
;;;
;;; Included by the training examples.  It builds on nanograd-diagnostics:
;;;
;;;   make-monitored-optimizer  wraps an optimizer so that every step!
;;;                             first records the gradients in a gradient
;;;                             monitor (record-step!), optionally clips
;;;                             them (clip-gradients!), and then applies the
;;;                             wrapped optimizer's update
;;;   print-layer-gradients     a per-parameter table from gradient-stats
;;;   print-norm-trace          the global gradient norm over the recorded
;;;                             steps, in a few step ranges
;;;   print-training-diagnosis  the summary of diagnose-training
;;   recent-mean-norm          the mean gradient norm of the last steps
;;;
;;; Wrapping the optimizer is what lets the diagnostics work with both
;;; tensor kinds.  With strict tensors the training loop calls
;;; backward! and then step!.  With lazy tensors, am-training-step/ssa
;;; runs the compiled program, installs the gradients in the parameters,
;;; calls step! and then clears the gradients, so step! is the only
;;; point where they can be read.  In both cases tensor-grad returns the
;;; gradient buffer itself, so clipping it in place changes the update.

(import yasos (chicken sort) (only (chicken string) string-intersperse) (only srfi-1 take take-right drop last count filter remove iota))
(import nanograd-diagnostics)

;;; ------------------------------------------------------------------
;;; Monitored optimizer
;;; ------------------------------------------------------------------

(define-operation (monitored-steps opt))
(define-operation (clipped-steps opt))
(define-operation (flagged-steps opt))
(define-operation (gradient-monitor-of opt))

;; INNER is the optimizer whose update is applied; PARAMS are its
;; parameters; MONITOR is a gradient monitor.  Keys:
;;   clip-norm      clip the global L2 norm to this value (#f: no clipping)
;;   layer-report   list of step numbers at which to print the
;;                  per-parameter table (print-layer-gradients)
;;   max-warnings   number of steps with health issues that are printed;
;;                  later ones are only counted
;;   exclude-inactive?  at the first step, find the parameters whose
;;                  gradient norm is below 1e-6 times the global norm,
;;                  report them, and leave them out of the monitor (so
;;                  they do not trigger the gradient-imbalance check);
;;                  clipping still covers all parameters
(define (make-monitored-optimizer inner params monitor
                                  #!key (clip-norm #f) (layer-report '()) (max-warnings 3)
                                  (exclude-inactive? #f))
  (let ((n 0) (clipped 0) (flagged 0) (monitored params))
    (object
     ((optimizer? self) #t)
     ((monitored-steps self) n)
     ((clipped-steps self) clipped)
     ((flagged-steps self) flagged)
     ((gradient-monitor-of self) monitor)
     ((get-learning-rate self) (get-learning-rate inner))
     ((set-learning-rate! self lr) (set-learning-rate! inner lr))
     ((optimizer-state self)
      (append (optimizer-state inner)
              `((monitored-steps . ,n) (clipped-steps . ,clipped))))
     ((step! self)
      (set! n (+ n 1))
      (when (and exclude-inactive? (= n 1))
        (set! monitored (report-inactive-parameters params)))
      ;; The monitor records the gradients before clipping
      (let* ((health (record-step! monitor n monitored))
             (issues (cdr (assq 'issues health))))
        (unless (null? issues)
          (set! flagged (+ flagged 1))
          (when (<= flagged max-warnings)
            (printf "    [gradients] step ~A: ~A (global L2 norm ~A)~A\n"
                    n issues (fmt-norm (global-l2 health))
                    (if (= flagged max-warnings) "; further warnings are only counted" "")))))
      (when (memv n layer-report)
        (print-layer-gradients params (sprintf "step ~A, before clipping" n)))
      (when (and clip-norm (clip-gradients! params max-norm: clip-norm verbose?: #f))
        (set! clipped (+ clipped 1)))
      (step! inner)))))

(define (global-l2 health)
  (cdr (assq 'global-l2-norm (cdr (assq 'stats health)))))

;; The mean global L2 norm of the last K steps recorded by MONITOR.
(define (recent-mean-norm monitor k)
  (let* ((h (get-gradient-history monitor))
         (recent (if (> (length h) k) (take-right h k) h)))
    (if (null? recent)
        0.0
        (/ (apply + (map (lambda (e) (cdr (assq 'global-l2-norm e))) recent))
           (length recent)))))

;; Prints the parameters of PARAMS whose gradient norm is below RELATIVE
;; times the global norm, and returns the other parameters.
(define (report-inactive-parameters params #!key (relative 1e-6))
  (let* ((global (compute-gradient-norm params))
         (inactive? (lambda (p)
                      (let ((g (tensor-grad p)))
                        (or (not g)
                            (<= (gradient-l2-norm g (tensor-dtype p)) (* relative global)))))))
    (let ((idle (filter (lambda (ip) (inactive? (cdr ip)))
                        (map cons (iota (length params)) params))))
      (unless (null? idle)
        (printf "    [gradients] ~A of ~A parameters have a gradient norm below ~A x the global norm:\n"
                (length idle) (length params) relative)
        (printf "      ~A\n"
                (string-intersperse
                 (map (lambda (ip) (sprintf "param ~A ~A" (car ip) (tensor-shape (cdr ip)))) idle)
                 ", "))
        (printf "      they are left out of the monitor\n"))
      (remove inactive? params))))

;;; ------------------------------------------------------------------
;;; Reports
;;; ------------------------------------------------------------------

;; A number rounded for display: 4 decimals from 1e-3 on, 3 significant
;; digits in scientific notation below.
(define (fmt-norm x)
  (cond ((not (finite? x)) x)
        ((or (zero? x) (>= (abs x) 1e-3)) (/ (round (* 1e4 x)) 1e4))
        (else
         (let* ((e (inexact->exact (floor (/ (log (abs x)) (log 10)))))
                (m (/ (round (* 100 (/ x (expt 10.0 e)))) 100)))
           (sprintf "~Ae~A" m e)))))

(define (gm-pad s n #!optional (left? #t))
  (let ((s (if (string? s) s (sprintf "~A" s))))
    (if (>= (string-length s) n)
        s
        (let ((fill (make-string (- n (string-length s)) #\space)))
          (if left? (string-append fill s) (string-append s fill))))))

;; A table of the per-parameter gradient statistics of PARAMS (which must
;; hold gradients), followed by the parameters with the largest and
;; smallest gradient norms and the ratio of the two.  At most MAX-ROWS
;; rows are printed; the summary lines cover all parameters.
(define (print-layer-gradients params label #!key (max-rows 12))
  (let* ((stats (gradient-stats params))
         (layers (cdr (assq 'layer-stats stats)))
         (shape-of (lambda (i) (tensor-shape (list-ref params i))))
         (l2 (lambda (s) (cdr (assq 'l2-norm s)))))
    (printf "\n    Per-parameter gradients (~A)\n" label)
    (printf "    ~A~A~A~A~A\n" (gm-pad "param" 7) (gm-pad "shape" 18) (gm-pad "L2" 12)
            (gm-pad "max |g|" 12) (gm-pad "mean |g|" 12))
    (for-each
     (lambda (s)
       (let ((i (cdr (assq 'layer s))))
         (printf "    ~A~A~A~A~A\n" (gm-pad i 7) (gm-pad (shape-of i) 18)
                 (gm-pad (fmt-norm (l2 s)) 12)
                 (gm-pad (fmt-norm (cdr (assq 'inf-norm s))) 12)
                 (gm-pad (fmt-norm (cdr (assq 'mean-abs s))) 12))))
     (if (> (length layers) max-rows) (take layers max-rows) layers))
    (when (> (length layers) max-rows)
      (printf "    ... ~A more parameters\n" (- (length layers) max-rows)))
    (unless (null? layers)
      (let* ((sorted (sort layers (lambda (a b) (> (l2 a) (l2 b)))))
             (hi (car sorted))
             (lo (last sorted)))
        (printf "    global L2 ~A, global max |g| ~A\n"
                (fmt-norm (cdr (assq 'global-l2-norm stats)))
                (fmt-norm (cdr (assq 'global-inf-norm stats))))
        (printf "    largest: param ~A ~A (L2 ~A); smallest: param ~A ~A (L2 ~A)\n"
                (cdr (assq 'layer hi)) (shape-of (cdr (assq 'layer hi))) (fmt-norm (l2 hi))
                (cdr (assq 'layer lo)) (shape-of (cdr (assq 'layer lo))) (fmt-norm (l2 lo)))
        (when (> (l2 lo) 0)
          (printf "    largest/smallest ratio ~A:1\n" (fmt-norm (/ (l2 hi) (l2 lo)))))))))

;; The global L2 gradient norm recorded by MONITOR, as mean and maximum
;; over BUCKETS consecutive ranges of steps.
(define (print-norm-trace monitor #!key (buckets 5))
  (let* ((history (get-gradient-history monitor))
         (n (length history)))
    (unless (zero? n)
      (let ((size (max 1 (quotient (+ n buckets -1) buckets))))
        (printf "    Gradient norm over training (L2, before clipping):\n")
        (let loop ((h history))
          (unless (null? h)
            (let* ((chunk (if (> (length h) size) (take h size) h))
                   (norms (map (lambda (e) (cdr (assq 'global-l2-norm e))) chunk))
                   (unhealthy (count (lambda (e) (not (cdr (assq 'healthy? e)))) chunk)))
              (printf "      steps ~A-~A: mean ~A, max ~A~A\n"
                      (cdr (assq 'step (car chunk))) (cdr (assq 'step (last chunk)))
                      (fmt-norm (/ (apply + norms) (length norms)))
                      (fmt-norm (apply max norms))
                      (if (zero? unhealthy) "" (sprintf ", ~A flagged" unhealthy)))
              (loop (drop h (length chunk))))))))))

;; The summary of diagnose-training for MONITOR (which looks at the last
;; ten recorded steps), and the clipping count of the monitored
;; optimizer OPT.
(define (print-training-diagnosis opt label)
  (let* ((d (diagnose-training (gradient-monitor-of opt)))
         (get (lambda (k) (cdr (assq k d)))))
    (printf "\n    Gradient diagnosis: ~A\n" label)
    (if (assq 'diagnosis d)
        (printf "      ~A\n" (get 'diagnosis))
        (begin
          (printf "      steps recorded ~A, flagged ~A, clipped ~A\n"
                  (get 'total-steps) (get 'warning-count) (clipped-steps opt))
          (printf "      last ~A steps: mean norm ~A, range [~A, ~A], stability (min/max) ~A\n"
                  (get 'recent-steps) (fmt-norm (get 'mean-gradient-norm))
                  (fmt-norm (get 'min-gradient-norm)) (fmt-norm (get 'max-gradient-norm))
                  (fmt-norm (get 'gradient-stability)))
          (for-each (lambda (r) (printf "      - ~A\n" r)) (get 'recommendations))))))
