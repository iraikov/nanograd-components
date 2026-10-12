;;; bench-am-common.scm -- shared code for the lazy-model benchmarks
;;;
;;; Included by bench-am-resnet-crunch.scm, bench-am-mlp-mixer-crunch.scm
;;; and bench-am-transformer-crunch.scm.  It selects the array-morphisms
;;; backends, provides wall-clock timing helpers, and runs the common
;;; benchmark protocol (see run-benchmark below).
;;;
;;; Environment variables:
;;;   AM_BENCH_BACKEND   crunch         crunch GEMM/conv kernels, threaded
;;;                                     crunch element-wise kernels (default)
;;;                      crunch-single  crunch GEMM/conv kernels,
;;;                                     single-thread element-wise kernels
;;;                      blas           system-BLAS egg backend, Scheme
;;;                                     element-wise code and Scheme Adam
;;;                      blas-crunch-act  system BLAS with threaded crunch
;;;                                     element-wise kernels
;;;                      scheme         no backend registered: the Scheme
;;;                                     code of array-morphisms throughout
;;;                    Every configuration except blas and scheme runs the
;;;                    Adam update on the crunch kernel.
;;;   AM_BENCH_CONFIG    small (default) or medium model and batch sizes
;;;   AM_BENCH_STEPS     timed replay steps (default 20)
;;;   AM_BENCH_STRICT    1 (default) to also time the strict model, 0 to skip
;;;   AM_BENCH_STRICT_STEPS  timed strict steps (default 3)
;;;   AM_BENCH_MONITOR   1 (default) to also time replay steps with gradient
;;;                      monitoring (gradient-monitoring.scm) and print the
;;;                      gradient diagnosis, 0 to skip
;;;   AM_CRUNCH_THREADS  thread count of the crunch kernels
;;;
;;; OPENBLAS_NUM_THREADS=1 is recommended with the blas backends.

(import scheme (chicken base) (chicken format) (chicken random) (chicken blob)
        (chicken process-context) (chicken sort))
(import (only (scheme time) current-jiffy jiffies-per-second))
(import (only srfi-1 fold iota take drop last append-map))
(import srfi-4)
(import yasos)
(import array-morphisms-core array-morphisms-context array-morphisms-realization)
(import array-morphisms-blas-exec array-morphisms-activation-exec)
(import (only array-morphisms-ssa replay-timing-reset! replay-timing-results))
(import array-morphisms-blas-egg-backend)
(import array-morphisms-crunch-conv-backend array-morphisms-crunch-blas-backend
        array-morphisms-crunch-activations array-morphisms-crunch-threads
        array-morphisms-crunch-optimizers)
(import (prefix array-morphisms-grad am:))
(import nanograd-autograd nanograd-layer nanograd-optimizer)
(import nanograd-array-morphisms nanograd-am-common)
(include-relative "gradient-monitoring.scm")

;;; ------------------------------------------------------------
;;; Settings
;;; ------------------------------------------------------------

(define (env-string name default)
  (or (get-environment-variable name) default))

(define (env-number name default)
  (let ((s (get-environment-variable name)))
    (or (and s (string->number s)) default)))

(define backend-choice (string->symbol (env-string "AM_BENCH_BACKEND" "crunch")))
(define bench-config (string->symbol (env-string "AM_BENCH_CONFIG" "small")))
(define bench-steps (env-number "AM_BENCH_STEPS" 20))
(define bench-strict? (not (equal? (env-string "AM_BENCH_STRICT" "1") "0")))
(define bench-strict-steps (env-number "AM_BENCH_STRICT_STEPS" 3))
(define bench-monitor? (not (equal? (env-string "AM_BENCH_MONITOR" "1") "0")))

(unless (memq bench-config '(small medium))
  (error "AM_BENCH_CONFIG must be small or medium" bench-config))

;;; ------------------------------------------------------------
;;; Backend selection
;;; ------------------------------------------------------------

(case backend-choice
  ((crunch)
   (register-blas-backend! (make-crunch-blas-backend))
   (register-activation-backend! (make-crunch-threaded-activation-backend)))
  ((crunch-single)
   (register-blas-backend! (make-crunch-blas-backend))
   (register-activation-backend! (make-crunch-activation-backend)))
  ((blas)
   (register-blas-backend! (make-blas-egg-backend)))
  ((blas-crunch-act)
   (register-blas-backend! (make-blas-egg-backend))
   (register-activation-backend! (make-crunch-threaded-activation-backend)))
  ((scheme) (void))
  (else (error "AM_BENCH_BACKEND must be crunch, crunch-single, blas, blas-crunch-act or scheme"
               backend-choice)))

(unless (memq backend-choice '(blas scheme))
  (register-optimizer-kernel! 'adam 'f32 crunch-adam-f32)
  (register-optimizer-kernel! 'adam 'f64 crunch-adam-f64))

;;; ------------------------------------------------------------
;;; Timing and formatting
;;; ------------------------------------------------------------

(define (now-ms) (/ (* 1000.0 (current-jiffy)) (jiffies-per-second)))

(define (time-once thunk)
  (let* ((t0 (now-ms)) (v (thunk)))
    (values (- (now-ms) t0) v)))

;; Runs THUNK once as a warm-up, then REPS times; returns the mean and
;; the minimum time per run in milliseconds, and the last result.
(define (time-runs reps thunk)
  (thunk)
  (let loop ((i 0) (total 0.0) (best #f) (last-v #f))
    (if (= i reps)
        (values (/ total reps) best last-v)
        (let-values (((dt v) (time-once thunk)))
          (loop (+ i 1) (+ total dt) (if best (min best dt) dt) v)))))

(define (fmt x) (/ (round (* 100.0 x)) 100.0))

(define (pad-right s n)
  (if (>= (string-length s) n) s (string-append s (make-string (- n (string-length s)) #\space))))
(define (pad-left s n)
  (if (>= (string-length s) n) s (string-append (make-string (- n (string-length s)) #\space) s)))

;;; ------------------------------------------------------------
;;; Data helpers
;;; ------------------------------------------------------------

(set-pseudo-random-seed! (string->blob "nanograd-components-bench"))

(define (random-f32vector n #!key (scale 1.0))
  (let ((v (make-f32vector n)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (f32vector-set! v i (* scale (- (pseudo-random-real) 0.5))))))

;; Wraps an f32vector as a lazy tensor without copying.
(define (vector->lazy data shape)
  (get-or-make-lazy (am:make-var (make-morphism data shape 'f32) #f)))

(define (scalar-value t) (f32vector-ref (tensor-data t) 0))

(define (relative-difference a b)
  (/ (abs (- a b)) (max 1e-12 (abs a) (abs b))))

;;; ------------------------------------------------------------
;;; Benchmark protocol
;;; ------------------------------------------------------------

(define (print-header title description)
  (printf "~%==============================================================~%")
  (printf "  ~A~%" title)
  (printf "==============================================================~%")
  (printf "  Backend: ~A (~A)~%" backend-choice (blas-backend-name (active-blas-backend)))
  (when (memq backend-choice '(crunch crunch-single blas-crunch-act))
    (printf "  Crunch threads: ~A, native build: ~A~%"
            (crunch-thread-count) crunch-native-build?))
  (printf "  Configuration: ~A~%" bench-config)
  (for-each (lambda (line) (printf "  ~A~%" line)) description))

(define (print-replay-breakdown steps lazy-step)
  (printf "~%  4. Replay time per instruction type (mean over ~A steps)~%" steps)
  (replay-timing-reset!)
  (let ((t0 (now-ms)))
    (do ((i 0 (+ i 1))) ((= i steps)) (lazy-step))
    (let* ((step-ms (/ (- (now-ms) t0) steps))
           (timing (replay-timing-results))
           (per-tag (cdr (assq 'per-tag timing)))
           (counts (cdr (assq 'per-tag-counts timing)))
           (rows (sort per-tag (lambda (a b) (> (cdr a) (cdr b)))))
           (timed (fold (lambda (r acc) (+ acc (cdr r))) 0.0 per-tag)))
      (printf "    ~A~A~A~A~%" (pad-right "instruction" 24) (pad-left "per step" 10)
              (pad-left "ms/step" 10) (pad-left "% step" 9))
      (for-each
       (lambda (r)
         (let ((ms (/ (cdr r) steps)))
           (printf "    ~A~A~A~A~%"
                   (pad-right (symbol->string (car r)) 24)
                   (pad-left (number->string (quotient (cdr (assq (car r) counts)) steps)) 10)
                   (pad-left (number->string (fmt ms)) 10)
                   (pad-left (number->string (fmt (* 100 (/ ms step-ms)))) 9))))
       rows)
      (printf "    instructions ~A ms/step of ~A ms; outside the replay ~A ms~%"
              (fmt (/ timed steps)) (fmt step-ms) (fmt (- step-ms (/ timed steps)))))))

;; The protocol:
;;   1. STRICT-LOSS (a thunk returning the strict model's loss tensor for
;;      the benchmark batch, after the weights were copied into the lazy
;;      model) is compared with the loss of the first lazy step;
;;   2. the first lazy step (trace and plan compilation) is timed;
;;   3. BENCH-STEPS replay steps are timed;
;;   4. the replay time is broken down by instruction type;
;;   5. STRICT-STEP (forward, backward!, optimizer step, zero gradients)
;;      is timed when AM_BENCH_STRICT is not 0.
;;   6. when AM_BENCH_MONITOR is not 0, BENCH-STEPS replay steps are
;;      timed with LAZY-OPT wrapped by make-monitored-optimizer, which
;;      records the gradients of every step in a gradient monitor, and
;;      the gradient diagnosis of those steps is printed.
;; LAZY-STEP runs one am-training-step/ssa and returns the loss tensor;
;; called with an optimizer, it uses that one instead of LAZY-OPT.
;; IMBALANCE-THRESHOLD is the monitor's limit on the ratio of the largest
;; to the smallest per-parameter gradient norm.
(define (run-benchmark title description lazy-model lazy-opt lazy-step strict-loss strict-step
                       #!key (imbalance-threshold 1000.0))
  (print-header title description)
  (printf "  Parameters: ~A~%" (am-parameter-count lazy-model))

  (printf "~%  1. Cross-check at the first step (f32)~%")
  (let*-values (((strict-l) (scalar-value (strict-loss)))
                ((first-ms loss-t) (time-once lazy-step))
                ((lazy-l) (scalar-value loss-t))
                ((rel) (relative-difference strict-l lazy-l)))
    (printf "     strict loss ~A, lazy loss ~A, relative difference ~A: ~A~%"
            strict-l lazy-l rel (if (< rel 1e-4) "PASS" "FAIL"))
    (printf "~%  2. First lazy step (trace + plan compilation): ~A ms~%" (fmt first-ms)))

  (let-values (((mean best last-loss) (time-runs bench-steps lazy-step)))
    (printf "~%  3. Replay step: mean ~A ms, best ~A ms over ~A steps; loss now ~A~%"
            (fmt mean) (fmt best) bench-steps (scalar-value last-loss))
    (print-replay-breakdown (min bench-steps 10) lazy-step)
    (if bench-strict?
        (let-values (((s-mean s-best _) (time-runs bench-strict-steps strict-step)))
          (printf "~%  5. Strict model step: mean ~A ms, best ~A ms over ~A steps~%"
                  (fmt s-mean) (fmt s-best) bench-strict-steps)
          (printf "     speedup of the lazy replay step: ~Ax~%" (fmt (/ s-mean mean))))
        (printf "~%  5. Strict model step: skipped (AM_BENCH_STRICT=0)~%"))
    (if bench-monitor?
        (let* ((params (parameters lazy-model))
               (monitored (make-monitored-optimizer
                           lazy-opt params (make-gradient-monitor history-size: (+ bench-steps 1)
                                                  imbalance-threshold: imbalance-threshold)
                           exclude-inactive?: #t max-warnings: 1)))
          (printf "~%  6. Replay step with gradient monitoring~%")
          (let-values (((m-mean m-best _) (time-runs bench-steps (lambda () (lazy-step monitored)))))
            (printf "     mean ~A ms, best ~A ms over ~A steps; difference from step 3: ~A ms (~A%)~%"
                    (fmt m-mean) (fmt m-best) bench-steps (fmt (- m-mean mean))
                    (fmt (* 100 (/ (- m-mean mean) mean)))))
          (print-norm-trace (gradient-monitor-of monitored) buckets: 3)
          (print-training-diagnosis monitored "benchmark steps"))
        (printf "~%  6. Gradient monitoring: skipped (AM_BENCH_MONITOR=0)~%")))
  (printf "~%==============================================================~%"))
