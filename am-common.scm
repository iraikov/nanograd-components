;;; am-common.scm
;;;
;;; Shared building blocks for the lazy (array-morphisms) models.
;;;
;;; The models in am-resnet.scm, am-mlp-mixer.scm and am-transformer.scm
;;; are written with the differentiable morph-variable operations of
;;; array-morphisms-grad (imported with the prefix am:).  Their forward
;;; pass builds a morphism graph instead of computing values, so a whole
;;; training step can be compiled once by am-training-step/ssa and then
;;; replayed, with the matrix products, convolutions and element-wise
;;; kernels run by whatever array-morphisms backends are registered.
;;;
;;; This module provides:
;;;   - parameter and constant constructors,
;;;   - row-wise softmax, RMSNorm and LayerNorm on [M, D] matrices,
;;;   - RMSNorm, LayerNorm and NHWC batch normalization layers,
;;;   - copying parameter values from a strict nanograd model, so that a
;;;     lazy model can be checked against its strict counterpart.
;;;
;;; Layers follow the nanograd layer protocol: forward, parameters,
;;; zero-grad-layer!, layer?, layer-name and the two mode operations.
;;; Their forward methods take and return lazy tensors.

(module nanograd-am-common

  (;; Morph-variable helpers
   mv lazy
   am-param am-param-fill am-param-uniform am-param-normal
   am-const am-const-fill am-scalar
   am-random-normal

   ;; Row-wise operations on [M, D] morph-variables
   am-softmax-rows
   am-rmsnorm-rows
   am-layernorm-rows

   ;; Layers
   make-am-rmsnorm
   make-am-layernorm
   make-am-batch-norm-nhwc

   ;; Losses
   am-half-mse-loss

   ;; Parameters
   am-copy-parameters!
   am-parameter-count
   am-parameter-shapes)

  (import scheme (chicken base) (chicken format) (chicken random))
  (import (only srfi-1 fold every iota))
  (import srfi-4)
  (import yasos)
  (import array-morphisms-core)
  (import array-morphisms-basic-ops)
  (import (prefix array-morphisms-grad am:))
  (import nanograd-autograd)
  (import nanograd-layer)
  (import nanograd-array-morphisms)

  ;;;; ================================================================
  ;;;; Morph-variables and lazy tensors
  ;;;; ================================================================

  (define (mv lt)
    "The morph-variable of lazy tensor LT."
    (lazy-tensor-morph-variable lt))

  (define (lazy v)
    "The lazy tensor of morph-variable V."
    (get-or-make-lazy v))

  (define (fill-vector n dtype f)
    "A typed vector of length N whose element i is (F i)."
    (let ((v (allocate-typed-vector dtype n)))
      (do ((i 0 (+ i 1))) ((= i n) v)
        (typed-vector-set! v dtype i (exact->inexact (f i))))))

  (define (am-random-normal)
    "A standard normal sample (Box-Muller)."
    (let ((u1 (max (pseudo-random-real) 1e-10))
          (u2 (pseudo-random-real)))
      (* (sqrt (* -2.0 (log u1)))
         (cos (* 2.0 3.141592653589793 u2)))))

  (define (am-param data shape dtype)
    "A trainable lazy tensor holding DATA (a typed vector) with SHAPE."
    (lazy (am:make-var (make-morphism data shape dtype) #t)))

  (define (am-param-fill shape value dtype)
    (am-param (fill-vector (apply * shape) dtype (lambda (i) value)) shape dtype))

  (define (am-param-uniform shape scale dtype)
    "Uniform on [-SCALE/2, SCALE/2)."
    (am-param (fill-vector (apply * shape) dtype
                           (lambda (i) (* scale (- (pseudo-random-real) 0.5))))
              shape dtype))

  (define (am-param-normal shape std dtype)
    (am-param (fill-vector (apply * shape) dtype
                           (lambda (i) (* std (am-random-normal))))
              shape dtype))

  (define (am-const data shape dtype)
    "A non-trainable morph-variable holding DATA with SHAPE."
    (am:make-var (make-morphism data shape dtype) #f))

  (define (am-const-fill shape dtype f)
    "A constant whose element i (row-major) is (F i)."
    (am-const (fill-vector (apply * shape) dtype f) shape dtype))

  (define (am-scalar value dtype)
    "A constant of shape (1); it broadcasts against any shape."
    (am:make-var (morph-from-list (list value) '(1) dtype) #f))

  (define (dtype-of v)
    (morph-dtype (am:var-value v)))

  ;;;; ================================================================
  ;;;; Row-wise operations
  ;;;; ================================================================

  (define (am-softmax-rows s)
    "Softmax over the last axis of the [M, N] morph-variable S.

     The row maximum is subtracted first for numerical stability.  It is
     a constant in the graph (the reduce-max has no gradient), which does
     not change the gradient because softmax is invariant under a shift
     of each row."
    (let* ((m (am:make-var (morph-reduce 'max (am:var-value s) '(1) #t) #f))
           (e (am:var-exp (am:var- s m))))
      (am:var/ e (am:var-sum e '(1) #t))))

  (define (am-rmsnorm-rows x g eps)
    "RMSNorm of each row of the [M, D] morph-variable X:
       y = x / sqrt(mean(x^2) + EPS) * G,  G of shape [D]."
    (let* ((ms (am:var-mean (am:var* x x) '(1) #t))
           (rms (am:var-sqrt (am:var+ ms (am-scalar eps (dtype-of x))))))
      (am:var* (am:var/ x rms) g)))

  (define (am-layernorm-rows x g b eps)
    "LayerNorm of each row of the [M, D] morph-variable X:
       y = (x - mean(x)) / sqrt(var(x) + EPS) * G + B."
    (let* ((mu (am:var-mean x '(1) #t))
           (d (am:var- x mu))
           (var (am:var-mean (am:var* d d) '(1) #t))
           (sd (am:var-sqrt (am:var+ var (am-scalar eps (dtype-of x))))))
      (am:var+ (am:var* (am:var/ d sd) g) b)))

  ;;;; ================================================================
  ;;;; Normalization layers
  ;;;; ================================================================

  (define (make-am-rmsnorm dim #!key (eps 1e-6) (dtype 'f32) (name "AM-RMSNorm"))
    "RMSNorm over the last axis of a [M, DIM] lazy tensor, with a
     learnable scale initialized to ones."
    (let ((g (am-param-fill (list dim) 1.0 dtype)))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((layer-input-size self) dim)
       ((layer-output-size self) dim)
       ((parameters self) (list g))
       ((zero-grad-layer! self) (am:zero-grad! (mv g)))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x) (lazy (am-rmsnorm-rows (mv x) (mv g) eps)))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-layernorm dim #!key (eps 1e-6) (dtype 'f32) (name "AM-LayerNorm"))
    "LayerNorm over the last axis of a [M, DIM] lazy tensor, with a
     learnable scale (ones) and shift (zeros)."
    (let ((g (am-param-fill (list dim) 1.0 dtype))
          (b (am-param-fill (list dim) 0.0 dtype)))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((layer-input-size self) dim)
       ((layer-output-size self) dim)
       ((parameters self) (list g b))
       ((zero-grad-layer! self) (am:zero-grad! (mv g)) (am:zero-grad! (mv b)))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x) (lazy (am-layernorm-rows (mv x) (mv g) (mv b) eps)))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  (define (make-am-batch-norm-nhwc channels
                                   #!key (eps 1e-5) (dtype 'f32) (name "AM-BN"))
    "Batch normalization of an NHWC lazy tensor [N, H, W, CHANNELS].

     Each channel is normalized with the mean and biased variance of the
     batch over N, H and W, then scaled by gamma (ones) and shifted by
     beta (zeros).  The statistics are always those of the current
     batch: running averages are not kept, because a compiled SSA step
     returns only the loss and the parameter gradients."
    (let ((gamma (am-param-fill (list channels) 1.0 dtype))
          (beta (am-param-fill (list channels) 0.0 dtype)))
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       ((layer-input-size self) channels)
       ((layer-output-size self) channels)
       ((parameters self) (list gamma beta))
       ((zero-grad-layer! self)
        (am:zero-grad! (mv gamma))
        (am:zero-grad! (mv beta)))
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       ((forward self x-lt)
        (let* ((x (mv x-lt))
               (shape (vector->list (morph-shape (am:var-value x))))
               (rows (quotient (apply * shape) channels))
               ;; [N*H*W, C] view; the statistics have shape [C]
               (x2 (am:var-reshape x (list rows channels)))
               (mu (am:var-mean x2 '(0)))
               (d (am:var- x2 mu))
               (var (am:var-mean (am:var* d d) '(0)))
               (sd (am:var-sqrt (am:var+ var (am-scalar eps dtype))))
               (y (am:var+ (am:var* (am:var/ d sd) (mv gamma)) (mv beta))))
          (lazy (am:var-reshape y shape))))
       ((layer->serializable self) #f)
       ((save-layer self path) (void)))))

  ;;;; ================================================================
  ;;;; Losses
  ;;;; ================================================================

  (define (am-half-mse-loss pred-lt target-lt)
    "0.5 * mean((pred - target)^2), the convention of nanograd's strict
     mse-loss (am-mse-loss omits the factor 0.5)."
    (let ((l (mv (am-mse-loss pred-lt target-lt))))
      (lazy (am:var* (am-scalar 0.5 (dtype-of l)) l))))

  ;;;; ================================================================
  ;;;; Parameters
  ;;;; ================================================================

  (define (tensor-size t) (apply * (tensor-shape t)))

  (define (am-parameter-count model)
    "Total number of parameter elements of MODEL."
    (fold + 0 (map tensor-size (parameters model))))

  (define (am-parameter-shapes model)
    (map tensor-shape (parameters model)))

  (define (am-copy-parameters! src dst)
    "Copy the parameter values of model SRC into model DST, element by
     element in row-major order.  The two parameter lists must have the
     same length and matching element counts; shapes may differ (for
     example a strict conv weight [out, in, kh, kw] and a lazy one
     [out, in*kh*kw]).  Values are copied, never shared."
    (let ((ps (parameters src))
          (pd (parameters dst)))
      (unless (= (length ps) (length pd))
        (error 'am-copy-parameters! "parameter counts differ"
               (length ps) (length pd)))
      (for-each
       (lambda (s d k)
         (unless (= (tensor-size s) (tensor-size d))
           (error 'am-copy-parameters! "parameter sizes differ" k
                  (tensor-shape s) (tensor-shape d)))
         (let ((sv (tensor-data s))
               (dv (tensor-data d))
               (sdt (tensor-dtype s))
               (ddt (tensor-dtype d)))
           (do ((i 0 (+ i 1))) ((= i (tensor-size s)))
             (typed-vector-set! dv ddt i (typed-vector-ref sv sdt i)))))
       ps pd (iota (length ps)))))

  ) ;; end module
