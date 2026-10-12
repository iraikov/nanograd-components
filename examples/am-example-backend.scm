;;; am-example-backend.scm -- backend selection and data helpers for the
;;; lazy-model examples
;;;
;;; Included by am_resnet_example.scm and am_transformer_example.scm.
;;; The environment variable AM_EXAMPLE_BACKEND selects the backends that
;;; run the compiled SSA programs:
;;;
;;;   crunch   crunch GEMM/convolution kernels, threaded crunch
;;;            element-wise kernels and the crunch Adam kernel (default)
;;;   blas     system-BLAS egg backend; element-wise code and Adam in
;;;            Scheme
;;;   scheme   no backend registered: the Scheme code of array-morphisms
;;;
;;; The crunch backend needs the array-morphisms-crunch egg, which is not
;;; a dependency of nanograd-components.  OPENBLAS_NUM_THREADS=1 is
;;; recommended with the blas backend.

(import (chicken process-context))
(import array-morphisms-core array-morphisms-context)
(import array-morphisms-blas-exec array-morphisms-activation-exec)
(import array-morphisms-blas-egg-backend)
(import array-morphisms-crunch-conv-backend array-morphisms-crunch-activations
        array-morphisms-crunch-optimizers)
(import (prefix array-morphisms-grad am:))
(import nanograd-array-morphisms nanograd-am-common)

(define am-backend
  (string->symbol (or (get-environment-variable "AM_EXAMPLE_BACKEND") "crunch")))

(case am-backend
  ((crunch)
   (register-blas-backend! (make-crunch-blas-backend))
   (register-activation-backend! (make-crunch-threaded-activation-backend))
   (register-optimizer-kernel! 'adam 'f32 crunch-adam-f32)
   (register-optimizer-kernel! 'adam 'f64 crunch-adam-f64))
  ((blas)
   (register-blas-backend! (make-blas-egg-backend)))
  ((scheme) (void))
  (else (error "AM_EXAMPLE_BACKEND must be crunch, blas or scheme" am-backend)))

(define (vector->lazy data shape)
  "Wrap the f32vector DATA as a non-trainable lazy tensor of SHAPE,
   without copying."
  (get-or-make-lazy (am:make-var (make-morphism data shape 'f32) #f)))

(define (lazy-scalar t) (f32vector-ref (tensor-data t) 0))
