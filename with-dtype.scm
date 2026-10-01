;; Hygienic macro for dtype-based operation dispatch using ir-macro-transformer
;; Transforms generic vector operations to type-specific BLAS operations

(define-syntax with-dtype
  (ir-macro-transformer
   (lambda (x inject compare)
     (let ((dtype-expr (cadr x))
           (body (cddr x)))
       ;; Map generic operation to type-specific operation
       (define (map-operation op dtype)
         (cond
          ((compare op 'subvec)
           (case dtype
             ((f32) 'ssub)
             ((f64) 'dsub)
             (else op)))
          ((compare op 'elt-ref)
           (case dtype
             ((f32) 'f32vector-ref)
             ((f64) 'f64vector-ref)
             (else op)))
          ((compare op 'elt-set!)
           (case dtype
             ((f32) 'f32vector-set!)
             ((f64) 'f64vector-set!)
             (else op)))
          ((compare op 'axpy!)
           (case dtype
             ((f32) 'saxpy!)
             ((f64) 'daxpy!)
             (else op)))
          ((compare op 'gemm!)
           (case dtype
             ((f32) 'sgemm!)
             ((f64) 'dgemm!)
             (else op)))
          ((compare op 'gemv!)
           (case dtype
             ((f32) 'sgemv!)
             ((f64) 'dgemv!)
             (else op)))
          ((compare op 'dot)
           (case dtype
             ((f32) 'sdot)
             ((f64) 'ddot)
             (else op)))
          ((compare op 'scal!)
           (case dtype
             ((f32) 'sscal!)
             ((f64) 'dscal!)
             (else op)))
          ((compare op 'vec)
           (case dtype
             ((f32) 'make-f32vector)
             ((f64) 'make-f64vector)
             (else op)))
          ((compare op 'vec0)
           (case dtype
             ((f32) 'f32vector)
             ((f64) 'f64vector)
             (else op)))
          ((compare op 'blit)
           (case dtype
             ((f32) 'sblit)
             ((f64) 'dblit)
             (else op)))
          ((compare op 'copy-to)
           (case dtype
             ((f32) 'sblit)
             ((f64) 'dblit)
             (else op)))
          ((compare op 'fold)
           (case dtype
             ((f32) 'f32vector-fold)
             ((f64) 'f64vector-fold)
             (else op)))
          (else op)))
       
       ;; Transform expression for a specific dtype
       ;; Only transform symbols in operator position (first element of a form)
       (define (transform-for-dtype expr dtype)
         (cond
          ((null? expr) expr)
          ((not (pair? expr)) expr)  ; Atoms (including symbols) are left unchanged
          ;; Don't transform quoted expressions
          ((and (pair? expr) (compare (car expr) 'quote)) expr)
          ;; Transform operator position, then recursively transform arguments
          (else
           (let ((op (car expr))
                 (args (cdr expr)))
             (cons (cond
                    ;; If operator is a symbol, try to map it
                    ((symbol? op) (map-operation op dtype))
                    ;; If operator is a pair (like in let-bindings), recurse into it
                    ((pair? op) (transform-for-dtype op dtype))
                    ;; Otherwise leave it as-is
                    (else op))
                   (map (lambda (arg) (transform-for-dtype arg dtype)) args))))))
       
       ;; Generate the case expression with all body forms
       `(case ,(inject dtype-expr)
          ((f32) (begin ,@(map (lambda (expr) (transform-for-dtype expr 'f32)) body)))
          ((f64) (begin ,@(map (lambda (expr) (transform-for-dtype expr 'f64)) body))))))))


;; Alternative version: compile-time dtype specialization
(define-syntax with-dtype/static
  (ir-macro-transformer
   (lambda (x inject compare)
     (let ((dtype (cadr x))
           (body (cddr x)))
       
       ;; Map generic operation to type-specific operation
       (define (map-operation op)
         (cond
          ((compare op 'elt-ref)
           (case dtype
             ((f32) 'f32vector-ref)
             ((f64) 'f64vector-ref)
             (else op)))
          ((compare op 'elt-set!)
           (case dtype
             ((f32) 'f32vector-set!)
             ((f64) 'f64vector-set!)
             (else op)))
          ((compare op 'axpy!)
           (case dtype
             ((f32) 'saxpy!)
             ((f64) 'daxpy!)
             (else op)))
          ((compare op 'gemm!)
           (case dtype
             ((f32) 'sgemm!)
             ((f64) 'dgemm!)
             (else op)))
          ((compare op 'gemv!)
           (case dtype
             ((f32) 'sgemv!)
             ((f64) 'dgemv!)
             (else op)))
          ((compare op 'dot)
           (case dtype
             ((f32) 'sdot)
             ((f64) 'ddot)
             (else op)))
          ((compare op 'scal!)
           (case dtype
             ((f32) 'sscal!)
             ((f64) 'dscal!)
             (else op)))
          ((compare op 'vec)
           (case dtype
             ((f32) 'make-f32vector)
             ((f64) 'make-f64vector)
             (else op)))
          ((compare op 'blit)
           (case dtype
             ((f32) 'sblit)
             ((f64) 'dblit)
             (else op)))
          (else op)))
       
       ;; Transform expression
       ;; Only transform symbols in operator position (first element of a form)
       (define (transform expr)
         (cond
          ((null? expr) expr)
          ((not (pair? expr)) expr)  ; Atoms (including symbols) are left unchanged
          ((and (pair? expr) (compare (car expr) 'quote)) expr)
          ;; Transform operator position, then recursively transform arguments
          (else
           (let ((op (car expr))
                 (args (cdr expr)))
             (cons (cond
                    ;; If operator is a symbol, try to map it
                    ((symbol? op) (map-operation op))
                    ;; If operator is a pair (like in let-bindings), recurse into it
                    ((pair? op) (transform op))
                    ;; Otherwise leave it as-is
                    (else op))
                   (map transform args))))))
       
       ;; Return transformed body
       `(begin ,@(map transform body))))))
