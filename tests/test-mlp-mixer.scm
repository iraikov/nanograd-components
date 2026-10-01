;; Test suite for MLP-Mixer implementation in Nanograd
;; Uses CHICKEN Scheme test module

(import scheme
        (chicken base)
        (chicken format)
        (srfi 1)
        (srfi 4)
        test
        yasos
        nanograd-autograd
        nanograd-layer
        nanograd-mlp-mixer)

;;; ==================================================================
;;; Test Utilities
;;; ==================================================================

(define (assert-shape expected-shape tensor name)
  "Check if tensor has expected shape"
  (test-assert name (equal? expected-shape (tensor-shape tensor))))

(define (assert-gradient-exists tensor name)
  "Check if tensor has non-zero gradient after backward"
  (let ((grad (tensor-grad tensor)))
    (if (not grad)
        (test-assert (string-append name " gradient allocated") #f)
        (let* ((dtype (tensor-dtype tensor))
               (sum (case dtype
                      ((f32)
                       (let ((n (f32vector-length grad)))
                         (let loop ((i 0) (sum 0.0))
                           (if (= i n)
                               sum
                               (loop (+ i 1)
                                     (+ sum (abs (f32vector-ref grad i))))))))
                      ((f64)
                       (let ((n (f64vector-length grad)))
                         (let loop ((i 0) (sum 0.0))
                           (if (= i n)
                               sum
                               (loop (+ i 1)
                                     (+ sum (abs (f64vector-ref grad i))))))))
                      (else 0.0))))
          (test-assert (string-append name " gradient non-zero") (> sum 0.0))))))

(define (approx-equal? actual expected tolerance)
  "Check if two numbers are approximately equal within tolerance"
  (<= (abs (- actual expected)) tolerance))

(define-syntax test-approximate
  (syntax-rules ()
    ((test-approximate name expected actual tolerance)
     (test-assert name (approx-equal? actual expected tolerance)))))

;;; ==================================================================
;;; Helper Functions
;;; ==================================================================

(define (make-test-input seq-len hidden-dim #!key (dtype 'f32))
  "Create a test input tensor with predictable values."
  (let* ((size (* seq-len hidden-dim))
         (data (case dtype
                 ((f32) (list->f32vector (iota size 1.0 0.1)))
                 ((f64) (list->f64vector (iota size 1.0 0.1)))
                 (else (error "Unsupported dtype" dtype)))))
    (make-base-tensor data (list seq-len hidden-dim) dtype #t)))

(define (count-parameters obj)
  "Count total number of parameters in a layer."
  (let ((params (parameters obj)))
    (apply + (map (lambda (p)
                   (apply * (tensor-shape p)))
                 params))))

;;; ==================================================================
;;; Basic Mixer MLP
;;; ==================================================================

(test-group "Basic Mixer MLP"
  (let* ((mlp (make-mixer-mlp 128 512 dtype: 'f32))
         (input (make-test-input 4 128))
         (output (forward mlp input)))
    
    (assert-shape '(4 128) output "Output shape matches input")
    (test-assert "Output requires grad" (tensor-requires-grad? output))
    
    (let ((param-count (count-parameters mlp)))
      (test "Parameter count" 131712 param-count))
    
    (zero-grad-layer! mlp)
    (let ((loss (mean-tensor output)))
      (backward! loss)
      (test-assert "Gradients computed" 
                   (every (lambda (p) (tensor-grad p))
                         (parameters mlp))))))

;;; ==================================================================
;;; Token Mixing MLP
;;; ==================================================================

(test-group "Token Mixing MLP"
  (let* ((token-mixer (make-token-mixing-mlp 36 128 expansion-factor: 4))
         (input (make-test-input 36 128))
         (output (forward token-mixer input)))
    
    (assert-shape '(36 128) output "Output shape preserved")
    (test-assert "Output requires grad" (tensor-requires-grad? output))
    
    (let ((param-count (count-parameters token-mixer)))
      (test "Parameter count" 10548 param-count))))

;;; ==================================================================
;;; Channel Mixing MLP
;;; ==================================================================

(test-group "Channel Mixing MLP"
  (let* ((channel-mixer (make-channel-mixing-mlp 128 expansion-factor: 4))
         (input (make-test-input 36 128))
         (output (forward channel-mixer input)))
    
    (assert-shape '(36 128) output "Output shape preserved")
    (test-assert "Output requires grad" (tensor-requires-grad? output))
    
    (let ((param-count (count-parameters channel-mixer)))
      (test "Parameter count" 131712 param-count))))

;;; ==================================================================
;;; MLP-Mixer Block
;;; ==================================================================

(test-group "MLP-Mixer Block"
  (let* ((block (make-mlp-mixer-block 36 128 
                                     token-expansion: 4
                                     channel-expansion: 4
                                     use-rmsnorm: #t))
         (input (make-test-input 36 128))
         (output (forward block input)))
    
    (assert-shape '(36 128) output "Output shape preserved")
    (test-assert "Output requires grad" (tensor-requires-grad? output))
    
    (let ((param-count (count-parameters block)))
      (test "Parameter count" 142516 param-count))
    
    (let* ((input-mean (mean-tensor input))
           (output-mean (mean-tensor output))
           (input-val (f32vector-ref (tensor-data input-mean) 0))
           (output-val (f32vector-ref (tensor-data output-mean) 0)))
      (test-assert "Skip connections affect output"
                   (not (= input-val output-val))))))

;;; ==================================================================
;;; Full MLP-Mixer Model
;;; ==================================================================

(test-group "Full MLP-Mixer Model"
  (let* ((mixer (make-mlp-mixer 36 128 2
                               token-expansion: 4
                               channel-expansion: 4
                               use-rmsnorm: #t))
         (input (make-test-input 36 128))
         (output (forward mixer input)))
    
    (assert-shape '(36 128) output "Output shape preserved")
    (test-assert "Output requires grad" (tensor-requires-grad? output))
    (test "Number of blocks" 2 (num-blocks mixer))
    
    (let ((param-count (count-parameters mixer)))
      (test "Parameter count (2 blocks)" 285032 param-count))
    
    (zero-grad-layer! mixer)
    (let ((loss (mean-tensor output)))
      (backward! loss)
      (test-assert "Gradients flow through all blocks"
                   (every (lambda (p) (tensor-grad p))
                         (parameters mixer))))))

;;; ==================================================================
;;; Different Configurations
;;; ==================================================================

(test-group "Different Configurations"
  (let* ((small-mixer (make-mlp-mixer 16 64 1))
         (small-input (make-test-input 16 64))
         (small-output (forward small-mixer small-input)))
    (assert-shape '(16 64) small-output "Small mixer output"))
  
  (let* ((large-mixer (make-mlp-mixer 64 256 4))
         (large-input (make-test-input 64 256))
         (large-output (forward large-mixer large-input)))
    (assert-shape '(64 256) large-output "Large mixer output"))
  
  (let* ((custom-mixer (make-mlp-mixer 36 128 2
                                      token-expansion: 2
                                      channel-expansion: 8))
         (custom-input (make-test-input 36 128))
         (custom-output (forward custom-mixer custom-input)))
    (assert-shape '(36 128) custom-output "Custom expansion output")))

;;; ==================================================================
;;; Gradient Flow and Skip Connections
;;; ==================================================================

(test-group "Gradient Flow and Skip Connections"
  (let* ((block (make-mlp-mixer-block 8 16 
                                     token-expansion: 2 
                                     channel-expansion: 2))
         (input (make-test-input 8 16))
         (output (forward block input)))
    
    (zero-grad-layer! block)
    (let ((loss (mean-tensor (mul output output))))
      (backward! loss))
    
    (let ((params (parameters block)))
      (test-assert "All parameters have gradients"
                   (every (lambda (p) (tensor-grad p)) params))
      
      (test-assert "Gradients are non-zero"
                   (any (lambda (p)
                         (let ((grad (tensor-grad p)))
                           (and grad
                                (> (abs (f32vector-ref grad 0)) 1e-10))))
                       params)))))

;;; ==================================================================
;;; TRM Integration Test
;;; ==================================================================

(test-group "TRM Integration"
  (let* ((trm-mixer (make-mlp-mixer 36 128 2
                                   token-expansion: 4
                                   channel-expansion: 4
                                   use-rmsnorm: #t
                                   name: "TRM-Mixer"))
         (x (make-test-input 36 128))
         (y (make-test-input 36 128))
         (z (make-test-input 36 128))
         (combined (add (add x y) z))
         (output (forward trm-mixer combined)))
    
    (assert-shape '(36 128) output "TRM output shape")
    (test-assert "TRM output requires grad" (tensor-requires-grad? output))
    
    (let* ((z1 (forward trm-mixer (add (add x y) z)))
           (z2 (forward trm-mixer (add (add x y) z1)))
           (z3 (forward trm-mixer (add (add x y) z2))))
      (assert-shape '(36 128) z1 "Recursive z1 shape")
      (assert-shape '(36 128) z2 "Recursive z2 shape")
      (assert-shape '(36 128) z3 "Recursive z3 shape"))))

;;; ==================================================================
;;; Memory and Performance
;;; ==================================================================

(test-group "Memory and Performance"
  (let* ((mixer (make-mlp-mixer 36 128 2))
         (total-params (count-parameters mixer)))
    
    (test-assert "MLP-Mixer is parameter efficient"
                 (< total-params 500000))
    
    (let* ((input (make-test-input 36 128))
           (out1 (forward mixer input))
           (out2 (forward mixer input))
           (val1 (f32vector-ref (tensor-data out1) 0))
           (val2 (f32vector-ref (tensor-data out2) 0)))
      (test-approximate "Deterministic forward pass" val1 val2 1e-6))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)
