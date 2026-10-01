;; Comprehensive Test Suite for ResNet Implementation
;; Tests components, gradient flow, and complete models
;; Uses Chicken Scheme test module

(import scheme
        (chicken base)
        (chicken blob)
        (chicken format)
        (chicken flonum)
        (chicken random)
        (srfi 1)
        (srfi 4)
        test
        yasos
        nanograd-autograd
        nanograd-layer
        nanograd-resnet)


;;; ==================================================================
;;; Test Utilities
;;; ==================================================================

(define (set-random-seed! seed)
  (set-pseudo-random-seed! (string->blob (number->string seed))))

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

(define (assert-no-nan tensor name)
  "Check that tensor contains no NaN values"
  (let ((data (tensor-data tensor))
        (dtype (tensor-dtype tensor)))
    (test-assert 
     (string-append name " no NaN")
     (not (case dtype
            ((f32)
             (let loop ((i 0))
               (cond
                ((= i (f32vector-length data)) #f)
                ((nan? (f32vector-ref data i)) #t)
                (else (loop (+ i 1))))))
            ((f64)
             (let loop ((i 0))
               (cond
                ((= i (f64vector-length data)) #f)
                ((nan? (f64vector-ref data i)) #t)
                (else (loop (+ i 1))))))
            (else #f)))
     )))

(define (approx-equal? actual expected tolerance)
  "Check if two numbers are approximately equal within tolerance"
  (<= (abs (- actual expected)) tolerance))

(define-syntax test-approximate
  (syntax-rules ()
    ((test-approximate name expected actual tolerance)
     (test-assert name (approx-equal? actual expected tolerance)))))

(define (tensor-mean tensor)
  "Compute mean of tensor values"
  (let ((data (tensor-data tensor))
        (dtype (tensor-dtype tensor)))
    (case dtype
      ((f32)
       (let ((n (f32vector-length data)))
         (/ (let loop ((i 0) (sum 0.0))
              (if (= i n) sum
                  (loop (+ i 1) (+ sum (f32vector-ref data i)))))
            n)))
      ((f64)
       (let ((n (f64vector-length data)))
         (/ (let loop ((i 0) (sum 0.0))
              (if (= i n) sum
                  (loop (+ i 1) (+ sum (f64vector-ref data i)))))
            n)))
      (else 0.0))))

(define (tensor-std tensor)
  "Compute standard deviation of tensor values"
  (let* ((mean-val (tensor-mean tensor))
         (data (tensor-data tensor))
         (dtype (tensor-dtype tensor)))
    (case dtype
      ((f32)
       (let ((n (f32vector-length data)))
         (sqrt (/ (let loop ((i 0) (sum 0.0))
                    (if (= i n) sum
                        (let ((diff (- (f32vector-ref data i) mean-val)))
                          (loop (+ i 1) (+ sum (* diff diff))))))
                  n))))
      ((f64)
       (let ((n (f64vector-length data)))
         (sqrt (/ (let loop ((i 0) (sum 0.0))
                    (if (= i n) sum
                        (let ((diff (- (f64vector-ref data i) mean-val)))
                          (loop (+ i 1) (+ sum (* diff diff))))))
                  n))))
      (else 0.0))))

;;; ==================================================================
;;; Batch Normalization 2D
;;; ==================================================================

(test-group "Batch Normalization 2D"
  (let* ((dtype 'f32)
         ;; Create input: (2, 3, 2, 2) - batch=2, channels=3, height=2, width=2
         (input-data (f32vector 1.0 2.0 3.0 4.0    ;; Channel 0, sample 0
                                5.0 6.0 7.0 8.0    ;; Channel 1, sample 0
                                9.0 10.0 11.0 12.0 ;; Channel 2, sample 0
                                13.0 14.0 15.0 16.0 ;; Channel 0, sample 1
                                17.0 18.0 19.0 20.0 ;; Channel 1, sample 1
                                21.0 22.0 23.0 24.0)) ;; Channel 2, sample 1
         (input (make-tensor32 input-data '(2 3 2 2)))
         (bn (make-batch-norm-2d 3 dtype: dtype)))
    
    (set-training-mode! bn #t)
    
    (let ((output (forward bn input)))
      (assert-shape '(2 3 2 2) output "output shape")
      (assert-no-nan output "output")
      
      ;; Check channel 0 mean after normalization
      (let ((out-data (tensor-data output)))
        (let ((ch0-mean (/ (+ (f32vector-ref out-data 0)
                              (f32vector-ref out-data 1)
                              (f32vector-ref out-data 2)
                              (f32vector-ref out-data 3)
                              (f32vector-ref out-data 12)
                              (f32vector-ref out-data 13)
                              (f32vector-ref out-data 14)
                              (f32vector-ref out-data 15))
                           8.0)))
          (test-approximate "channel 0 mean" 0.0 ch0-mean 0.1 )))
      
      ;; Test gradient flow with non-uniform gradients
      (let ((out-grad (tensor-grad output)))
        (do ((i 0 (+ i 1)))
            ((= i (f32vector-length out-grad)))
          (f32vector-set! out-grad i (exact->inexact (+ i 1)))))
      
      (backward! output)
      (assert-gradient-exists input "input gradient")
      
      ;; Test eval mode
      (set-eval-mode! bn)
      (let ((eval-output (forward bn input)))
        (assert-shape '(2 3 2 2) eval-output "eval output shape")
        (assert-no-nan eval-output "eval output"))
      )))

;;; ==================================================================
;;; Basic Residual Block
;;; ==================================================================

(test-group "Basic Residual Block"
  (let* ((dtype 'f32)
         (input-data (make-f32vector (* 64 8 8) 0.5))
         (input (make-tensor32 input-data '(64 8 8)))
         (block (make-residual-block 64 64 stride: 1 dtype: dtype)))
    
    (let ((output (forward block input)))
      (assert-shape '(64 8 8) output "output shape")
      (assert-no-nan output "output")
      
      (backward! output)
      (assert-gradient-exists input "input gradient")
      
      (let ((params (parameters block)))
        (test "parameter count" 8 (length params) )))))

;;; ==================================================================
;;; Basic Block with Downsample
;;; ==================================================================

(test-group "Basic Block with Downsample"
  (let* ((dtype 'f32)
         (input-data (make-f32vector (* 64 8 8) 0.5))
         (input (make-tensor32 input-data '(64 8 8)))
         (downsample (make-downsample 64 128 2 dtype))
         (block (make-residual-block 64 128 stride: 2 
                                     downsample: downsample
                                     dtype: dtype)))
    
    (let ((output (forward block input)))
      (assert-shape '(128 4 4) output "output shape")
      (assert-no-nan output "output")
      
      (backward! output)
      (assert-gradient-exists input "input gradient"))))

;;; ==================================================================
;;; Bottleneck Block
;;; ==================================================================

(test-group "Bottleneck Block"
  (let* ((dtype 'f32)
         (input-data (make-f32vector (* 256 8 8) 0.5))
         (input (make-tensor32 input-data '(256 8 8)))
         (block (make-bottleneck-block 256 256 stride: 1 dtype: dtype)))
    
    (let ((output (forward block input)))
      (assert-shape '(256 8 8) output "output shape")
      (assert-no-nan output "output")
      
      (backward! output)
      (assert-gradient-exists input "input gradient")
      
      (let ((params (parameters block)))
        (test "parameter count" 12 (length params) )))))

;;; ==================================================================
;;; Global Average Pooling
;;; ==================================================================

(test-group "Global Average Pooling"
  (let* ((dtype 'f32)
         (input-data (f32vector 1.0 2.0 3.0 4.0    ;; Channel 0
                                5.0 6.0 7.0 8.0
                                9.0 10.0 11.0 12.0
                                13.0 14.0 15.0 16.0
                                
                                2.0 4.0 6.0 8.0    ;; Channel 1
                                10.0 12.0 14.0 16.0
                                18.0 20.0 22.0 24.0
                                26.0 28.0 30.0 32.0
                                
                                3.0 6.0 9.0 12.0   ;; Channel 2
                                15.0 18.0 21.0 24.0
                                27.0 30.0 33.0 36.0
                                39.0 42.0 45.0 48.0))
         (input (make-tensor32 input-data '(3 4 4))))
    
    (let ((output (global-avg-pool2d input)))
      (assert-shape '(3) output "output shape")
      
      (let ((out-data (tensor-data output)))
        (test-approximate "channel 0 mean" 8.5 (f32vector-ref out-data 0) 0.001 )
        (test-approximate "channel 1 mean" 17.0 (f32vector-ref out-data 1) 0.001 )
        (test-approximate "channel 2 mean" 25.5 (f32vector-ref out-data 2) 0.001 ))
      
      (backward! output)
      (assert-gradient-exists input "input gradient"))))

;;; ==================================================================
;;; Global Average Pooling with Batch
;;; ==================================================================

(test-group "Global Average Pooling (Batched)"
  (let* ((dtype 'f32)
         (input-data (make-f32vector (* 2 3 2 2) 1.0))
         (input (make-tensor32 input-data '(2 3 2 2))))
    
    (let ((output (global-avg-pool2d input)))
      (assert-shape '(2 3) output "output shape")
      
      (let ((out-data (tensor-data output)))
        (test-approximate "value" 1.0 (f32vector-ref out-data 0) 0.001 ))
      
      (backward! output)
      (assert-gradient-exists input "input gradient"))))

;; ;;; ==================================================================
;; ;;; MaxPool2D
;; ;;; ==================================================================

(test-group "MaxPool2D"
  (let* ((dtype 'f32)
         (input-data (f32vector 1.0 2.0 3.0 4.0    ;; Channel 0
                                5.0 6.0 7.0 8.0
                                9.0 10.0 11.0 12.0
                                13.0 14.0 15.0 16.0
                                
                                16.0 15.0 14.0 13.0 ;; Channel 1
                                12.0 11.0 10.0 9.0
                                8.0 7.0 6.0 5.0
                                4.0 3.0 2.0 1.0))
         (input (make-tensor32 input-data '(2 4 4))))
    
    (let ((output (maxpool2d input 2 stride: 2)))
      (assert-shape '(2 2 2) output "output shape")
      
      (let ((out-data (tensor-data output)))
        (test-approximate "[0,0,0]" 6.0 (f32vector-ref out-data 0) 0.001 )
        (test-approximate "[0,0,1]" 8.0 (f32vector-ref out-data 1) 0.001 )
        (test-approximate "[1,1,0]" 8.0 (f32vector-ref out-data 6) 0.001 ))
      
      (backward! output)
      (assert-gradient-exists input "input gradient"))))

;;; ==================================================================
;;; ResNet First Layer
;;; ==================================================================

(test-group "ResNet First Layer (Conv+BN+ReLU)"
            (let* ((dtype 'f32)
                   (input-data (let ((v (make-f32vector (* 3 32 32))))
                                 (do ((i 0 (+ i 1)))
                                     ((= i (* 3 32 32)) v)
                                   (f32vector-set! v i (+ 0.3 (* 0.4 (pseudo-random-real)))))))
                   (input (make-tensor32 input-data '(3 32 32)))
                   (conv1 (make-conv2d-layer 3 64 7 stride: 2 padding: 3
                                             activation: (make-identity)
                                             dtype: dtype))
                   (bn1 (make-batch-norm-2d 64 dtype: dtype)))
              
              (set-training-mode! bn1 #t)
              
              (let* ((after-conv (forward conv1 input))
                     (after-bn (forward bn1 after-conv))
                     (after-relu (relu after-bn)))
                
                (assert-shape '(64 16 16) after-conv "after conv1")
                (assert-shape '(64 16 16) after-bn "after bn1")
                (assert-shape '(64 16 16) after-relu "after relu")
                (assert-no-nan after-relu "after relu"))))

;;; ==================================================================
;;; ResNet-18 Forward Pass
;;; ==================================================================

(test-group "ResNet-18 Forward Pass"
  (set-random-seed! 42)
  
  (let* ((dtype 'f32)
         (num-classes 10)
         (input-data (let ((v (make-f32vector (* 3 64 64))))
                       (do ((i 0 (+ i 1)))
                           ((= i (* 3 64 64)) v)
                         (f32vector-set! v i 
                                         (+ 0.3 (* 0.4 (pseudo-random-real)))))))
         (input (make-tensor32 input-data '(3 64 64)))
         (model (make-resnet-18 num-classes dtype: dtype)))
    
    (set-training-mode! model #t)
    
    (let ((output (forward model input)))
      (assert-shape (list num-classes) output "output shape")
      (assert-no-nan output "output")
      
      ;; Check output range (logits typically in [-10, 10])
      (let* ((out-data (tensor-data output))
             (max-val (f32vector-fold max -inf.0 out-data))
             (min-val (f32vector-fold min +inf.0 out-data)))
        (test-assert "output range reasonable"
                     (and (> max-val -100.0) (< max-val 100.0)
                          (> min-val -100.0) (< min-val 100.0))
                     ))
      
      ;; Test gradient flow
      (let ((out-grad (tensor-grad output)))
        (do ((i 0 (+ i 1)))
            ((= i (f32vector-length out-grad)))
          (f32vector-set! out-grad i (exact->inexact (+ i 1)))))
      
      (backward! output)
      (assert-gradient-exists input "input gradient")
      
      (let ((params (parameters model)))
        (test-assert "parameter count" (= (length params) 82) )))))

;; ;;; ==================================================================
;; ;;; ResNet-18 with Loss
;; ;;; ==================================================================

(test-group "ResNet-18 with Loss"
  (let* ((dtype 'f32)
         (num-classes 10)
         (input-data (make-f32vector (* 3 64 64) 0.5))
         (input (make-tensor32 input-data '(3 64 64)))
         (target-data (f32vector 3.0))
         (target (make-tensor32 target-data '(1) requires-grad?: #f))
         (model (make-resnet-18 num-classes dtype: dtype)))
    
    (set-training-mode! model #t)
    
    (let* ((logits (forward model input))
           (loss (cross-entropy-loss logits target from-logits: #t)))
      
      (assert-shape '(1) loss "loss shape")
      
      (let ((loss-val (f32vector-ref (tensor-data loss) 0)))
        (test-assert "loss positive" (> loss-val 0.0) ))
      
      (backward! loss)
      (assert-gradient-exists input "input gradient")
      (assert-gradient-exists logits "logits gradient")
      
      (let* ((params (parameters model))
             (params-with-grad 
              (length (filter (lambda (p) 
                               (let ((g (tensor-grad p)))
                                 (and g (> (f32vector-length g) 0))))
                             params))))
        (test-assert "parameters have gradients" (> params-with-grad 0) )))))

;;; ==================================================================
;;; ResNet-18 Training/Eval Mode
;;; ==================================================================

(test-group "ResNet-18 Training/Eval Mode"
  (let* ((dtype 'f32)
         (num-classes 10)
         (input-data (make-f32vector (* 3 64 64) 0.5))
         (input (make-tensor32 input-data '(3 64 64)))
         (model (make-resnet-18 num-classes dtype: dtype)))
    
    (set-training-mode! model #t)
    (let ((output1 (forward model input)))
      (test-assert "training mode forward" #t ))
    
    (set-eval-mode! model)
    (let ((output2 (forward model input)))
      (test-assert "eval mode forward" #t ))))

;; ;;; ==================================================================
;; ;;; ResNet-18 Batch Support
;; ;;; ==================================================================

(test-group "ResNet-18 Batch Support"
  (let* ((dtype 'f32)
         (num-classes 10)
         (batch-size 4)
         (input-data (make-f32vector (* batch-size 3 64 64) 0.5))
         (input (make-tensor32 input-data (list batch-size 3 64 64)))
         (model (make-resnet-18 num-classes dtype: dtype)))
    
    (set-training-mode! model #t)
    
    (let ((output (forward model input)))
      (assert-shape (list batch-size num-classes) output "batch output shape")
      (assert-no-nan output "batch output")
      
      (backward! output)
      (assert-gradient-exists input "batch input gradient"))))

;;; ==================================================================
;;; ResNet-50 Architecture
;;; ==================================================================

(test-group "ResNet-50 Architecture"
  (let* ((dtype 'f32)
         (num-classes 10)
         (input-data (make-f32vector (* 3 64 64) 0.5))
         (input (make-tensor32 input-data '(3 64 64)))
         (model (make-resnet-50 num-classes dtype: dtype)))
    
    (let ((output (forward model input)))
      (assert-shape (list num-classes) output "output shape")
      (assert-no-nan output "output")
      
      (let ((params (parameters model)))
        (test-assert "parameter count" (= (length params) 214) )))))

;;; ==================================================================
;;; Gradient Magnitude Sanity Check
;;; ==================================================================

(test-group "Gradient Magnitude Sanity"
  (let* ((dtype 'f32)
         (num-classes 10)
         (input-data (make-f32vector (* 3 64 64) 0.5))
         (input (make-tensor32 input-data '(3 64 64)))
         (target-data (f32vector 5.0))
         (target (make-tensor32 target-data '(1) requires-grad?: #f))
         (model (make-resnet-18 num-classes dtype: dtype)))
    
    (set-training-mode! model #t)
    
    (let* ((logits (forward model input))
           (loss (cross-entropy-loss logits target from-logits: #t)))
      (backward! loss)
      
      (let* ((params (parameters model))
             (first-param (car params))
             (grad (tensor-grad first-param)))
        (when grad
          (let* ((n (f32vector-length grad))
                 (grad-sum (let loop ((i 0) (sum 0.0))
                            (if (= i n) sum
                                (loop (+ i 1) (+ sum (f32vector-ref grad i))))))
                 (grad-mean (/ grad-sum n))
                 (grad-max (let loop ((i 0) (max-val -inf.0))
                            (if (= i n) max-val
                                (loop (+ i 1) (max max-val (f32vector-ref grad i)))))))
            (test-assert "gradient magnitudes reasonable"
                         (and (> grad-mean -1.0) (< grad-max 1000.0))
                        )))))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)
