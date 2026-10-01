;; Numerical Validation Tests for ResNet Components
;; Tests with exact expected values computed analytically

(import scheme
        (chicken base)
        (chicken format)
        (chicken flonum)
        (srfi 1)
        (srfi 4)
        yasos
        blas
        nanograd-autograd
        nanograd-layer
        nanograd-resnet)

;;; ==================================================================
;;; Test Utilities
;;; ==================================================================

(define (print-vector vec name #!optional (max-elems 10))
  "Print first few elements of a vector"
  (printf "~A: [" name)
  (let ((n (min max-elems (f32vector-length vec))))
    (do ((i 0 (+ i 1)))
        ((= i n))
      (printf "~A~A" (f32vector-ref vec i)
              (if (< i (- n 1)) ", " ""))))
  (printf "~A]\n" (if (> (f32vector-length vec) max-elems) "..." "")))

;; Number of failed checks; the script exits with status 1 if nonzero.
(define *tests-failed* 0)

(define (assert-close expected actual tolerance name)
  "Assert that two values are within tolerance"
  (let ((diff (abs (- expected actual))))
    (if (< diff tolerance)
        (printf "O ~A: PASS (~A ~~= ~A, diff=~A)\n" name expected actual diff)
        (begin
          (set! *tests-failed* (+ *tests-failed* 1))
          (printf "X ~A: FAIL (~A != ~A, diff=~A > ~A)\n"
                  name expected actual diff tolerance)))))


(define (test-assert condition name)
  "Test if condition is true."
  (if condition
      (printf "  O ~A\n" name)
      (begin
        (set! *tests-failed* (+ *tests-failed* 1))
        (printf "  X ~A: assertion failed\n" name))))

;;; ==================================================================
;;; Batch Norm with Known Statistics
;;; ==================================================================

(define (test-batchnorm-known-values)
  (printf "\n=== BatchNorm with Known Values ===\n")
  
  ;; Simple case: 1 channel, 2 samples, 2x2 spatial
  ;; Input: [1,2,3,4] for sample 0, [5,6,7,8] for sample 1
  ;; Channel mean: (1+2+3+4+5+6+7+8)/8 = 4.5
  ;; Channel var: sum((x-4.5)^2)/8 = 5.25
  ;; Channel std: sqrt(5.25 + 1e-5) ~= 2.291
  
  (let* ((input-data (f32vector 1.0 2.0 3.0 4.0  ;; Sample 0
                                5.0 6.0 7.0 8.0)) ;; Sample 1
         (input (make-tensor32 input-data '(2 1 2 2)))
         (bn (make-batch-norm-2d 1 epsilon: 1e-5 momentum: 0.1 dtype: 'f32)))
    
    (set-training-mode! bn #t)
    
    ;; Forward pass
    (let ((output (forward bn input)))
      
      (printf "Input values: ")
      (print-vector input-data "input")
      (printf "Output values: ")
      (print-vector (tensor-data output) "output")
      
      ;; Expected: (x - 4.5) / 2.291
      ;; For x=1: (1-4.5)/2.291 ~= -1.527
      ;; For x=8: (8-4.5)/2.291 ~= 1.527
      (let ((out-data (tensor-data output)))
        ;; With gamma=1, beta=0, output should be normalized
        (assert-close -1.527 (f32vector-ref out-data 0) 0.01 "BatchNorm x=1")
        (assert-close 1.527 (f32vector-ref out-data 7) 0.01 "BatchNorm x=8")
        
        ;; Mean of output should be ~=0
        (let ((mean (/ (f32vector-fold + 0.0 out-data) 
                       (f32vector-length out-data))))
          (assert-close 0.0 mean 0.01 "BatchNorm output mean"))))))

;;; ==================================================================
;;; Global Average Pooling - Exact Values
;;; ==================================================================

(define (test-gap-exact-values)
  (printf "\n=== Global Average Pooling Exact ===\n")
  
  ;; Single channel, simple values
  ;; Input: (1, 2, 2) = [1, 2, 3, 4]
  ;; Expected output: mean = (1+2+3+4)/4 = 2.5
  
  (let* ((input-data (f32vector 1.0 2.0 3.0 4.0))
         (input (make-tensor32 input-data '(1 2 2))))
    
    (let ((output (global-avg-pool2d input)))
      
      (printf "Input: ")
      (print-vector input-data "input")
      (printf "Output: ")
      (print-vector (tensor-data output) "output")
      
      (assert-close 2.5 (f32vector-ref (tensor-data output) 0) 0.001
                   "GAP mean value"))))

;;; ==================================================================
;;; Global Average Pooling - Multi-Channel
;;; ==================================================================

(define (test-gap-multichannel-exact)
  (printf "\n=== GAP Multi-Channel Exact ===\n")
  
  ;; Two channels: [1,2,3,4] and [5,6,7,8]
  ;; Expected: [2.5, 6.5]
  
  (let* ((input-data (f32vector 1.0 2.0 3.0 4.0   ;; Channel 0
                                5.0 6.0 7.0 8.0))  ;; Channel 1
         (input (make-tensor32 input-data '(2 2 2))))
    
    (let ((output (global-avg-pool2d input)))
      
      (let ((out-data (tensor-data output)))
        (assert-close 2.5 (f32vector-ref out-data 0) 0.001 "GAP channel 0")
        (assert-close 6.5 (f32vector-ref out-data 1) 0.001 "GAP channel 1")))))

;;; ==================================================================
;;; MaxPool2D - Known Max Values
;;; ==================================================================

(define (test-maxpool-exact)
  (printf "\n=== MaxPool2D Exact Values ===\n")
  
  ;; Input: (1, 4, 4) with sequential values
  ;; 2x2 pooling with stride 2
  ;; Window [0,0]: max(1,2,5,6) = 6
  ;; Window [0,1]: max(3,4,7,8) = 8
  ;; Window [1,0]: max(9,10,13,14) = 14
  ;; Window [1,1]: max(11,12,15,16) = 16
  
  (let* ((input-data (f32vector 1.0 2.0 3.0 4.0
                                5.0 6.0 7.0 8.0
                                9.0 10.0 11.0 12.0
                                13.0 14.0 15.0 16.0))
         (input (make-tensor32 input-data '(1 4 4))))
    
    (let ((output (maxpool2d input 2 stride: 2)))
      
      (let ((out-data (tensor-data output)))
        (assert-close 6.0 (f32vector-ref out-data 0) 0.001 "MaxPool [0,0]")
        (assert-close 8.0 (f32vector-ref out-data 1) 0.001 "MaxPool [0,1]")
        (assert-close 14.0 (f32vector-ref out-data 2) 0.001 "MaxPool [1,0]")
        (assert-close 16.0 (f32vector-ref out-data 3) 0.001 "MaxPool [1,1]")))))

;;; ==================================================================
;;; Skip Connection - Identity Path
;;; ==================================================================

(define (test-skip-connection-identity)
  (printf "\n=== Skip Connection Identity ===\n")
  
  ;; Test that skip connection properly adds input to output
  ;; Create simple tensors for manual verification
  
  (let* ((a-data (f32vector 1.0 2.0 3.0 4.0))
         (b-data (f32vector 5.0 6.0 7.0 8.0))
         (a (make-tensor32 a-data '(4)))
         (b (make-tensor32 b-data '(4))))
    
    (let ((result (add a b)))
      
      (let ((res-data (tensor-data result)))
        (assert-close 6.0 (f32vector-ref res-data 0) 0.001 "Skip add [0]")
        (assert-close 8.0 (f32vector-ref res-data 1) 0.001 "Skip add [1]")
        (assert-close 10.0 (f32vector-ref res-data 2) 0.001 "Skip add [2]")
        (assert-close 12.0 (f32vector-ref res-data 3) 0.001 "Skip add [3]"))
      
      ;; Test backward pass
      (backward! result)
      
      ;; Gradient should flow equally to both inputs
      (let ((grad-a (tensor-grad a))
            (grad-b (tensor-grad b)))
        (printf "Gradient to input A: ")
        (print-vector grad-a "grad-a")
        (printf "Gradient to input B: ")
        (print-vector grad-b "grad-b")))))

;;; ==================================================================
;;; Conv2D Output Size Computation
;;; ==================================================================

(define (test-conv2d-output-size)
  (printf "\n=== Conv2D Output Size ===\n")
  
  ;; Formula: out_size = floor((in_size + 2*padding - kernel_size) / stride) + 1
  
  ;; Test case 1: 32x32 input, 7x7 kernel, stride 2, padding 3
  ;; out = floor((32 + 2*3 - 7) / 2) + 1 = floor(31/2) + 1 = 15 + 1 = 16
  
  (let* ((input-data (make-f32vector (* 3 32 32) 1.0))
         (input (make-tensor32 input-data '(3 32 32)))
         (conv (make-conv2d-layer 3 64 7 stride: 2 padding: 3 
                                  activation: (make-identity)
                                  dtype: 'f32)))
    
    (let ((output (forward conv input)))
      (let ((out-shape (tensor-shape output)))
        (printf "Input shape: (3, 32, 32)\n")
        (printf "Conv: 7x7, stride=2, padding=3\n")
        (printf "Output shape: ~A\n" out-shape)
        (test-assert (= 64 (car out-shape)) "Conv output channels")
        (test-assert (= 16 (cadr out-shape)) "Conv output height")
        (test-assert (= 16 (caddr out-shape)) "Conv output width"))))
  
  ;; Test case 2: 16x16 input, 3x3 kernel, stride 2, padding 1
  ;; out = floor((16 + 2*1 - 3) / 2) + 1 = floor(15/2) + 1 = 7 + 1 = 8
  
  (let* ((input-data (make-f32vector (* 64 16 16) 1.0))
         (input (make-tensor32 input-data '(64 16 16)))
         (conv (make-conv2d-layer 64 128 3 stride: 2 padding: 1
                                  activation: (make-identity)
                                  dtype: 'f32)))
    
    (let ((output (forward conv input)))
      (let ((out-shape (tensor-shape output)))
        (printf "Input shape: (64, 16, 16)\n")
        (printf "Conv: 3x3, stride=2, padding=1\n")
        (printf "Output shape: ~A\n" out-shape)
        (test-assert (= 128 (car out-shape)) "Conv output channels")
        (test-assert (= 8 (cadr out-shape)) "Conv output height")
        (test-assert (= 8 (caddr out-shape)) "Conv output width")))))

;;; ==================================================================
;;; ResNet Stage Output Sizes
;;; ==================================================================

(define (test-resnet-stage-sizes)
  (printf "\n=== ResNet Stage Output Sizes ===\n")
  
  ;; Track output size through ResNet-18 stages
  ;; Input: (3, 32, 32)
  
  (printf "ResNet-18 on 32x32 input:\n")
  (printf "  Input:      (3, 32, 32)\n")
  (printf "  Conv1:      (64, 16, 16)  [7x7, stride=2, padding=3]\n")
  (printf "  MaxPool:    (64, 8, 8)    [3x3, stride=2]\n")
  (printf "  Layer1:     (64, 8, 8)    [2 blocks, stride=1]\n")
  (printf "  Layer2:     (128, 4, 4)   [2 blocks, stride=2]\n")
  (printf "  Layer3:     (256, 2, 2)   [2 blocks, stride=2]\n")
  (printf "  Layer4:     (512, 1, 1)   [2 blocks, stride=2]\n")
  (printf "  GAP:        (512,)        [global avg pool]\n")
  (printf "  FC:         (10,)         [classifier]\n")
  
  ;; Verify with actual forward pass
  (let* ((input-data (make-f32vector (* 3 32 32) 0.5))
         (input (make-tensor32 input-data '(3 32 32)))
         (model (make-resnet-18 10 dtype: 'f32)))
    
    (let ((output (forward model input)))
      (printf "\nActual output shape: ~A\n" (tensor-shape output))
      (test-assert (= 10 (car (tensor-shape output)))
                   "ResNet-18 output size"))))

;;; ==================================================================
;;; Gradient Backward Through Skip
;;; ==================================================================

(define (test-gradient-through-skip)
  (printf "\n=== Gradient Through Skip Connection ===\n")
  
  ;; Simple test: y = x + f(x) where f is identity
  ;; dy/dx should be 2.0 (one from direct path, one from function path)
  
  (let* ((x-data (f32vector 1.0 2.0 3.0 4.0))
         (x (make-tensor32 x-data '(4)))
         ;; Simulate f(x) = x (identity function)
         (fx (make-tensor32 x-data '(4)))
         (y (add x fx)))
    
    ;; y = [2, 4, 6, 8]
    (let ((y-data (tensor-data y)))
      (printf "y = x + f(x): ")
      (print-vector y-data "y"))
    
    ;; Backward from y (gradient is 1.0 everywhere)
    (backward! y)
    
    ;; Check gradient at x
    (let ((grad-x (tensor-grad x)))
      (printf "Gradient at x: ")
      (print-vector grad-x "grad")
      
      ;; Each element should have gradient 1.0 (from direct sum)
      ;; Note: fx has separate gradient storage
      (assert-close 1.0 (f32vector-ref grad-x 0) 0.001 "Skip gradient [0]"))))

;;; ==================================================================
;;; Cross-Entropy Loss Values
;;; ==================================================================

(define (test-cross-entropy-values)
  (printf "\n=== Cross-Entropy Loss Values ===\n")
  
  ;; Test with known logits and targets
  ;; For perfect prediction (target = argmax(logits)), loss should be small
  
  ;; Case 1: Perfect prediction
  (let* ((logits-data (f32vector -10.0 -10.0 10.0 -10.0 -10.0))
         (logits (make-tensor32 logits-data '(5)))
         (target-data (f32vector 2.0))  ;; Class 2 is the max
         (target (make-tensor32 target-data '(1) requires-grad?: #f)))
    
    (let ((loss (cross-entropy-loss logits target from-logits: #t)))
      (let ((loss-val (f32vector-ref (tensor-data loss) 0)))
        (printf "Perfect prediction loss: ~A\n" loss-val)
        ;; Loss should be close to 0
        (if (< loss-val 0.1)
            (printf "O Perfect prediction: PASS (loss < 0.1)\n")
            (printf "X Perfect prediction: FAIL (loss = ~A)\n" loss-val)))))
  
  ;; Case 2: Uniform distribution (maximum uncertainty)
  (let* ((logits-data (f32vector 0.0 0.0 0.0 0.0 0.0))
         (logits (make-tensor32 logits-data '(5)))
         (target-data (f32vector 2.0))
         (target (make-tensor32 target-data '(1) requires-grad?: #f)))
    
    (let ((loss (cross-entropy-loss logits target from-logits: #t)))
      (let ((loss-val (f32vector-ref (tensor-data loss) 0)))
        (printf "Uniform distribution loss: ~A\n" loss-val)
        ;; Loss should be -log(1/5) = log(5) ~= 1.609
        (assert-close 1.609 loss-val 0.01 "Uniform distribution loss")))))

;;; ==================================================================
;;; Run All Numerical Tests
;;; ==================================================================

(define (run-numerical-tests)
  (printf "\n")
  (printf "========================================\n")
  (printf "  ResNet Numerical Validation Tests\n")
  (printf "========================================\n")
  
  (test-batchnorm-known-values)
  (test-gap-exact-values)
  (test-gap-multichannel-exact)
  (test-maxpool-exact)
  (test-skip-connection-identity)
  (test-conv2d-output-size)
  (test-resnet-stage-sizes)
  (test-gradient-through-skip)
  (test-cross-entropy-values)
  
  (printf "\n")
  (printf "========================================\n")
  (printf "  All Numerical Tests Complete\n")
  (printf "========================================\n"))

;; Run numerical tests
(run-numerical-tests)
(printf "Failed checks: ~A\n" *tests-failed*)
(exit (if (zero? *tests-failed*) 0 1))
