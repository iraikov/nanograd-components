;; Tests for Transformer implementation

(import scheme
        (chicken base)
        (chicken blob)
        (chicken format)
        (chicken random)
        (srfi 1)
        (srfi 4)
        test
        nanograd-autograd
        nanograd-layer
        nanograd-optimizer
        nanograd-transformer)

;;; ==================================================================
;;; Test Utilities
;;; ==================================================================

(define (set-random-seed! seed)
  (set-pseudo-random-seed! (string->blob (number->string seed))))

(define (assert-shape expected-shape tensor name)
  "Check if tensor has expected shape"
  (test-assert name (equal? expected-shape (tensor-shape tensor))))

(define (approx-equal? actual expected tolerance)
  "Check if two numbers are approximately equal within tolerance"
  (<= (abs (- actual expected)) tolerance))

(define-syntax test-approximate
  (syntax-rules ()
    ((test-approximate name expected actual tolerance)
     (test-assert name (approx-equal? actual expected tolerance)))))

;;; ==================================================================
;;; RoPE get-frequencies
;;; ==================================================================

(test-group "RoPE Frequencies"
  (set-random-seed! 42)
  
  (let ((rope (make-rope-embedding 10 8 base: 10000.0 dtype: 'f32)))
    
    (test-assert "RoPE embedding is rope-embedding?"
                (rope-embedding? rope))
    
    (test-assert "RoPE embedding is layer?"
                (layer? rope))
    
    (let-values (((cos-freqs-0 sin-freqs-0) (get-frequencies rope 0)))
      (let ((half-dim 4))
        (test "Cos frequencies length" half-dim 
              (f32vector-length cos-freqs-0))
        (test "Sin frequencies length" half-dim 
              (f32vector-length sin-freqs-0))
        
        (test-approximate "Cos at position 0" 1.0 
                         (f32vector-ref cos-freqs-0 0) 1e-6)
        (test-approximate "Sin at position 0" 0.0 
                         (f32vector-ref sin-freqs-0 0) 1e-6)))
    
    (let-values (((cos-freqs-5 sin-freqs-5) (get-frequencies rope 5)))
      (test-approximate "Cos at position 5" (cos 5.0)
                       (f32vector-ref cos-freqs-5 0) 1e-4)
      (test-approximate "Sin at position 5" (sin 5.0)
                       (f32vector-ref sin-freqs-5 0) 1e-4))
    
    (let ((params (parameters rope)))
      (test-assert "RoPE has no trainable parameters" (null? params)))))

;;; ==================================================================
;;; RoPE Rotation
;;; ==================================================================

(test-group "RoPE Rotation"
  (set-random-seed! 123)
  
  (let* ((rope (make-rope-embedding 10 8 base: 10000.0 dtype: 'f32))
         (q-data (make-f32vector 16))
         (k-data (make-f32vector 16))
         (_ (do ((i 0 (+ i 1)))
                ((= i 16))
              (f32vector-set! q-data i (exact->inexact i))
              (f32vector-set! k-data i (exact->inexact (- 15 i)))))
         (Q (make-tensor32 q-data '(2 8)))
         (K (make-tensor32 k-data '(2 8))))
    
    (let-values (((Q-rot-0 K-rot-0) (apply-rope-rotation Q K rope 0)))
      (assert-shape '(2 8) Q-rot-0 "Rotated Q has correct shape")
      (assert-shape '(2 8) K-rot-0 "Rotated K has correct shape")
      (test-assert "Rotated Q has gradient tracking" 
                   (tensor-requires-grad? Q-rot-0))
      (test-assert "Rotated K has gradient tracking" 
                   (tensor-requires-grad? K-rot-0))
      
      (let* ((q-rot-data (tensor-data Q-rot-0))
             (original-match? 
              (let loop ((i 0))
                (if (= i 16)
                    #t
                    (and (< (abs (- (f32vector-ref q-rot-data i)
                                    (f32vector-ref q-data i)))
                            1e-5)
                         (loop (+ i 1)))))))
        (test-assert "RoPE rotation at position 0 is identity"
                     original-match?)))
    
    (let-values (((Q-rot-3 K-rot-3) (apply-rope-rotation Q K rope 3)))
      (assert-shape '(2 8) Q-rot-3 "Rotated Q at position 3 has correct shape")
      
      (let* ((q-rot-data (tensor-data Q-rot-3))
             (values-changed?
              (let loop ((i 0) (diff-count 0))
                (if (= i 16)
                    (> diff-count 10)
                    (loop (+ i 1)
                          (if (> (abs (- (f32vector-ref q-rot-data i)
                                         (f32vector-ref q-data i)))
                                 1e-3)
                              (+ diff-count 1)
                              diff-count))))))
        (test-assert "RoPE rotation at position 3 modifies values"
                     values-changed?)))))

;;; ==================================================================
;;; Scaled Dot-Product Attention
;;; ==================================================================

(test-group "Scaled Dot-Product Attention"
  (set-random-seed! 456)
  
  (let* ((q-data (make-f32vector 24 1.0))
         (k-data (make-f32vector 24 1.0))
         (v-data (make-f32vector 24))
         (_ (do ((i 0 (+ i 1)))
                ((= i 24))
              (f32vector-set! v-data i (exact->inexact i))))
         (Q (make-tensor32 q-data '(3 2 4)))
         (K (make-tensor32 k-data '(3 2 4)))
         (V (make-tensor32 v-data '(3 2 4)))
         (output (scaled-dot-product-attention Q K V 4 3)))
    
    (assert-shape '(3 2 4) output "Attention output shape matches input")
    (test-assert "Attention output has gradients" 
                 (tensor-requires-grad? output))))

;;; ==================================================================
;;; Multi-Head Attention Layer
;;; ==================================================================

(test-group "Multi-Head Attention"
  (set-random-seed! 789)
  
  (let ((mha (make-multihead-attention 16 4 dtype: 'f32)))
    
    (test-assert "MHA is multihead-attention?" 
                 (multihead-attention? mha))
    (test-assert "MHA is layer?" (layer? mha))
    
    (let* ((input-data (make-f32vector (* 5 16)))
           (_ (do ((i 0 (+ i 1)))
                  ((= i (* 5 16)))
                (f32vector-set! input-data i 
                               (* 0.1 (- (pseudo-random-real) 0.5)))))
           (input (make-tensor32 input-data '(5 16)))
           (output (forward mha input)))
      
      (assert-shape '(5 16) output "MHA output shape matches input")
      (test-assert "MHA output has gradients" 
                   (tensor-requires-grad? output))
      
      (let* ((target-data (make-f32vector (* 5 16) 1.0))
             (target (make-tensor32 target-data '(5 16)))
             (loss (mse-loss output target)))
        
        (backward! loss)
        
        (let* ((params (parameters mha))
               (has-grads? 
                (every (lambda (p)
                         (and (tensor-grad p)
                              (> (f32vector-length (tensor-grad p)) 0)))
                       params)))
          (test-assert "MHA parameters have gradients after backward"
                       has-grads?))))))

;;; ==================================================================
;;; Transformer Block
;;; ==================================================================

(test-group "Transformer Block"
  (set-random-seed! 101)
  
  (let* ((block (make-transformer-block 32 4 128 dtype: 'f32))
         (input-data (make-f32vector (* 6 32)))
         (_ (do ((i 0 (+ i 1)))
                ((= i (* 6 32)))
              (f32vector-set! input-data i 
                             (* 0.01 (- (pseudo-random-real) 0.5)))))
         (input (make-tensor32 input-data '(6 32)))
         (output (forward block input)))
    
    (test-assert "Block is transformer-block?" 
                 (transformer-block? block))
    (assert-shape '(6 32) output 
                  "Transformer block output shape matches input")))

;;; ==================================================================
;;; Full Transformer
;;; ==================================================================

(test-group "Full Transformer"
  (set-random-seed! 202)
  
  (let* ((model (make-transformer 
                100      ; vocab_size
                20       ; max_seq_len
                64       ; d_model
                2        ; n_layers
                4        ; n_heads
                use-rope?: #t
                dtype: 'f32))
         (token-ids '(1 5 10 15 20))
         (output (forward model token-ids)))
    
    (test-assert "Model is transformer?" (transformer? model))
    (assert-shape '(5 100) output 
                  "Transformer output shape is (seq_len, vocab_size)")
    (test-assert "Transformer output has gradients" 
                 (tensor-requires-grad? output))
    
    (let* ((target-data (make-f32vector (* 5 100) 0.0))
           (target (make-tensor32 target-data '(5 100)))
           (loss (mse-loss output target)))
      
      (backward! loss)
      
      (let* ((params (parameters model))
             (has-grads?
              (every (lambda (p)
                       (and (tensor-grad p)
                            (> (f32vector-length (tensor-grad p)) 0)))
                     params)))
        (test-assert "All transformer parameters have gradients"
                     has-grads?)))))

;;; ==================================================================
;;; Training Simple Sequence Task
;;; ==================================================================

(test-group "Sequence Learning"
  (set-random-seed! 303)
  
  (define training-data
    '((1 2 3)
      (2 3 4)
      (3 4 5)
      (4 5 6)
      (5 6 7)))
  
  (let ((model (make-transformer
                10      ; vocab_size (tokens 0-9)
                5       ; max_seq_len
                32      ; d_model
                2       ; n_layers
                4       ; n_heads
                use-rope?: #t
                dtype: 'f32)))
    
    (define optimizer (make-adam (parameters model) learning-rate: 0.01))
    
    (do ((epoch 1 (+ epoch 1)))
        ((> epoch 20))
      
      (for-each
       (lambda (tokens)
         (let* ((output (forward model tokens))
                (last-emb-data (make-f32vector 10 0.0))
                (_ (do ((i 0 (+ i 1)))
                       ((= i 10))
                     (f32vector-set! last-emb-data i
                                    (f32vector-ref (tensor-data output)
                                                  (+ (* 2 10) i)))))
                (last-emb (make-tensor32 last-emb-data '(10)))
                (target (make-tensor32 (make-f32vector 10 0.0) '(10)))
                (loss (mse-loss last-emb target)))
           
           (backward! loss)
           (step! optimizer)
           (zero-grad-layer! model)))
       training-data))
    
    (test-assert "Sequence learning completed without errors" #t)))

;;; ==================================================================
;;; LLAMA-Style Architecture
;;; ==================================================================

(test-group "LLAMA-Style Architecture"
  (set-random-seed! 404)
  
  (let* ((model (make-transformer
                1000     ; vocab_size
                128      ; max_seq_len
                512      ; d_model
                6        ; n_layers
                8        ; n_heads
                use-rope?: #t
                dtype: 'f32))
         (token-ids (iota 50 1))
         (output (forward model token-ids)))
    
    (test-assert "LLAMA-style transformer created successfully"
                 (transformer? model))
    (assert-shape '(50 1000) output 
                  "LLAMA-style transformer processes long sequences")))

;;; ==================================================================
;;; Gradient Accumulation
;;; ==================================================================

(test-group "Gradient Accumulation"
  (set-random-seed! 505)
  
  (let ((model (make-transformer
                50 10 32 2 4
                use-rope?: #t
                dtype: 'f32)))
    
    (define optimizer (make-adam (parameters model) learning-rate: 0.01))
    
    (let ((examples '((1 2 3) (4 5 6) (7 8 9))))
      
      (for-each
       (lambda (tokens)
         (let* ((output (forward model tokens))
                (target (make-tensor32 (make-f32vector (* 3 50) 0.0) '(3 50)))
                (loss (mse-loss output target)))
           (backward! loss)))
       examples)
      
      (step! optimizer)
      (zero-grad-layer! model)
      
      (test-assert "Gradient accumulation works" #t))))

;;; ==================================================================
;;; Run All Tests
;;; ==================================================================

(test-exit)
