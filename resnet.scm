;; ResNet implementation for Nanograd.
;; Implements ResNet-18, 34, 50, 101, 152.

(module nanograd-resnet
  (
   
   ;; Residual Blocks
   make-residual-block
   make-bottleneck-block
   residual-block?
   bottleneck-block?

   
   ;; ResNet Models
   make-resnet
   resnet?
   make-resnet-18
   make-resnet-34
   make-resnet-50
   make-resnet-101
   make-resnet-152
   make-resnet-cifar

   ;; Helper layers 
   make-downsample
   make-residual-layer

   )
  
  (import
   scheme
   (chicken base)
   (chicken format)
   (srfi 1)
   (srfi 4)
   yasos
   nanograd-autograd
   nanograd-layer
   )

  ;;; ==================================================================
  ;;; Basic Residual Block (for ResNet-18, 34)
  ;;; ==================================================================
  
  (define-predicate residual-block?)
  
  (define (make-residual-block in-channels out-channels
                               #!key
                               (stride 1)
                               (downsample #f)
                               (dtype 'f32)
                               (name "BasicBlock"))
    "Basic residual block with two 3x3 convolutions.
     
     Structure:
       3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> (+) -> ReLU
                                            ^
                                       Identity/Downsample
     
     Args:
       in-channels: Input channels
       out-channels: Output channels
       stride: Stride for first convolution (1 or 2)
       downsample: Optional downsample layer for skip connection
       dtype: Data type"
    
    (let ((conv1 (make-conv2d-layer in-channels out-channels 3
                                    stride: stride
                                    padding: 1
                                    activation: (make-identity)
                                    dtype: dtype
                                    name: (sprintf "~A-Conv1" name)))
          (bn1 (make-batch-norm-2d out-channels dtype: dtype
                                   name: (sprintf "~A-BN1" name)))
          (conv2 (make-conv2d-layer out-channels out-channels 3
                                    stride: 1
                                    padding: 1
                                    activation: (make-identity)
                                    dtype: dtype
                                    name: (sprintf "~A-Conv2" name)))
          (bn2 (make-batch-norm-2d out-channels dtype: dtype
                                   name: (sprintf "~A-BN2" name))))
      
      (object
       ((layer? self) #t)
       ((residual-block? self) #t)
       ((layer-name self) name)
       
       ((forward self x)
        (let ((identity x))
          
          ;; First conv block
          (let* ((out1 (forward conv1 x))
                 (out2 (forward bn1 out1))
                 (out3 (relu out2)))
            
            ;; Second conv block
            (let* ((out4 (forward conv2 out3))
                   (out5 (forward bn2 out4)))
              
              ;; Skip connection
              (let ((out (if downsample
                            (add out5 (forward downsample identity))
                            (add out5 identity))))
                
                ;; Final ReLU
                (relu out))))))
       
       ((parameters self)
        (append (parameters conv1)
                (parameters bn1)
                (parameters conv2)
                (parameters bn2)
                (if downsample
                    (parameters downsample)
                    '())))
       
       ((zero-grad-layer! self)
        (zero-grad-layer! conv1)
        (zero-grad-layer! bn1)
        (zero-grad-layer! conv2)
        (zero-grad-layer! bn2)
        (when downsample
          (zero-grad-layer! downsample)))
       
       ;; Mode control
       ((set-training-mode! self train?)
        (set-training-mode! bn1 train?)
        (set-training-mode! bn2 train?)
        (when downsample
          (set-training-mode! downsample train?)))
       
       ((set-eval-mode! self)
        (set-eval-mode! bn1)
        (set-eval-mode! bn2)
        (when downsample
          (set-eval-mode! downsample))))))

  ;;; ==================================================================
  ;;; Bottleneck Residual Block (for ResNet-50+)
  ;;; ==================================================================
  
  (define-predicate bottleneck-block?)
  
  (define (make-bottleneck-block in-channels out-channels
                                 #!key
                                 (stride 1)
                                 (downsample #f)
                                 (dtype 'f32)
                                 (name "Bottleneck"))
    "Bottleneck residual block with 1x1 -> 3x3 -> 1x1 convolutions.
     
     Structure:
       1x1 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 1x1 conv -> BN -> (+) -> ReLU
                                                                      ^
                                                                 Downsample
     
     Reduces channels by 4x in the middle (bottleneck).
     
     Args:
       in-channels: Input channels
       out-channels: Output channels (expansion * base_channels)
       stride: Stride for 3x3 convolution (1 or 2)
       downsample: Optional downsample layer for skip connection
       dtype: Data type"
    
    (let* ((expansion 4)
           (width (/ out-channels expansion))  ;; Bottleneck width
           
           (conv1 (make-conv2d-layer in-channels width 1
                                    stride: 1
                                    padding: 0
                                    activation: (make-identity)
                                    dtype: dtype
                                    name: (sprintf "~A-Conv1" name)))
           (bn1 (make-batch-norm-2d width dtype: dtype
                                    name: (sprintf "~A-BN1" name)))
           
           (conv2 (make-conv2d-layer width width 3
                                    stride: stride
                                    padding: 1
                                    activation: (make-identity)
                                    dtype: dtype
                                    name: (sprintf "~A-Conv2" name)))
           (bn2 (make-batch-norm-2d width dtype: dtype
                                    name: (sprintf "~A-BN2" name)))
           
           (conv3 (make-conv2d-layer width out-channels 1
                                    stride: 1
                                    padding: 0
                                    activation: (make-identity)
                                    dtype: dtype
                                    name: (sprintf "~A-Conv3" name)))
           (bn3 (make-batch-norm-2d out-channels dtype: dtype
                                    name: (sprintf "~A-BN3" name))))
      
      (object
       ((layer? self) #t)
       ((bottleneck-block? self) #t)
       ((layer-name self) name)
       
       ((forward self x)
        (let ((identity x))
          
          ;; 1x1 conv (reduce)
          (let* ((out1 (forward conv1 x))
                 (out2 (forward bn1 out1))
                 (out3 (relu out2)))
            
            ;; 3x3 conv (process)
            (let* ((out4 (forward conv2 out3))
                   (out5 (forward bn2 out4))
                   (out6 (relu out5)))
              
              ;; 1x1 conv (expand)
              (let* ((out7 (forward conv3 out6))
                     (out8 (forward bn3 out7)))
                
                ;; Skip connection
                (let ((out (if downsample
                              (add out8 (forward downsample identity))
                              (add out8 identity))))
                  
                  ;; Final ReLU
                  (relu out)))))))
       
       ((parameters self)
        (append (parameters conv1) (parameters bn1)
                (parameters conv2) (parameters bn2)
                (parameters conv3) (parameters bn3)
                (if downsample (parameters downsample) '())))
       
       ((zero-grad-layer! self)
        (zero-grad-layer! conv1) (zero-grad-layer! bn1)
        (zero-grad-layer! conv2) (zero-grad-layer! bn2)
        (zero-grad-layer! conv3) (zero-grad-layer! bn3)
        (when downsample (zero-grad-layer! downsample)))
       
       ((set-training-mode! self train?)
        (set-training-mode! bn1 train?)
        (set-training-mode! bn2 train?)
        (set-training-mode! bn3 train?)
        (when downsample (set-training-mode! downsample train?)))
       
       ((set-eval-mode! self)
        (set-eval-mode! bn1)
        (set-eval-mode! bn2)
        (set-eval-mode! bn3)
        (when downsample (set-eval-mode! downsample))))))

  ;;; ==================================================================
  ;;; Helper: Downsample Layer
  ;;; ==================================================================
  
  (define (make-downsample in-channels out-channels stride dtype)
    "Create downsample layer for skip connection when dimensions change."
    (let ((conv (make-conv2d-layer in-channels out-channels 1
                                   stride: stride
                                   padding: 0
                                   activation: (make-identity)
                                   dtype: dtype
                                   name: "Downsample-Conv"))
          (bn (make-batch-norm-2d out-channels dtype: dtype
                                  name: "Downsample-BN")))
      
      (object
       ((layer? self) #t)
       ((layer-name self) "Downsample")
       
       ((forward self x)
        (let* ((out (forward conv x))
               (out (forward bn out)))
          out))
       
       ((parameters self)
        (append (parameters conv) (parameters bn)))
       
       ((zero-grad-layer! self)
        (zero-grad-layer! conv)
        (zero-grad-layer! bn))
       
       ((set-training-mode! self train?)
        (set-training-mode! conv train?)
        (set-training-mode! bn train?))

       ((set-eval-mode! self)
        (set-eval-mode! conv)
        (set-eval-mode! bn)))))

  ;;; ==================================================================
  ;;; Helper: Residual Layer
  ;;; ==================================================================
  
  (define (make-residual-layer block-fn in-channels out-channels num-blocks
                               #!key (stride 1) (dtype 'f32) (name "Layer"))
    "Create a residual layer with multiple blocks."
    
    (let* ((downsample (if (or (not (= stride 1))
                               (not (= in-channels out-channels)))
                           (make-downsample in-channels out-channels stride dtype)
                           #f))
           
           ;; First block (may downsample)
           (first-block (block-fn in-channels out-channels
                                  stride: stride
                                  downsample: downsample
                                  dtype: dtype
                                  name: (sprintf "~A-Block0" name)))
           
           ;; Remaining blocks (no downsampling)
           (rest-blocks (map (lambda (i)
                               (block-fn out-channels out-channels
                                         stride: 1
                                         downsample: #f
                                         dtype: dtype
                                         name: (sprintf "~A-Block~A" name (+ i 1))))
                            (iota (- num-blocks 1))))
           
           (all-blocks (cons first-block rest-blocks)))
      
      (object
       ((layer? self) #t)
       ((layer-name self) name)
       
       ((forward self x)
        (fold (lambda (block x) (forward block x))
              x
              all-blocks))
       
       ((parameters self)
        (append-map parameters all-blocks))
       
       ((zero-grad-layer! self)
        (for-each zero-grad-layer! all-blocks))
       
       ;; The downsample layer belongs to the first block, which
       ;; forwards mode changes to it.
       ((set-training-mode! self train?)
        (for-each (lambda (block) (set-training-mode! block train?))
                  all-blocks))
       
       ((set-eval-mode! self)
        (for-each set-eval-mode! all-blocks)))))

  ;;; ==================================================================
  ;;; Complete ResNet Model
  ;;; ==================================================================


  (define (compute-min-input-size-for-resnet variant)
    "Compute minimum input size for ResNet variants.
   
     ResNet-18/34/50/101/152 standard architecture:
     - Conv1: 7x7, stride=2, padding=3
     - MaxPool: 3x3, stride=2
     - Layer1: stride=1
     - Layer2: stride=2
     - Layer3: stride=2
     - Layer4: stride=2
   
     Total spatial reduction: 2 * 2 * 2 * 2 * 2 = 32x
   
     To get 1x1 output from Layer4, need at least 32x32 input.
     For safe 2x2 or larger output: 64x64 minimum."
  
    (case variant
      ((resnet-18 resnet-34 resnet-50 resnet-101 resnet-152)
       '((minimum . 32)     ;; Absolute minimum (produces 1x1 after Layer3)
         (recommended . 64) ;; Recommended (produces 2x2 after Layer4)
         (standard . 224))) ;; ImageNet standard
      (else
       '((minimum . 32)
         (recommended . 64)
         (standard . 224)))))

  (define (validate-resnet-input-size input-shape variant)
    "Validate input size is appropriate for ResNet variant."
  
    (let* ((ndim (length input-shape))
           (spatial-dims (cond
                          ((= ndim 3) (cdr input-shape))
                          ((= ndim 4) (cddr input-shape))
                          (else #f)))
           (size-info (compute-min-input-size-for-resnet variant)))
      
      (when spatial-dims
        (let ((H (car spatial-dims))
              (W (cadr spatial-dims))
              (min-size (cdr (assq 'minimum size-info)))
              (rec-size (cdr (assq 'recommended size-info))))
          
          ;; Error if below minimum
          (when (or (< H min-size) (< W min-size))
            (error 'resnet-forward
                   (format #f "Input too small for ~A!
  Input size: ~Ax~A
  Minimum size: ~Ax~A (produces 1x1 at Layer3)
  Recommended size: ~Ax~A (produces 2x2 at Layer4)
  Standard size: 224x224 (ImageNet)
  
  Spatial dimensions collapse during forward pass.
  Use larger images or modify architecture for CIFAR-10 (32x32)."
                           variant H W
                           min-size min-size
                           rec-size rec-size)))
          
          ;; Warning if below recommended
          (when (and (>= H min-size) (< H rec-size))
            (printf "  Warning: Input size ~Ax~A is below recommended ~Ax~A for ~A\n"
                    H W rec-size rec-size variant)
            (printf "  Output from Layer4 will be very small (1x1 or 2x2)\n"))))))

  
  (define-predicate resnet?)
  
  (define (make-resnet block-fn layers num-classes
                      #!key
                      (dtype 'f32)
                      (name "ResNet"))
    
    "Create a complete ResNet model.
     
     Args:
       block-fn: Block constructor (make-residual-block or make-bottleneck-block)
       layers: List of number of blocks per layer (e.g., [2 2 2 2] for ResNet-18)
       num-classes: Number of output classes
       dtype: Data type
       
     Architecture:
       Conv1 (7x7, stride 2) -> BN -> ReLU -> MaxPool
       -> Layer1 (64 channels)
       -> Layer2 (128 channels, stride 2)
       -> Layer3 (256 channels, stride 2)
       -> Layer4 (512 channels, stride 2)
       -> Global Avg Pool -> FC"
    
    (let* ((expansion (if (eq? block-fn make-bottleneck-block) 4 1))
           
           ;; Initial convolution
           (conv1 (make-conv2d-layer 3 64 7
                                    stride: 2
                                    padding: 3
                                    activation: (make-identity)
                                    dtype: dtype
                                    name: "Conv1"))
           (bn1 (make-batch-norm-2d 64 dtype: dtype name: "BN1"))
           
           ;; Residual layers
           (layer1 (make-residual-layer block-fn 64 (* 64 expansion) (car layers)
                                        stride: 1 dtype: dtype name: "Layer1"))
           (layer2 (make-residual-layer block-fn (* 64 expansion) (* 128 expansion)
                                        (cadr layers)
                                        stride: 2 dtype: dtype name: "Layer2"))
           (layer3 (make-residual-layer block-fn (* 128 expansion) (* 256 expansion)
                                        (caddr layers)
                                        stride: 2 dtype: dtype name: "Layer3"))
           (layer4 (make-residual-layer block-fn (* 256 expansion) (* 512 expansion)
                                        (cadddr layers)
                                        stride: 2 dtype: dtype name: "Layer4"))
           
           ;; Final classifier
           (fc (make-dense-layer (* 512 expansion) num-classes
                                 activation: (make-identity)
                                 dtype: dtype
                                 name: "FC")))
      
      (object
       ((layer? self) #t)
       ((resnet? self) #t)
       ((layer-name self) name)
       
       ((forward self x)
        "Forward pass through ResNet.
         
         Input: (3, H, W) where H, W >= 32
         Output: (num_classes,)"

        (validate-resnet-input-size (tensor-shape x) name)

        
        ;; Initial conv
        (let* ((x (forward conv1 x))
               (x (forward bn1 x))
               (x (relu x))
               (x (maxpool2d x 3 stride: 2)))  ;; Using existing maxpool2d
          
          ;; Residual layers
          (let* ((x (forward layer1 x))
                 (x (forward layer2 x))
                 (x (forward layer3 x))
                 (x (forward layer4 x)))
            
            ;; Global average pooling
            (let ((x (global-avg-pool2d x)))
              ;; Classifier
              (forward fc x)))))
       
       ((parameters self)
        (append (parameters conv1)
                (parameters bn1)
                (parameters layer1)
                (parameters layer2)
                (parameters layer3)
                (parameters layer4)
                (parameters fc)))
       
       ((zero-grad-layer! self)
        (zero-grad-layer! conv1)
        (zero-grad-layer! bn1)
        (zero-grad-layer! layer1)
        (zero-grad-layer! layer2)
        (zero-grad-layer! layer3)
        (zero-grad-layer! layer4)
        (zero-grad-layer! fc))
       
       ((set-training-mode! self train?)
        (set-training-mode! bn1 train?)
        (set-training-mode! layer1 train?)
        (set-training-mode! layer2 train?)
        (set-training-mode! layer3 train?)
        (set-training-mode! layer4 train?))
       
       ((set-eval-mode! self)
        (set-eval-mode! bn1)
        (set-eval-mode! layer1)
        (set-eval-mode! layer2)
        (set-eval-mode! layer3)
        (set-eval-mode! layer4)))))

  ;;; ==================================================================
  ;;; Standard ResNet Configurations
  ;;; ==================================================================
  
  (define (make-resnet-18 num-classes #!key (dtype 'f32))
    "ResNet-18: [2, 2, 2, 2] basic blocks"
    (make-resnet make-residual-block '(2 2 2 2) num-classes
                dtype: dtype name: "ResNet-18"))
  
  (define (make-resnet-34 num-classes #!key (dtype 'f32))
    "ResNet-34: [3, 4, 6, 3] basic blocks"
    (make-resnet make-residual-block '(3 4 6 3) num-classes
                dtype: dtype name: "ResNet-34"))
  
  (define (make-resnet-50 num-classes #!key (dtype 'f32))
    "ResNet-50: [3, 4, 6, 3] bottleneck blocks"
    (make-resnet make-bottleneck-block '(3 4 6 3) num-classes
                dtype: dtype name: "ResNet-50"))
  
  (define (make-resnet-101 num-classes #!key (dtype 'f32))
    "ResNet-101: [3, 4, 23, 3] bottleneck blocks"
    (make-resnet make-bottleneck-block '(3 4 23 3) num-classes
                dtype: dtype name: "ResNet-101"))
  
  (define (make-resnet-152 num-classes #!key (dtype 'f32))
    "ResNet-152: [3, 8, 36, 3] bottleneck blocks"
    (make-resnet make-bottleneck-block '(3 8 36 3) num-classes
                dtype: dtype name: "ResNet-152"))

  (define (make-resnet-cifar stages widths num-classes
                             #!key
                             (in-channels 3)
                             (dtype 'f32)
                             (name "ResNet-CIFAR"))
    "ResNet with basic blocks for small (e.g. 32x32) images.

     Architecture:
       Conv (3x3, stride 1, padding 1) -> BN -> ReLU
       -> one residual layer per entry of STAGES; layer i has
          (list-ref STAGES i) blocks and (list-ref WIDTHS i) channels,
          and every layer after the first starts with stride 2
       -> Global Avg Pool -> FC

     There is no max pooling, so the spatial size is divided by
     2^(length STAGES - 1).  Input: (C, H, W) or (N, C, H, W)."

    (let* ((w0 (car widths))
           (conv1 (make-conv2d-layer in-channels w0 3
                                    stride: 1
                                    padding: 1
                                    activation: (make-identity)
                                    dtype: dtype
                                    name: "Conv1"))
           (bn1 (make-batch-norm-2d w0 dtype: dtype name: "BN1"))
           (layers (let loop ((ss stages) (ws widths) (in w0) (i 0) (acc '()))
                     (if (null? ss)
                         (reverse acc)
                         (loop (cdr ss) (cdr ws) (car ws) (+ i 1)
                               (cons (make-residual-layer
                                      make-residual-block in (car ws) (car ss)
                                      stride: (if (= i 0) 1 2)
                                      dtype: dtype
                                      name: (sprintf "Layer~A" (+ i 1)))
                                     acc)))))
           (fc (make-dense-layer (last widths) num-classes
                                 activation: (make-identity)
                                 dtype: dtype
                                 name: "FC")))

      (object
       ((layer? self) #t)
       ((resnet? self) #t)
       ((layer-name self) name)

       ((forward self x)
        (let* ((x (relu (forward bn1 (forward conv1 x))))
               (x (fold (lambda (layer x) (forward layer x)) x layers)))
          (forward fc (global-avg-pool2d x))))

       ((parameters self)
        (append (parameters conv1)
                (parameters bn1)
                (append-map parameters layers)
                (parameters fc)))

       ((zero-grad-layer! self)
        (zero-grad-layer! conv1)
        (zero-grad-layer! bn1)
        (for-each zero-grad-layer! layers)
        (zero-grad-layer! fc))

       ((set-training-mode! self train?)
        (set-training-mode! bn1 train?)
        (for-each (lambda (l) (set-training-mode! l train?)) layers))

       ((set-eval-mode! self)
        (set-eval-mode! bn1)
        (for-each set-eval-mode! layers)))))

) ;; end module
