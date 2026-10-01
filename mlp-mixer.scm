;; MLP-Mixer Implementation for Nanograd
;; Based on "MLP-Mixer: An all-MLP Architecture for Vision" (Tolstikhin et al., 2021)

(module nanograd-mlp-mixer
  (
   ;; Core components
   make-mixer-mlp
   make-token-mixing-mlp
   make-channel-mixing-mlp
   
   ;; Block and model
   make-mlp-mixer-block
   make-mlp-mixer
   mlp-mixer?
   get-blocks
   num-blocks

   ;; Normalization layers
   make-rmsnorm-layer
   make-layer-norm
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

  (define-predicate mlp-mixer?)
  (define-operation (get-blocks mlp))
  (define-operation (num-blocks mlp))
  
  
  ;;; ==================================================================
  ;;; Core MLP Building Block
  ;;; ==================================================================

  (define (make-mixer-mlp input-dim hidden-dim #!key (dtype 'f32) (name "MixerMLP"))
    "Create a simple two-layer MLP with GELU activation.
     
     Architecture: Linear(input-dim -> hidden-dim) -> GELU -> Linear(hidden-dim -> input-dim)
     
     This is the basic building block used in both token and channel mixing.
     
     Parameters:
       input-dim  : Input dimension
       hidden-dim : Hidden layer dimension (typically 4x input-dim for channel mixing)
       dtype      : Data type (f32 or f64)
       name       : Layer name for debugging
     
     Returns: YASOS object with forward method
     
     Example:
       (define mlp (make-mixer-mlp 128 512 dtype: 'f32))
       (define output (forward mlp input))
     "
    (let ((layer1 (make-dense-layer input-dim hidden-dim dtype: dtype name: (format "~A-Dense1" name)))
          (layer2 (make-dense-layer hidden-dim input-dim dtype: dtype name: (format "~A-Dense2" name)))
          (activation (make-gelu)))
      
      (object
       ;; Forward method
       ((forward self input)
        (let* ((h1 (forward layer1 input))
               (h2 (activation-forward activation h1))
               (output (forward layer2 h2)))
          output))
       
       ;; Parameters method
       ((parameters self)
        (append (parameters layer1)
                (parameters layer2)))
       
       ;; Name method
       ((layer-name self) name)
       
       ;; Standard layer protocol
       ((layer? self) #t)
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       
       ((zero-grad-layer! self)
        (zero-grad-layer! layer1)
        (zero-grad-layer! layer2))
        

       )))

  ;;; ==================================================================
  ;;; Token Mixing MLP
  ;;; ==================================================================

  (define (make-token-mixing-mlp seq-len hidden-dim #!key (expansion-factor 4) (dtype 'f32) (name "TokenMixing"))
    "Create token mixing MLP that operates across sequence positions.
     
     Token mixing mixes information ACROSS different tokens (positions) in the sequence,
     allowing tokens to communicate. This is the spatial mixing operation.
     
     Operation:
       1. Transpose: (seq-len, hidden-dim) -> (hidden-dim, seq-len)
       2. Apply MLP across seq-len dimension
       3. Transpose back: (hidden-dim, seq-len) -> (seq-len, hidden-dim)
     
     The MLP learns to mix tokens: each position can see and combine information
     from all other positions.
     
     Parameters:
       seq-len          : Sequence length (number of tokens)
       hidden-dim       : Hidden dimension per token
       expansion-factor : Hidden layer expansion (typically 4)
       dtype            : Data type
       name             : Layer name
     
     Returns: YASOS object with forward method
     
     Example (TRM usage):
       ;; For a 6x6 grid flattened to 36 tokens
       (define token-mixer (make-token-mixing-mlp 36 128))
       (define mixed (forward token-mixer input))  ; input: (36, 128)
     "
    (let* ((mlp-hidden-dim (* expansion-factor seq-len))
           (mlp (make-mixer-mlp seq-len mlp-hidden-dim dtype: dtype name: (format "~A-MLP" name))))
      
      (object
       ;; Forward method
       ((forward self input)
        ;; Input: (seq-len, hidden-dim)
        ;; Transpose to (hidden-dim, seq-len) to work on tokens dimension
        (let* ((transposed (transpose-tensor input '(1 0)))  ; (hidden-dim, seq-len)
               ;; Apply MLP along the last dimension (seq-len)
               ;; Each row (feature) now mixes across all tokens
               (mixed (forward mlp transposed))       ; (hidden-dim, seq-len)
               ;; Transpose back
               (output (transpose-tensor mixed '(1 0))))     ; (seq-len, hidden-dim)
          output))
       
       ;; Parameters
       ((parameters self)
        (parameters mlp))
       
       ;; Name
       ((layer-name self) name)
       
       ;; Layer protocol
       ((layer? self) #t)
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))

       ((zero-grad-layer! self)
        (zero-grad-layer! mlp))

       )))

  ;;; ==================================================================
  ;;; Channel Mixing MLP
  ;;; ==================================================================

  (define (make-channel-mixing-mlp hidden-dim #!key (expansion-factor 4) (dtype 'f32) (name "ChannelMixing"))
    "Create channel mixing MLP that operates across feature dimensions.
     
     Channel mixing mixes information WITHIN each token across different features.
     This is the feature mixing operation applied independently to each position.
     
     Operation:
       Apply MLP directly to (seq-len, hidden-dim) tensor.
       Each token's features are mixed independently.
     
     The MLP learns feature combinations: each position's features interact
     but different positions don't communicate (that's token mixing's job).
     
     Parameters:
       hidden-dim       : Hidden dimension (feature size)
       expansion-factor : Hidden layer expansion (typically 4)
       dtype            : Data type
       name             : Layer name
     
     Returns: YASOS object with forward method
     
     Example:
       (define channel-mixer (make-channel-mixing-mlp 128))
       (define mixed (forward channel-mixer input))  ; input: (seq-len, 128)
     "
    (let* ((mlp-hidden-dim (* expansion-factor hidden-dim))
           (mlp (make-mixer-mlp hidden-dim mlp-hidden-dim dtype: dtype name: (format "~A-MLP" name))))
      
      (object

       ;; Forward method
       ((forward self input)
        ;; Input: (seq-len, hidden-dim)
        ;; Apply MLP directly - each row processed independently
        (forward mlp input))  ; (seq-len, hidden-dim)
       
       ;; Parameters
       ((parameters self)
        (parameters mlp))
       
       ;; Name
       ((layer-name self) name)
       
       ;; Layer protocol
       ((layer? self) #t)
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))

       ((zero-grad-layer! self)
        (zero-grad-layer! mlp))

       )))

  ;;; ==================================================================
  ;;; MLP-Mixer Block
  ;;; ==================================================================

  (define (make-mlp-mixer-block seq-len 
                                hidden-dim
                                #!key 
                                (token-expansion 4)
                                (channel-expansion 4)
                                (use-rmsnorm #t)
                                (dtype 'f32)
                                (name "MixerBlock"))
    "Create a single MLP-Mixer block with both token and channel mixing.
     
     Architecture (Pre-Norm variant, better for TRM):
       x -> RMSNorm -> TokenMixing -> Add(x, .) -> RMSNorm -> ChannelMixing -> Add(., .)
     
     This is the core building block. A full MLP-Mixer stacks multiple blocks.
     
     Why Pre-Norm (RMSNorm before mixing):
       - More stable training for deep networks
       - Better for recursive models like TRM
       - Used in modern transformers (GPT, LLaMA)
     
     Parameters:
       seq-len           : Sequence length (number of tokens/patches)
       hidden-dim        : Hidden dimension (features per token)
       token-expansion   : Expansion factor for token mixing MLP (default 4)
       channel-expansion : Expansion factor for channel mixing MLP (default 4)
       use-rmsnorm       : Use RMSNorm (True) or LayerNorm (False)
       dtype             : Data type
       name              : Block name
     
     Returns: YASOS object with forward method
     
     Example - TRM with 6x6 grid:
       (define block (make-mlp-mixer-block 36 128  ; 36 patches, 128 features
                                          token-expansion: 4
                                          channel-expansion: 4))
       (define output (forward block input))
     "
    (let ((token-mixing (make-token-mixing-mlp seq-len hidden-dim 
                                               expansion-factor: token-expansion
                                               dtype: dtype
                                               name: (format "~A-TokenMix" name)))
          (channel-mixing (make-channel-mixing-mlp hidden-dim
                                                   expansion-factor: channel-expansion
                                                   dtype: dtype
                                                   name: (format "~A-ChannelMix" name)))
          (norm1 (if use-rmsnorm
                     (make-rmsnorm-layer `(,hidden-dim) dtype: dtype name: (format "~A-Norm1" name))
                     (make-layer-norm `(,hidden-dim) dtype: dtype name: (format "~A-Norm1" name))))
          (norm2 (if use-rmsnorm
                     (make-rmsnorm-layer `(,hidden-dim) dtype: dtype name: (format "~A-Norm2" name))
                     (make-layer-norm `(,hidden-dim) dtype: dtype name: (format "~A-Norm2" name)))))
      
      (object
       ;; Forward method with skip connections
       ((forward self input)
        ;; Token mixing path
        (let* ((normed1 (forward norm1 input))
               (token-mixed (forward token-mixing normed1))
               (skip1 (add input token-mixed))  ; Skip connection
               
               ;; Channel mixing path
               (normed2 (forward norm2 skip1))
               (channel-mixed (forward channel-mixing normed2))
               (output (add skip1 channel-mixed)))  ; Skip connection
          output))
       
       ;; Parameters
       ((parameters self)
        (append (parameters token-mixing)
                (parameters channel-mixing)
                (parameters norm1)
                (parameters norm2)))
       
       ;; Name
       ((layer-name self) name)
       
       ;; Layer protocol
       ((layer? self) #t)
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))

       ((zero-grad-layer! self)
        (zero-grad-layer! token-mixing)
        (zero-grad-layer! channel-mixing)
        (zero-grad-layer! norm1)
        (zero-grad-layer! norm2)
        )

       )))

  ;;; ==================================================================
  ;;; Full MLP-Mixer Model
  ;;; ==================================================================

  (define (make-mlp-mixer seq-len 
                          hidden-dim
                          n-blocks
                          #!key 
                          (token-expansion 4)
                          (channel-expansion 4)
                          (use-rmsnorm #t)
                          (dtype 'f32)
                          (name "MLPMixer"))
    "Create a full MLP-Mixer model with multiple blocks.
     
     Stacks n-blocks MLP-Mixer blocks sequentially. This is the complete
     MLP-Mixer architecture from the paper.
     
     Architecture:
       Input -> Block1 -> Block2 -> ... -> BlockN -> Output
     
     Each block does: RMSNorm -> TokenMix -> Skip -> RMSNorm -> ChannelMix -> Skip
     
     Parameters:
       seq-len           : Sequence length (number of tokens)
       hidden-dim        : Hidden dimension (features per token)
       n-blocks          : Number of mixer blocks to stack
       token-expansion   : Token mixing expansion factor
       channel-expansion : Channel mixing expansion factor
       use-rmsnorm       : Use RMSNorm (True) or LayerNorm (False)
       dtype             : Data type
       name              : Model name
     
     Returns: YASOS object with forward method
     
     Example - TRM network (2 blocks):
       (define mixer (make-mlp-mixer 
                       36        ; 6x6 grid = 36 patches
                       128       ; 128 features per patch
                       2         ; 2 blocks (TRM paper)
                       token-expansion: 4
                       channel-expansion: 4
                       use-rmsnorm: #t))
       
       (define output (forward mixer input))
     
     Example - Deeper model (4 blocks):
       (define mixer (make-mlp-mixer 36 128 4))
     "
    (let ((blocks (map (lambda (i)
                        (make-mlp-mixer-block seq-len hidden-dim
                                             token-expansion: token-expansion
                                             channel-expansion: channel-expansion
                                             use-rmsnorm: use-rmsnorm
                                             dtype: dtype
                                             name: (format "~A-Block~A" name i)))
                      (iota n-blocks))))
      
      (object

       ((mlp-mixer? self) #t)
       
       ;; Forward method - sequential application
       ((forward self input)
        (fold (lambda (block acc)
                (forward block acc))
              input
              blocks))
       
       ;; Parameters - all blocks
       ((parameters self)
        (apply append (map parameters blocks)))
       
       ;; Name
       ((layer-name self) name)
       
       ;; Layer protocol
       ((layer? self) #t)
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))

       ((zero-grad-layer! self)
        (for-each (lambda (block)
                    (zero-grad-layer! block)) blocks))
      
       ;; Access individual blocks
       ((get-blocks self) blocks)
       
       ;; Block count
       ((num-blocks self) n-blocks)
       )
      ))


  ;;; ==================================================================
  ;;; LayerNorm Layer Wrapper
  ;;; ==================================================================
  (define (make-layer-norm normalized-shape #!key (eps 1e-6) (dtype 'f32) (name "LayerNorm"))
    "Create a LayerNorm layer over the last axis.

     y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta, per row,
     with learnable gamma (initialized to 1) and beta (initialized to 0)."
    (let ((gamma (make-parameter-tensor normalized-shape 1.0 dtype: dtype))
          (beta (make-parameter-tensor normalized-shape 0.0 dtype: dtype)))
      (object
       ((forward self input)
        (layer-norm input gamma beta epsilon: eps))

       ((zero-grad-layer! self)
        (zero-grad! gamma)
        (zero-grad! beta))

       ((parameters self)
        (list gamma beta))

       ((layer-name self) name)

       ((layer? self) #t)
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void)))))

  ;;; ==================================================================
  ;;; RMSNorm Layer Wrapper
  ;;; ==================================================================

  (define (make-rmsnorm-layer normalized-shape #!key (eps 1e-6) (dtype 'f32) (name "RMSNorm"))
    "Create RMSNorm layer object that wraps the rmsnorm operation.
     
     RMSNorm (Root Mean Square Normalization) is simpler and faster than LayerNorm,
     and works better for many modern architectures (used in LLaMA, Mistral, etc.).
     
     Formula: y = x / RMS(x) * g
     where RMS(x) = sqrt(mean(x^2) + eps)
     
     Parameters:
       normalized-shape : Shape of the normalization dimension
       eps              : Small constant for numerical stability
       dtype            : Data type
       name             : Layer name
     
     Returns: YASOS object with forward method
     "
    ;; Create learnable scale parameter (g in formula)
    (let ((scale (make-parameter-tensor normalized-shape 1.0 dtype: dtype name: (format "~A-scale" name))))
             
      (object
       ;; Forward
       ((forward self input)
        (let ((normed (rmsnorm input scale epsilon: eps)))
          normed))
       
       ;; Parameters
       ((parameters self)
        (list scale))

       ((zero-grad-layer! self)
        (zero-grad! scale))
       
       ;; Name
       ((layer-name self) name)
       
       ;; Layer protocol
       ((layer? self) #t)
       ((set-training-mode! self train?) (void))
       ((set-eval-mode! self) (void))
       )
      ))

  ;;; ==================================================================
  ;;; Helper: Create Parameter Tensor
  ;;; ==================================================================

  (define (make-parameter-tensor shape init-value #!key (dtype 'f32) (name "param"))
    "Create a learnable parameter tensor initialized to a constant.
     
     For scale parameters in normalization layers.
     "
    (let* ((size (apply * shape))
           (data (case dtype
                   ((f32) (make-f32vector size init-value))
                   ((f64) (make-f64vector size init-value))
                   (else (error "Unsupported dtype" dtype)))))
      (make-base-tensor data shape dtype #t)))

) ;; end module
