# Dynamic3DGS Methodology

## 1. Problem Formalization

Given a monocular video sequence $\{I_t, C_t\}_{t=1}^T$, where $I_t$ is the image at time $t$ and $C_t$ represents the camera parameters (intrinsics and extrinsics), our goal is to learn a dynamic 3D scene representation $G = \{g_i\}_{i=1}^N$ that can render novel views $\hat{I}(t, C)$ for any time step $t$ and camera pose $C$.

## 2. Deformable Gaussian Representation

Each Gaussian $g_i$ in our representation consists of:

### Static Parameters (shared across all time steps)
- **Position**: $\mu_i \in \mathbb{R}^3$ - Initial 3D position
- **Covariance**: $\Sigma_i \in \mathbb{R}^{3 \times 3}$ - Covariance matrix
- **Opacity**: $\alpha_i \in [0, 1]$ - Surface opacity
- **Spherical Harmonics**: $\mathbf{c}_i \in \mathbb{R}^{3 \times 16}$ - SH coefficients for lighting

### Dynamic Parameters (time-dependent)
- **Position Offset**: $\Delta\mu_i(t) = F_\mu(\mu_i, t; \theta_\mu)$
- **Covariance Deformation**: $\Delta\Sigma_i(t) = F_\Sigma(\Sigma_i, t; \theta_\Sigma)$
- **Opacity Modulation**: $\alpha_i'(t) = \alpha_i \cdot M_i(t)$

Where $F_\mu$ and $F_\Sigma$ are learned deformation fields parameterized by neural networks.

## 3. Deformation Field Network Architecture

We employ SIREN (Sinusoidal Representation Networks) for the deformation field due to their ability to represent continuous, high-frequency functions:

$$
F(x,t) = W_L \circ \sin(W_{L-1} \circ \sin(\cdots \circ \sin(W_1[x;t] + b_1)\cdots)) + b_L
$$

**Network Structure:**
- **Input**: $(x, y, z, t) \in \mathbb{R}^4$
- **Output**: $(\Delta x, \Delta y, \Delta z) \in \mathbb{R}^3$
- **Hidden Layers**: [256, 256, 128, 128]
- **Activation**: Sine with proper weight initialization
- **Layer Normalization**: Applied after each layer

**Weight Initialization:**
- First layer: $W \sim \mathcal{N}(0, \sqrt{6/d_{in}})$
- Hidden layers: $W \sim \mathcal{N}(0, \sqrt{6/d_{in}})$

## 4. Rendering Pipeline

### 4.1 Projection to Image Space
For each Gaussian at time $t$:
1. Apply deformation: $\mu_i'(t) = \mu_i + F_\mu(\mu_i, t)$
2. Transform to camera coordinates: $\mu_i^{cam} = R^T(\mu_i' - t)$
3. Project to image plane: $u_i = f_x \cdot \frac{x}{z} + c_x$

### 4.2 Depth Sorting
Gaussians are sorted by depth in the forward-facing direction:
$$
\text{sort\_indices} = \arg\max_{\text{depths}}
$$

### 4.3 Alpha Compositing
Using the alpha blending equation:
$$
C_{out} = C_{in} \cdot \alpha_{in} + C_{prev} \cdot (1 - \alpha_{in})
$$

### 4.4 Spherical Harmonics Lighting
Lighting contribution from spherical harmonics coefficients:
$$
\mathbf{L}_i = \sum_{l=0}^{3} \sum_{m=-l}^{l} \mathbf{c}_{i,lm} Y_{lm}(\omega_i)
$$

## 5. Loss Functions

### 5.1 Reconstruction Loss
Combination of multiple perceptual metrics:
$$
\mathcal{L}_{recon} = \lambda_{L1} \mathcal{L}_{L1} + \lambda_{SSIM} \mathcal{L}_{SSIM} + \lambda_{DSSIM} \mathcal{L}_{DSSIM}
$$

Where:
- **L1 Loss**: $\mathcal{L}_{L1} = \| \hat{I} - I \|_1$
- **SSIM Loss**: $\mathcal{L}_{SSIM} = 1 - SSIM(\hat{I}, I)$
- **D-SSIM Loss**: Differentiable version of SSIM

### 5.2 Temporal Consistency Loss
Encourages smooth temporal evolution:
$$
\mathcal{L}_{temp} = \lambda_{flow} \mathcal{L}_{flow} + \lambda_{smooth} \mathcal{L}_{smooth} + \lambda_{rigid} \mathcal{L}_{rigid}
$$

**Components:**
- **Optical Flow Consistency**: Warp previous frame using optical flow and compare
- **Motion Smoothness**: Penalize rapid changes in deformation field
- **Rigidity Constraint**: Encourage small motion in static regions

### 5.3 Regularization Losses
Control model complexity and prevent degenerate solutions:
$$
\mathcal{L}_{reg} = \lambda_{opp} \mathcal{L}_{opacity} + \lambda_{sh} \mathcal{L}_{sh} + \lambda_{cov} \mathcal{L}_{cov} + \lambda_{deform} \mathcal{L}_{deform}
$$

## 6. Vector Quantization

To enable discrete representation learning, we use vector quantization on the deformed positions:

$$
\hat{\mu}_i(t) = \text{Quantize}(F_\mu(\mu_i, t))
$$

This encourages the network to learn a compact codebook of meaningful deformation patterns.

## 7. Training Strategy

### 7.1 Optimization Schedule
- **Warmup Phase**: Linear increase in learning rate over first 50 epochs
- **Main Phase**: Cosine annealing with restarts
- **Deformation Field Learning Rate**: 0.1× base learning rate

### 7.2 Multi-scale Training
- **Initial Stage**: Train with lower resolution images (256×256)
- **Progressive Refinement**: Increase resolution gradually
- **Final Stage**: Full resolution training (512×512)

### 7.3 Memory Management
- **Gaussian Pruning**: Remove low-opacity Gaussians periodically
- **Gaussian Splitting**: Split large Gaussians when needed
- **Hierarchical Pool**: Maintain Gaussians at multiple scales

## 8. Implementation Details

### 8.1 Software Stack
- **Framework**: PyTorch 2.0+
- **Rendering**: Custom CUDA kernels for efficiency
- **Data Loading**: Multi-threaded data pipeline
- **Distributed Training**: DDP support for multi-GPU training

### 8.2 Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (minimum 16GB VRAM)
- **CPU**: 8+ cores for data preprocessing
- **Memory**: 32GB+ RAM for large datasets
- **Storage**: SSD for fast data access

### 8.3 Performance Optimizations
- **Mixed Precision**: FP16 training with FP32 master weights
- **Gradient Checkpointing**: Trade memory for computation
- **Asynchronous Data Loading**: Overlap data loading with computation
- **Kernel Fusion**: Combine multiple operations into single kernels

## 9. Evaluation Protocol

### 9.1 Metrics
- **Image Quality**: PSNR, SSIM, LPIPS
- **Temporal Quality**: FVD, Temporal SSIM
- **Geometry Quality**: Chamfer Distance, Normal Consistency
- **Efficiency**: FPS, Memory Usage, Training Time

### 9.2 Baselines
Comparison with state-of-the-art methods:
- **Static 3DGS**: Original Gaussian splatting
- **D-NeRF**: Neural radiance fields for dynamics
- **HyperNeRF**: Hypernetwork approach to dynamics
- **TiAnGang**: Recent deformable representation

### 9.3 Ablation Studies
Systematic removal of components to validate design choices:
- **Without Deformation Field**
- **Without Temporal Constraints**
- **Without Occlusion Handling**
- **Without Hierarchical Pool**

## 10. Limitations and Future Work

### Current Limitations
1. **Computational Cost**: High memory and compute requirements
2. **Occlusion Handling**: Limited explicit occlusion reasoning
3. **Long-term Consistency**: May drift over very long sequences
4. **Scale Variation**: Struggles with extreme scale changes

### Future Directions
1. **Implicit Representations**: Combine with implicit neural representations
2. **Physics Constraints**: Incorporate physical priors for motion
3. **Uncertainty Estimation**: Model uncertainty in predictions
4. **Real-time Rendering**: Optimize for interactive applications
5. **Multi-view Consistency**: Leverage multi-view supervision

This methodology provides a comprehensive framework for dynamic 3D scene reconstruction while maintaining the efficiency benefits of 3D Gaussian splatting.