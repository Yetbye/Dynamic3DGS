# Dynamic3DGS: Deformable 3D Gaussian Splatting for Dynamic Scenes

A research project exploring deformable 3D Gaussian representations for dynamic scene reconstruction from monocular videos.

## 🎯 Research Motivation

Current 3D Gaussian Splatting (3DGS) excels in static scene reconstruction but faces challenges with dynamic scenes:
- **Dynamic Gaussian Modeling**: How to model temporal consistency for Gaussians across time steps
- **Occlusion Handling**: Dynamic occlusion relationships change over time
- **Memory Efficiency**: Independent Gaussians per frame lead to memory explosion
- **Generalization**: Reconstruction quality on unseen dynamic scenarios

## 🔬 Scientific Questions

This project aims to explore "Deformable Field-based Dynamic 3D Gaussian Splatting" with the core hypothesis:
- Shared static Gaussians + learnable temporal deformation fields can efficiently model dynamic scenes
- Physical constraints (motion smoothness, occlusion awareness) improve temporal consistency

## 💡 Innovation Points

1. **Deformable Gaussian Representation**: Each Gaussian has static attributes + dynamic offset fields
2. **Temporal Consistency Loss**: Optical flow consistency and motion smoothness regularization
3. **Occlusion-Aware Rendering**: Dynamic occlusion handling based on depth sorting
4. **Hierarchical Gaussian Pool**: Multi-scale Gaussian sets balancing detail and efficiency

## 📁 Project Structure

```
Dynamic3DGS/
├── configs/                    # Configuration files
├── data/                       # Data loading and preprocessing
├── models/                     # Core model implementations
├── training/                   # Training pipeline
├── evaluation/                 # Metrics and visualization
├── experiments/                # Ablation and baseline studies
├── utils/                      # Utility functions
├── scripts/                    # Training and evaluation scripts
├── tests/                      # Unit tests
├── docs/                       # Methodology documentation
└── notebooks/                  # Jupyter notebooks for analysis
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
python setup.py develop
```

### Training
```bash
python scripts/train.sh --config configs/experiments/final_model.yaml --data-path ./data/nvidia_dynamic
```

### Evaluation
```bash
python scripts/evaluate.sh --checkpoint ./checkpoints/latest.pth --dataset nvidia_dynamic
```

### Visualization
```bash
python scripts/visualize_results.py --results ./results --output ./visualizations
```

## 📊 Evaluation Metrics

- **Image Quality**: PSNR, SSIM, LPIPS
- **Temporal Quality**: FVD, Temporal SSIM
- **Geometry Quality**: Chamfer Distance, Normal Consistency
- **Efficiency**: FPS, Memory Usage, Training Time

## 🧪 Experiments

### Ablation Studies
- Without deformation field
- Without temporal constraints
- Without occlusion handling

### Baseline Comparisons
- D-NeRF (ECCV 2020)
- HyperNeRF (ICCV 2021)
- TianGang (SIGGRAPH 2022)
- Static 3DGS (SIGGRAPH 2023)

## 📚 Documentation

- [Methodology](docs/methodology.md): Detailed algorithm description
- [Experimental Design](docs/experiments_design.md): Comprehensive experiment framework
- [Results Analysis](docs/results_analysis.md): Expected outcomes and analysis

## 🤝 Contributing

This is a research project. Contributions should focus on:
- Algorithm improvements
- Experimental validation
- Performance optimization
- Documentation enhancement

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{dynamic3dgs2024,
  title={Dynamic3DGS: Deformable 3D Gaussian Splatting for Dynamic Scenes},
  author={Your Name},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## 📧 Contact

For questions about this research project, please contact: your.email@example.com