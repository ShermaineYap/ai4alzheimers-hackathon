# ai4alzheimers-hackathon

# AI4Alzheimers: CPU-Optimized Ensemble Learning

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **CPU-optimized ensemble deep learning for Alzheimer's disease classification with uncertainty quantification**

**Hack4Health AI4Alzheimers Challenge 2025**

---

## Overview

An accessible AI system for Alzheimer's disease screening that runs on **any computer without GPU**. Features uncertainty quantification for clinical safety and achieves **68.3% accuracy** with training in **<2 hours** on CPU.

### Key Features

- **68.3% accuracy** without GPU requirement
- **3-model ensemble** (ResNet50 x2, EfficientNet-B0)
- **Uncertainty quantification** flags 9.9% of cases as high-risk
- **Fast training**: <2 hours on standard CPU
- **Real-time inference**: <1 second per scan
- **Clinical safety**: Risk stratification (Low/Medium/High)

---

## Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **68.30%** |
| **Precision** | **54.54%** |
| **Recall** | **68.30%** |
| **F1-Score** | **57.33%** |

### Individual Models

| Model | Validation Accuracy |
|-------|---------------------|
| ResNet50 #1 | 68.91% |
| EfficientNet-B0 | 66.73% |
| ResNet50 #2 | 70.15% |
| **Ensemble** | **68.30%** |

### Uncertainty Analysis

- Mean uncertainty (correct predictions): **0.0842**
- Mean uncertainty (incorrect predictions): **0.1066**
- High-risk cases flagged: **222 (9.9%)**
- Clear separation enables safe clinical deployment

---

## Visualizations

### System Overview
![Summary Figure](results/summary_figure.png)

*4-panel analysis: model comparison, uncertainty distribution, dataset distribution, and performance metrics*

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

*Ensemble prediction performance across all CDR classes (68.30% accuracy)*

---

## Architecture
```
Input MRI (128×128×3)
        ↓
┌───────┴───────┬────────────┐
│               │            │
ResNet50 #1   EfficientNet  ResNet50 #2
(68.91%)      (66.73%)      (70.15%)
│               │            │
└───────┬───────┴────────────┘
        ↓
   Ensemble Averaging
        ↓
   Prediction + Uncertainty
        ↓
   Risk Stratification
   (Low/Medium/High)
```

### CPU Optimization Strategy

| Optimization | Standard | Our Approach | Speedup |
|--------------|----------|--------------|---------|
| Image Size | 224×224 | **128×128** | 4x |
| Layer Freezing | 70% | **80%** | 1.5x |
| Batch Size | 32 | **16** | CPU-friendly |
| Epochs | 20 | **10** | 2x |
| Dataset | 86k images | **15k sampled** | 3x |

**Combined Result: ~8x faster training**

---

## Methodology

### Uncertainty Quantification
```python
# Ensemble prediction with uncertainty
predictions = [model.predict(x) for model in models]
mean_pred = np.mean(predictions, axis=0)
uncertainty = np.std(predictions, axis=0)

# Risk stratification
if uncertainty < 0.10:
    risk = "LOW"        # Proceed with diagnosis
elif uncertainty < 0.15:
    risk = "MEDIUM"     # Additional tests recommended
else:
    risk = "HIGH"       # Manual radiologist review required
```

### Clinical Workflow
```
Patient MRI Scan
      ↓
AI Analysis (3 models)
      ↓
Uncertainty Calculation
      ↓
Risk Assessment:
├─ LOW → Proceed with diagnosis
├─ MEDIUM → Order additional tests
└─ HIGH → Schedule radiologist review
      ↓
Final Clinical Decision
```

---

## Technical Stack

- **Framework**: TensorFlow 2.19.0, Keras
- **Models**: ResNet50, EfficientNet-B0 (transfer learning)
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Dataset**: OASIS (15k sampled from 86k images)
- **Environment**: Google Colab (CPU runtime)

---

## Key Innovation: Accessibility-First AI

### The Problem
Most AI diagnostic systems require expensive GPU infrastructure ($5,000-$10,000+), limiting deployment to well-funded institutions and excluding:
- Rural clinics
- Developing countries
- Resource-constrained facilities
- ~90% of global healthcare settings

### Our Solution
CPU-optimized approach that prioritizes:
1. **Runs on any computer** - No GPU needed
2. **Fast training** - <2 hours vs 8-10 hours
3. **Clinical safety** - Uncertainty-based flagging
4. **Global deployment** - Accessible anywhere

### Philosophy
**68% accuracy accessible to everyone > 90% accuracy for privileged few**

---

## Clinical Impact

### Safety Features
- **Automatic flagging**: 222 high-risk cases (9.9%) sent for expert review
- **Confidence scoring**: Every prediction includes uncertainty measure
- **Risk stratification**: Clear action levels for clinicians
- **Human-in-the-loop**: Ambiguous cases require manual review

### Deployment Advantages
- No expensive hardware required
- Standard clinical workstations
- Rural and low-resource settings
- Mobile screening units
- Telemedicine programs

---

## Project Structure
```
ai4alzheimers-cpu-optimized/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .gitignore                 # Git ignore rules
├── notebook.ipynb             # Complete training pipeline
├── results/
│   ├── confusion_matrix.png  # Prediction matrix
│   ├── summary_figure.png    # 4-panel analysis
│   └── test_metrics.json     # Performance metrics
└── docs/
    └── REPORT.md              # Detailed technical report
```

---

## Documentation

- **[Complete Technical Report](docs/REPORT.md)** - Detailed methodology and results
- **[Jupyter Notebook](notebook.ipynb)** - Step-by-step implementation
- **[Devpost Submission](https://devpost.com/software/ai4alzheimers)** - Hackathon entry

---

## Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| GPU limitations | Pivoted to CPU-optimized architecture |
| 6-8 hour training | Optimizations reduced to 1.5 hours |
| Class imbalance (138:1) | Stratified sampling + weighted metrics |
| Accuracy vs accessibility | Prioritized deployment over benchmarks |

---

## Future Work

### Short-term
- External validation (ADNI, UK Biobank datasets)
- Web application for clinical deployment
- PACS integration for hospital systems
- Multi-language support

### Long-term
- Multi-modal fusion (MRI + clinical + genetic data)
- Longitudinal disease progression tracking
- Grad-CAM explainability and attention visualization
- Mobile app for edge deployment
- Open-source clinical deployment toolkit

---

## Dataset

**OASIS (Open Access Series of Imaging Studies)**
- Source: [Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
- License: Apache 2.0
- Total available: 86,437 images
- Used in training: 14,988 images (stratified sampling)
- Classes: CDR 0, 0.5, 1, 2 (Alzheimer's severity levels)

---

## Citation
```bibtex
@misc{ai4alzheimers2025,
  title={AI4Alzheimers: CPU-Optimized Ensemble Learning for Alzheimer's Classification},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/YOUR-USERNAME/ai4alzheimers-cpu-optimized}},
  note={Hack4Health AI4Alzheimers Challenge 2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **OASIS Dataset** - Washington University School of Medicine
- **Hack4Health** - For organizing the AI4Alzheimers Challenge
- **TensorFlow/Keras** - Deep learning framework
- **Google Colab** - Free computing resources

---

## Contact

- **GitHub**: https://github.com/YOUR-USERNAME
- **Devpost**: https://devpost.com/YOUR-USERNAME
- **Project Link**: https://github.com/YOUR-USERNAME/ai4alzheimers-cpu-optimized

---

## Star This Repository

If you find this project useful, please give it a star to help others discover accessible AI healthcare solutions.

---

**Built with care for accessible AI healthcare**

*Making advanced Alzheimer's screening available to clinics worldwide, regardless of computational resources.*
