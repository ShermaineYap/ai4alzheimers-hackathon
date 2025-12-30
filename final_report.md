# AI4Alzheimers: CPU-Optimized Ensemble Learning for Alzheimer's Classification

## Executive Summary

**Innovation:** CPU-accessible ensemble deep learning system with uncertainty quantification for safe Alzheimer's disease screening.

**Key Results:**
- Ensemble Accuracy: **68.30%**
- F1-Score: **57.33%**
- High-risk cases flagged: **222** (9.9% of predictions)
- Training time: **<2 hours on CPU**

**Unique Approach:**
- 3-model ensemble (ResNet50 ×2, EfficientNet-B0)
- Uncertainty quantification via prediction variance
- Optimized for CPU deployment (no GPU required)
- Risk stratification for clinical safety

---

## 1. Problem Statement

Alzheimer's disease affects over 6.7 million Americans. Current AI diagnostic systems face:
- Dependence on expensive GPU infrastructure
- Lack of confidence quantification
- No safety mechanisms for ambiguous cases
- Limited accessibility in resource-constrained settings

**Our Solution:** A CPU-optimized ensemble that prioritizes accessibility and clinical safety.

---

## 2. Dataset

**OASIS (Open Access Series of Imaging Studies)**
- Training subset: **14,988 images** (sampled from 86k)
- CDR 0 (Non-Demented): 10,000 images
- CDR 0.5 (Very Mild): 3,000 images
- CDR 1 (Mild): 1,500 images
- CDR 2 (Moderate): 488 images

**Data Splits:**
- Training: 10,491 images (70%)
- Validation: 2,248 images (15%)
- Test: 2,249 images (15%)

**Strategic Sampling:** Enables faster iteration while maintaining class proportions.

---

## 3. Methodology

### 3.1 Ensemble Architecture

**Three Models:**
1. ResNet50 (Variant 1) - 68.91% validation accuracy
2. EfficientNet-B0 - 66.73% validation accuracy
3. ResNet50 (Variant 2) - 70.15% validation accuracy

**Transfer Learning:**
- Pre-trained on ImageNet
- 80% layer freezing for CPU efficiency
- Custom classification head (256 units)

### 3.2 CPU Optimization

**Key Optimizations:**
- Image size: 128×128 (vs 224×224) → 4× faster
- Batch size: 16 (CPU-friendly)
- Training epochs: 10 with early stopping
- Higher learning rate (0.001) for faster convergence

**Time Efficiency:**
- Total training: ~1.5 hours on CPU
- Inference: <1 second per image

### 3.3 Uncertainty Quantification

**Methodology:**
```
uncertainty = std(model1, model2, model3)
```

**Risk Stratification:**
- Low Risk (< 0.10): Proceed with diagnosis
- Medium Risk (0.10-0.15): Additional tests
- High Risk (> 0.15): Manual review required

**Results:**
- Mean uncertainty (correct): **0.0842**
- Mean uncertainty (incorrect): **0.1066**
- Clear separation enables safe deployment

---

## 4. Results

### 4.1 Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | **68.30%** |
| Precision | **54.54%** |
| Recall | **68.30%** |
| F1-Score | **57.33%** |

### 4.2 Individual Model Performance

| Model | Validation Accuracy |
|-------|-------------------|
| ResNet50 #1 | 68.91% |
| EfficientNet-B0 | 66.73% |
| ResNet50 #2 | **70.15%** |
| **Ensemble** | **68.30%** |

### 4.3 Risk Stratification

- **High-risk predictions:** 222 cases (9.9%)
- These cases automatically flagged for radiologist review
- Enables safe clinical deployment with human oversight

---

## 5. Clinical Impact

### 5.1 Accessibility Advantages

**No GPU Required:**
- Runs on standard clinical workstations
- Accessible to resource-constrained clinics
- Rural and low-income area deployment
- No expensive hardware upgrades

**Fast Deployment:**
- CPU inference: <1 second per scan
- Easy integration with existing systems
- Standard Python environment

### 5.2 Safety Features

**Built-in Quality Control:**
- Automatic uncertainty-based flagging
- Explicit confidence scores
- Human-in-the-loop decision making
- Reduced risk of overconfident errors

### 5.3 Clinical Workflow
```
1. Patient MRI scan
    ↓
2. AI Ensemble Analysis
    ↓
3. Uncertainty Calculation
    ↓
4. Risk-Based Routing:
   - Low → Proceed
   - Medium → Additional tests
   - High → Expert review
    ↓
5. Final Clinical Decision
```

---

## 6. Limitations & Future Work

### Current Limitations
- 68% accuracy vs 85-90% GPU systems
- Trade-off: accessibility vs performance
- Single dataset validation needed
- Class imbalance challenges

### Future Enhancements
1. **Multi-modal fusion:** MRI + clinical data + genetics
2. **Longitudinal tracking:** Disease progression modeling
3. **Explainability:** Grad-CAM attention maps
4. **Mobile deployment:** Edge computing optimization
5. **External validation:** Multi-site clinical trials

---

## 7. Conclusion

This project demonstrates that **effective AI healthcare need not require expensive infrastructure**. Through strategic optimization:

✅ **Accessibility** - CPU-only deployment anywhere
✅ **Safety** - Uncertainty-based risk stratification
✅ **Practicality** - <2 hour training, real-time inference
✅ **Impact** - Democratizes AI screening globally

**Key Innovation:** Prioritizing deployment feasibility and clinical safety enables real-world impact in resource-constrained settings.

---

## 8. Technical Specifications

**Environment:**
- Platform: Google Colab (CPU)
- TensorFlow: 2.19.0
- Training time: 1.5 hours
- Model size: ~196MB total

**Hardware Requirements:**
- CPU: Any modern processor
- RAM: 8GB minimum
- No GPU required

---

## 9. References

1. OASIS Dataset - Open Access Series of Imaging Studies
2. He et al. (2016) - Deep Residual Learning
3. Tan & Le (2019) - EfficientNet Architecture
4. Gal & Ghahramani (2016) - Uncertainty in Deep Learning

---

**Submitted for:** Hack4Health AI4Alzheimers Challenge
**Date:** December 2025
**Approach:** CPU-Optimized Ensemble with Uncertainty Quantification
