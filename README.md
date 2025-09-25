# Explanatory Interactive Machine Learning for Bias Mitigation in Visual Gender Classification

This repository contains the **implementation, dataset, and experiments** for the paper:  
**“Explanatory Interactive Machine Learning for Bias Mitigation in Visual Gender Classification.”**

We explore two state-of-the-art **Explanatory Interactive Learning (XIL)** strategies — **CAIPI** and **Right for the Right Reasons (RRR)** — and propose a **novel hybrid approach** to guide visual classifiers toward more relevant features, reducing bias and improving fairness in gender classification tasks.

---

## 🚀 Features
- Implementation of **CAIPI** and **RRR** XIL strategies  
- Novel **Hybrid XIL approach** combining strengths of both methods  
- Evaluation of **fairness, transparency, and interpretability** in visual gender classification  
- Comparisons using **GradCAM** and **Bounded Logit Attention (BLA)** for model explanations  
- Experiments showing how XIL can reduce **bias** (e.g., balancing misclassification rates between male and female predictions)  

---

## 📦 Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/xil-bias-gender-classification.git
cd xil-bias-gender-classification

pip install -r requirements.txt
```

---

## 📊 Experiments
We evaluate models on:
- Baseline classifiers without XIL
- **CAIPI** with varying numbers of augmentations
- **RRR** with explanation-based regularization
- Hybrid combining **CAIPI** and **RRR**
- 2 explainability methods **GradCAM** and **Bounded Logit Attention (BLA)**
- 2 sampling strategies **Uncertainty** sampling **High Confidence** sampling

---

## 📂 Dataset
The dataset used in this study is a **subset of the [MS COCO dataset](https://cocodataset.org)** that contains images of persons.  
Since MS COCO does not natively provide annotations for gender or sex, a **manual data selection process** was carried out to construct a **binary gender classification dataset**.  

⚠️ Note: We are aware that binary gender represents an over-simplification. To limit task complexity and simplify result interpretation, the scope of this study was restricted to binary classification.

---

## 📖 Citation
If you use this work, please cite:
@article{tbd,
  title={Explanatory Interactive Machine Learning for Bias Mitigation in Visual Gender Classification},
  author={Slijepcevic, Djordje and Labadie, Roberto and Chen, Xihui and Böck, Adrian Jaques and Babic, Andreas and Zeppelzauer, Matthias},
  journal={Journal Name},
  year={2025}
}
