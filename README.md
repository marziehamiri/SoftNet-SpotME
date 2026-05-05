# Facial Expression Spotting improvement in Long Videos

![improvement](https://raw.githubusercontent.com/marziehamiri/SoftNet-SpotME/main/images/improvement.jpg)

## Overview
This project presents a comprehensive framework for temporal spotting of facial emotions in long video sequences. The system covers all major stages of the pipeline, including:

- Facial landmark detection  
- Optical flow and optical strain computation  
- Data preprocessing  
- Pseudo-label generation  
- Multi-stream CNN modeling  
- Post-processing and decision fusion  

The primary challenge addressed in this work is the low recall rate in temporal emotion spotting, where many true emotional intervals are missed. To tackle this issue, we introduce:

- Improved pseudo-labeling strategies  
- Targeted post-processing methods  
- Eyebrow-focused motion modeling  
- Weighted fusion of multiple networks  

The proposed framework improves recall while maintaining balanced precision, particularly for short facial expressions.

---

## Dataset

### CAS(ME)² Dataset
We evaluate our approach on the CAS(ME)² dataset, the first publicly available dataset that contains both:

- Short-term expressions (micro-expressions)  
- Long-term expressions (macro-expressions)

---

## Error Reduction Strategies

### 1. Reducing False Positives via IoU Thresholding in Pseudo-Labeling Strategy

Original function:

```text
G(IoU) = 1 if IoU > 0
0 otherwise

Changed function:

```text
G(IoU) = 1 if IoU > 0.2
0 otherwise
