

```markdown
# Facial Expression Spotting improvement in Long Videos

![improvement]
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
```
Original function:

```

G(IoU) = 1 if IoU > 0
0 otherwise

```
change in IoU Threshold:

G(IoU) = 1 if IoU > 0.2
0 otherwise

```

For short expressions:

| IoU | F1     |
|-----|--------|
| 0.0 | 0.1173 |
| 0.2 | 0.1365 |

**Best threshold:** 0.2

---

### 2. Overlapping Interval Removal (Post-Processing)

If two detected intervals overlap:

- Compute mean confidence of each interval  
- Keep only the one with higher score  

After applying:

- False Positives reduced from 264 → 174  
- F1 improved to 0.1594 (~34% error reduction)  

---

### 3. Eyebrow-Focused Optical Flow

To improve recall:

- Extract eyebrow landmarks  
- Apply binary mask  
- Compute optical flow only within eyebrow region  
- Train separate eyebrow network  

**Results**:

- 9 true positives  
- 7 unique detections not found by face network  
- F1 = 0.0549  

Although weaker alone, eyebrow network captures complementary motion information.

---

### 4. Decision Fusion

Two independently trained networks:

- Full face network  
- Eyebrow network  

**Final prediction**:

```

Final = w_face * P_face + w_eyebrow * P_eyebrow

```

**Best weights**:

- Face: 0.7  
- Eyebrow: 0.3  

**Results**:

| Model         | F1     | TP | FP  |
|---------------|--------|----|-----|
| Face only     | 0.1594 | 20 | 174 |
| Eyebrow only  | 0.0549 | 9  | 262 |
| Fusion        | 0.1765 | 21 | 160 |

Fusion improved:

- F1 to 0.1765  
- Reduced FP  
- Increased TP  
- ~50% overall improvement compared to baseline

---


## Long Expression Results

For long expressions:

- Baseline F1 = 0.2410  
- IoU thresholding did not improve performance  
- Post-processing not applied  
- No significant improvement observed  

This suggests long expressions may require different modeling strategies.

---

## Key Contributions

- Improved pseudo-labeling via IoU threshold tuning  
- Post-processing to remove overlapping intervals  
- Eyebrow-focused motion modeling  
- Weighted fusion strategy  
- Multi-stream shallow CNN architecture  
- Separate modeling for short and long expressions  

---

## Conclusion

This project presents a robust multi-stage framework for temporal emotion spotting in long videos.

The combination of:

- Refined pseudo-labeling  
- Optical strain modeling  
- Region-specific motion analysis  
- Decision fusion  

significantly improves detection performance for short expressions, particularly in recall enhancement.

The framework provides a strong foundation for:

- Affective computing  
- Human-computer interaction  
- Psychological analysis  
- Behavioral video understanding
```

---

---

---
---

## Pretrained Weights

You can download the pretrained model weights from the following link:

🔗 https://drive.google.com/drive/folders/1r8RXZPTUwKui2Q7b0-lKx9oe5q8s6Vvf?usp=drive_link

## Author / Credits

This project is based on the original code by **genbing67**  
Email: [genbing67@gmail.com](mailto:genbing67@gmail.com)  

All modifications and enhancements in this repository were made by the current contributor.
