حتماً! من متن شما را در قالب **Markdown مناسب برای فایل `README.md`** فرمت می‌کنم تا هنگام نمایش در گیت‌هاب درست و خوانا باشد، بدون تغییر محتوای اصلی:

```markdown
# Facial Emotion Temporal Spotting and Recognition in Long Videos

## Overview
This project presents a comprehensive framework for temporal spotting and recognition of facial emotions in long video sequences. The system covers all major stages of the pipeline, including:

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

### Dataset Characteristics
- 22 subjects  
- Long video sequences (0.5 to 2 minutes)  
- 30 FPS  
- Annotated onset, apex, and offset frames  
- Labels derived from:
  - Facial Action Units (AUs)  
  - Stimulus-induced emotion  
  - Self-reports from participants  

The dataset enables continuous temporal spotting research in realistic long video settings.

---

## Methodology

### 1. Feature Extraction & Preprocessing

**Face Alignment**  
- Face region detected using 68 facial landmarks  
- Cropped and resized to 128 × 128 pixels  

**Optical Flow Computation**  
For each frame i, optical flow is computed between:

```

Frame i and Frame i + k

```

Where:

```

k = 1/2 × average expression length

```

We extract:

- Horizontal optical flow (u)  
- Vertical optical flow (v)  
- Optical strain magnitude  

> Optical strain captures subtle deformation changes and is highly sensitive to micro-movements.

**Noise Reduction**  
- Global head motion removed using nose region reference  
- Eye regions removed to suppress blinking noise  

**Key regions selected**:  

- Left eye + eyebrow  
- Right eye + eyebrow  
- Mouth  

These regions are masked and resized to 42 × 42, then merged into a unified motion representation.

---

### 2. Pseudo-Labeling Strategy

Since frame-level labels are unavailable, we generate pseudo-labels using a sliding window approach.

**Sliding Window**  

- Window size = k  
- For each window, compute IoU with ground truth interval:

```

IoU = w ∩ ε / w ∪ ε

```

**Label Function**

Original function:

```

G(IoU) = 1 if IoU > 0
0 otherwise

```

We improved it by introducing IoU thresholds:

- 0.1  
- 0.2  
- 0.3  

For short expressions, IoU = 0.2 yielded the best performance.

---

### 3. Three-Stream Shallow CNN (SoftNet)

**Input Streams**:

- Horizontal optical flow  
- Vertical optical flow  
- Optical strain magnitude  

**Architecture**:

- Each stream contains one shallow convolution layer (5×5 filters)  
- Number of filters per stream:
  - Stream 1: 3 filters  
  - Stream 2: 5 filters  
  - Stream 3: 8 filters  
- Outputs concatenated channel-wise  
- Then:
  - Pooling  
  - Flatten  
  - Fully connected layer (400 nodes)  
  - Linear regression output  

> The model outputs a continuous confidence score per frame instead of binary classification.  

**Separate models are trained for**:

- Short expressions  
- Long expressions  

---

### 4. Temporal Peak Detection

**Frame scores are smoothed**:

```

Si = (1 / (2k+1)) ∑(j=i-k to i+k) sj

```

**Threshold**:

```

T = Smean + p * (Smax - Smean)

```

- Best parameters:
  - Short expressions: p = 0.55  
  - Long expressions: p = 0.35  

Peaks above threshold define detected emotional intervals.

---

## Error Reduction Strategies

### 1. Reducing False Positives via IoU Thresholding

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

## Evaluation Metrics

We use:

- Precision  
- Recall  
- F1-score  
- TP / FP / FN  

**Definitions**:

```

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)

```

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

## Author / Contact

This project and code are authored by **genbing67**.  
For any questions or correspondence, contact: [genbing67@gmail.com](mailto:genbing67@gmail.com)
