import numpy as np
import pandas as pd
import cv2
import mediapipe as mp


# ----------------------
# Convert polar → cartesian
# ----------------------
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


# ----------------------
# Compute optical strain
# ----------------------
def computeStrain(u, v):
    u_x = u - pd.DataFrame(u).shift(-1, axis=1)
    v_y = v - pd.DataFrame(v).shift(-1, axis=0)
    u_y = u - pd.DataFrame(u).shift(-1, axis=0)
    v_x = v - pd.DataFrame(v).shift(-1, axis=1)

    o_s = np.array(np.sqrt(u_x**2 + v_y**2 + 0.5 * (u_y + v_x)**2).ffill(1).ffill(0))
    return o_s


# ----------------------
# MediaPipe eyebrow rectangle extractor
# ----------------------
def get_eyebrow_rect(img):

    print("DEBUG get_eyebrow_rect:", type(img), img.shape)

    # ---- FIX 1: handle grayscale inputs ----
    if img is None:
        raise ValueError("ERROR: img is None in get_eyebrow_rect")

    if img.ndim == 2:                       # (H,W)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.ndim == 3 and img.shape[2] == 1: # (H,W,1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w, _ = img.shape

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    )
    res = face_mesh.process(img)

    if not res.multi_face_landmarks:
        return None

    face = res.multi_face_landmarks[0]

    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]

    pts = []
    for group in [LEFT_EYEBROW, RIGHT_EYEBROW]:
        for idx in group:
            pts.append((int(face.landmark[idx].x * w),
                        int(face.landmark[idx].y * h)))

    pts = np.array(pts)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # padding
    x_min = max(x_min - 2, 0)
    y_min = max(y_min - 2, 0)
    x_max = min(x_max + 2, w)
    y_max = min(y_max + 2, h)

    return x_min, y_min, x_max, y_max


# ----------------------
# Main extraction function
# ----------------------
def extract_preprocess2(final_images, k):

    dataset = []

    for vid in range(len(final_images)):

        print("\nProcessing video:", vid)
        print("VIDEO:", vid, final_images[vid][0].shape)

        OFF_video = []

        # ---------------------------
        # 1) Eyebrow ROI from first frame
        # ---------------------------
        first_frame = final_images[vid][0]

        eyebrow_rect = get_eyebrow_rect(first_frame)

        if eyebrow_rect is None:
            print("Eyebrow not found – skipping video.")
            continue

        x1, y1, x2, y2 = eyebrow_rect
        H = y2 - y1
        W = x2 - x1

        # ---------------------------
        # 2) Loop frames & compute OF
        # ---------------------------
        for i in range(final_images[vid].shape[0] - k):

            img1 = final_images[vid][i]
            img2 = final_images[vid][i + k]

            # Crop
            roi1 = img1[y1:y2, x1:x2]
            roi2 = img2[y1:y2, x1:x2]

            # ---- FIX 2: convert ROI to RGB before cvtColor ----
            def ensure_rgb(x):
                if x.ndim == 2:
                    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
                if x.ndim == 3 and x.shape[2] == 1:
                    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
                return x

            roi1 = ensure_rgb(roi1)
            roi2 = ensure_rgb(roi2)

            # Convert to grayscale
            g1 = cv2.cvtColor(roi1, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY)

            # Optical flow TV-L1
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            flow = optical_flow.calc(g1, g2, None)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            u, v = pol2cart(mag, ang)
            o_s = computeStrain(u, v)

            # Final (H×W×3)
            final_roi = np.zeros((H, W, 3))
            final_roi[:, :, 0] = u
            final_roi[:, :, 1] = v
            final_roi[:, :, 2] = o_s

            final_resized = cv2.resize(final_roi, (42, 42))

            OFF_video.append(final_resized)

        dataset.append(OFF_video)
        print("Video", vid, "Done.")

    print("\nALL Eyebrows DONE")
    return dataset
