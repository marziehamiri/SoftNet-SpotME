from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
random.seed(1)


#succesful pseudo_labeling threshold modifying 
def pseudo_labeling(final_images, final_samples, k, iou_threshold=0.2):
    pseudo_y = []
    video_count = 0 
    
    for subject in final_samples:
        for video in subject:
            num_frames = len(final_images[video_count])
            pseudo_y_each = np.zeros(num_frames - k, dtype=int)

            if len(video) > 0:
                
                gt_intervals = [np.arange(ME[0]+1, min(ME[1]+1, num_frames)) for ME in video]

                
                for idx in range(num_frames - k):
                    window = np.arange(idx, idx + k)
                    
                    for gt in gt_intervals:
                        iou = len(np.intersect1d(window, gt)) / len(np.union1d(window, gt))
                        if iou > iou_threshold:
                            pseudo_y_each[idx] = 1
                            break  

            pseudo_y.append(pseudo_y_each)
            video_count += 1
    
    pseudo_y = np.concatenate(pseudo_y)
    print('Total frames:', len(pseudo_y))
    return pseudo_y


def loso(dataset, pseudo_y, final_images, final_samples, k):
    #To split the dataset by subjects
    y = np.array(pseudo_y)
    videos_len = []
    groupsLabel = y.copy()
    prevIndex = 0
    countVideos = 0
    
    #Get total frames of each video
    for video_index in range(len(final_images)):
      videos_len.append(final_images[video_index].shape[0]-k)
    
    print('Frame Index for each subject:-')
    for video_index in range(len(final_samples)):
      countVideos += len(final_samples[video_index])
      index = sum(videos_len[:countVideos])
      groupsLabel[prevIndex:index] = video_index
      print('Subject', video_index, ':', prevIndex, '->', index)
      prevIndex = index
    
    X = [frame for video in dataset for frame in video]
    print('\nTotal X:', len(X), ', Total y:', len(y))
    return X, y, groupsLabel
    
def normalize(images):
    for index in range(len(images)):
        for channel in range(3):
            images[index][:,:,channel] = cv2.normalize(images[index][:,:,channel], None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX)
    return images

def generator(X, y, batch_size=12, epochs=1):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            num_images = end - start
            X[start:end] = normalize(X[start:end])
            u = np.array(X[start:end])[:,:,:,0].reshape(num_images,42,42,1)
            v = np.array(X[start:end])[:,:,:,1].reshape(num_images,42,42,1)
            os = np.array(X[start:end])[:,:,:,2].reshape(num_images,42,42,1)
            yield [u, v, os], np.array(y[start:end])
            
def shuffling(X, y):
    shuf = list(zip(X, y))
    random.shuffle(shuf)
    X, y = zip(*shuf)
    return list(X), list(y)

def data_augmentation(X, y):
    transformations = {
        0: lambda image: np.fliplr(image), 
        1: lambda image: cv2.GaussianBlur(image, (7,7), 0),
        2: lambda image: random_noise(image),
    }
    y1=y.copy()
    for index, label in enumerate(y1):
        if (label==1): #Only augment on expression samples (label=1)
            for augment_type in range(3):
                img_transformed = transformations[augment_type](X[index]).reshape(42,42,3)
                X.append(np.array(img_transformed))
                y.append(1)
    return X, y

def SOFTNet():
    inputs1 = layers.Input(shape=(42,42,1))
    conv1 = layers.Conv2D(3, (5,5), padding='same', activation='relu')(inputs1)
    pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv1)
    # channel 2
    inputs2 = layers.Input(shape=(42,42,1))
    conv2 = layers.Conv2D(5, (5,5), padding='same', activation='relu')(inputs2)
    pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv2)
    # channel 3
    inputs3 = layers.Input(shape=(42,42,1))
    conv3 = layers.Conv2D(8, (5,5), padding='same', activation='relu')(inputs3)
    pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv3)
    # merge
    merged = layers.Concatenate()([pool1, pool2, pool3])
    # interpretation
    merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(merged)
    flat = layers.Flatten()(merged_pool)
    dense = layers.Dense(400, activation='relu')(flat)
    outputs = layers.Dense(1, activation='linear')(dense)
    #Takes input u,v,s
    model = keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    sgd = keras.optimizers.SGD(lr=0.0005)
    model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model



#micro successful filtering
def smart_micro_filter_confidence(predictions, max_gap=10):
    for p in predictions:
        p[0] = int(p[0])  # start
        p[2] = int(p[2])  # end
    if len(predictions) == 0:
        return np.array([])

    
    intervals = [(p[0], p[2], p) for p in predictions]
    intervals.sort(key=lambda x: x[0])  

    result = [intervals[0][2]]  

    for i in range(1, len(intervals)):
        last = result[-1]        
        curr = intervals[i][2]   

        
        if curr[0] <= last[2]:
            
            if curr[-1] > last[-1]:
                result[-1] = curr  
            
        else:
            result.append(curr)

    return np.array(result)




def filter_by_majority_vote_custom(intervals, probs, frame_thresh=0.03, vote_ratio=0.5):
  

    valid = []

    for interval in intervals:
        s = interval[0]     # شروع بازه
        e = interval[2]     # پایان بازه
        s = int(s)
        e = int(e)

        window = probs[s:e]   # احتمال فریم‌های داخل بازه

        num_pos = np.sum(window > frame_thresh)
        total = e - s

        # فیلتر اکثریت رأی
        if num_pos >= vote_ratio * total:
            valid.append(interval)
    if len(valid) == 0:
        valid.append(intervals[0])

    return valid


def spotting(result, total_gt, final_samples,final_samples_macro, subject_count, dataset, k, metric_fn, p, show_plot):
    prev=0
    
    for videoIndex, video in enumerate(final_samples[subject_count-1]):
        preds = []
        gt = []
        countVideo = len([video for subject in final_samples[:subject_count-1] for video in subject])
        print('Video:', countVideo+videoIndex)
        
        score_plot = np.array(result[prev:prev+len(dataset[countVideo+videoIndex])]) 
        score_plot_agg = score_plot.copy()
        
        # Score aggregation
        for x in range(len(score_plot[k:-k])):
            score_plot_agg[x+k] = score_plot[x:x+2*k].mean()
        score_plot_agg = score_plot_agg[k:-k]
        
        if(show_plot):
            plt.figure(figsize=(15,4))
            plt.plot(score_plot_agg) 
            plt.xlabel('Frame')
            plt.ylabel('Score')
            
        threshold = score_plot_agg.mean() + p * (max(score_plot_agg) - score_plot_agg.mean())
        peaks, _ = find_peaks(score_plot_agg[:,0], height=threshold[0], distance=k)
        
        if(len(peaks)==0): 
            preds.append([0, 0, 0, 0, 0, 0, 0])  
        for peak in peaks:
            start = peak - k
            end   = peak + k
            start = max(0, start)
            end = max(0, end)
            #print(f"[DEBUG] video {videoIndex}: len(result)={len(result)}, start={start}, end={end}")
           
            #confidence = float(result[start:end,0].max())
            confidence = float(result[start:end, 0].mean())
            preds.append([start, 0, end, 0, 0, 0, confidence]) 
            
        print("preds before postprocessing")
        print(preds)
        
        # Smart filter
        #preds = smart_micro_filter(preds, max_gap=12)
        preds = smart_micro_filter_confidence(preds, max_gap=10)
        #preds = filter_by_majority_vote_custom(preds, result, frame_thresh=0.05, vote_ratio=0.5)
        #preds = preds.astype(int)
        print("preds after smart postprocessing ")
        print(preds)

        preds = np.array(preds)[:, :6]

        # ground truth
        for samples in video:
            gt.append([samples[0]-k, 0, samples[1]-k, 0, 0, 0, 0])
            total_gt += 1
            if(show_plot):
                plt.axvline(x=samples[0]-k, color='r')
                plt.axvline(x=samples[1]-k+1, color='r')
                plt.axhline(y=threshold, color='g')
        if(show_plot):
            plt.show()
        prev += len(dataset[countVideo+videoIndex])
        metric_fn.add(np.array(preds),np.array(gt)) 
        
    return preds, gt, total_gt



def evaluation(preds, gt, total_gt, metric_fn): #Get TP, FP, FN for final evaluation
    TP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN = total_gt - TP
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    return TP, FP, FN

def evaluation1(preds, gt, total_gt, metric_fn): #Get TP, FP, FN for final evaluation
    TP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN = total_gt - TP
    #print('TP:', TP, 'FP:', FP, 'FN:', FN)
    return TP, FP, FN

def training(X, y, groupsLabel,
             X2, y2, groupsLabel2,
             dataset_name, expression_type,
             final_samples, final_samples_macro, 
             k, dataset, train, show_plot):

    print("DEBUG shapes: ", len(X), len(y), len(groupsLabel))

    
    if not (len(X) == len(y) == len(groupsLabel)):
        print("Mismatch indices:")
        print("extra y:", list(range(len(X), len(y))))
        print("extra groups:", list(range(len(X), len(groupsLabel))))

    logo = LeaveOneGroupOut()
    logo1 = LeaveOneGroupOut()
    subject_count = 0

    epochs = 20
    batch_size = 12
    total_gt = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)
    metric_fn1 = MeanAveragePrecision2d(num_classes=1)
    p = 0.55

    # face model
    model = SOFTNet()
    # eyebrow model
    model1 = SOFTNet()

    weight_reset = model.get_weights()
    weight_reset1 = model1.get_weights()

    #  LOSO
    for train_index, test_index in logo.split(X, y, groupsLabel):
        subject_count += 1
        print('Subject:', subject_count)

        test_subject_id = list(set(groupsLabel[i] for i in test_index))[0]
        train_subjects = set(groupsLabel[i] for i in train_index)
        print(f"Testing on Subject ID: {test_subject_id}")
        print(f"Training on Subject IDs: {sorted(train_subjects)}")

        # -----------------------------------------------------
        #   face data
        # -----------------------------------------------------
        X_train = [X[i] for i in train_index]
        y_train = [y[i] for i in train_index]
        X_test  = [X[i] for i in test_index]
        y_test  = [y[i] for i in test_index]

        # -----------------------------------------------------
        #   eyebrow data
        # -----------------------------------------------------
        X2_train = [X2[i] for i in train_index]
        y2_train = [y2[i] for i in train_index]
        X2_test  = [X2[i] for i in test_index]
        y2_test  = [y2[i] for i in test_index]

        # weight destination
        path_face    = 'SOFTNet_Weights2/Micro/Threshold2' + '/s' + str(subject_count) + '.hdf5'
        path_eyebrow = 'SOFTNet_Weights2/Micro/Eyebrows' + '/s' + str(subject_count) + '.hdf5'

        # -----------------------------------------------------
        # TRAINING
        # -----------------------------------------------------
        if train:
            # ======== face network ========
            print('Training FACE network...')
            model.set_weights(weight_reset)

            # Downsample 
            print('Dataset Labels Face', Counter(y_train))
            unique, uni_count = np.unique(y_train, return_counts=True)
            rem_count = int(uni_count.max() * 1/2)

            rem_index = random.sample([i for i, t in enumerate(y_train) if t == 0], rem_count)
            rem_index += (i for i, t in enumerate(y_train) if t > 0)
            rem_index = sorted(rem_index)

            X_train = [X_train[i] for i in rem_index]
            y_train = [y_train[i] for i in rem_index]

            # Augmentation for micro
            if expression_type == 'micro-expression':
                X_train, y_train = data_augmentation(X_train, y_train)

            X_train, y_train = shuffling(X_train, y_train)

            model.fit(
                generator(X_train, y_train, batch_size, epochs),
                steps_per_epoch=len(X_train) / batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=generator(X_test, y_test, batch_size),
                validation_steps=len(X_test) / batch_size,
                shuffle=True,
            )
            model.save_weights(path_face)
            print(f">>> Face Weights Saved: {path_face}")

            # ======== eyebrow network ========
            print('Training EYEBROW network...')
            model1.set_weights(weight_reset1)

            print('Dataset Labels Eyebrow', Counter(y2_train))
            unique, uni_count = np.unique(y2_train, return_counts=True)
            rem_count = int(uni_count.max() * 1/2)

            rem_index = random.sample([i for i, t in enumerate(y2_train) if t == 0], rem_count)
            rem_index += (i for i, t in enumerate(y2_train) if t > 0)
            rem_index = sorted(rem_index)

            X2_train = [X2_train[i] for i in rem_index]
            y2_train = [y2_train[i] for i in rem_index]

            if expression_type == 'micro-expression':
                X2_train, y2_train = data_augmentation(X2_train, y2_train)

            X2_train, y2_train = shuffling(X2_train, y2_train)

            model1.fit(
                generator(X2_train, y2_train, batch_size, epochs),
                steps_per_epoch=len(X2_train) / batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=generator(X2_test, y2_test, batch_size),
                validation_steps=len(X2_test) / batch_size,
                shuffle=True,
            )
            model1.save_weights(path_eyebrow)
            print(f">>> Eyebrow Weights Saved: {path_eyebrow}")

        else:
            model.load_weights(path_face)
            model1.load_weights(path_eyebrow)
            print("Loaded pretrained weights.")

        # -----------------------------------------------------
        #   TEST → PREDICTION
        # -----------------------------------------------------
        print("Predicting FACE network...")
        result_face = model.predict_generator(
            generator(X_test, y_test, batch_size),
            steps=len(X_test) / batch_size,
            verbose=1
        )

        print("Predicting EYEBROW network...")
        result_eyebrow = model1.predict_generator(
            generator(X2_test, y2_test, batch_size),
            steps=len(X2_test) / batch_size,
            verbose=1
        )

        # -----------------------------------------------------
        #   Late fusion
        # -----------------------------------------------------
        

        w_face = 0.7
        w_eyebrow = 0.3
        result = w_face * result_face + w_eyebrow * result_eyebrow
        

        # -----------------------------------------------------
        #   SPOTTING + EVAL
        # -----------------------------------------------------
        preds, gt, total_gt = spotting(
            result, total_gt, final_samples, final_samples_macro,
            subject_count, dataset, k, metric_fn, p, show_plot
        )

        print("macro preds:", preds)
        TP, FP, FN = evaluation(preds, gt, total_gt, metric_fn)
        print("Done Subject", subject_count)

    return TP, FP, FN, metric_fn


def final_evaluation(TP, FP, FN, metric_fn):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    print("COCO AP@[.5:.95]:", round(metric_fn.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))
    
    
