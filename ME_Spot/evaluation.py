import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d

def spotting(dataset, final_samples, k, result, subject_count, p, show=False):
    prev = 0
    metric = MeanAveragePrecision2d(num_classes=1)
    preds_all = []
    gt_all = []
    for videoIndex, video in enumerate(final_samples[subject_count-1]):
        preds = []
        gt = []
        countVideo = len([video for subject in final_samples[:subject_count-1] for video in subject])
        score_plot = np.array(result[prev:prev+len(dataset[countVideo+videoIndex])]) #Get related frames to each video
        score_plot_agg = score_plot.copy()
        
        #Score aggregation
        for x in range(len(score_plot[k:-k])):
            score_plot_agg[x+k] = score_plot[x:x+2*k].mean()
        score_plot_agg = score_plot_agg[k:-k]
        
        #Plot the result to see the peaks
        #Note for some video the ground truth samples is below frame index 0 due to the effect of aggregation, but no impact to the evaluation
        if show:
            print('\nVideo:', countVideo+videoIndex)
            plt.figure(figsize=(15,3))
            plt.plot(score_plot_agg) 
            plt.xlabel('Frame')
            plt.ylabel('Score')

        threshold = score_plot_agg.mean() + p * (max(score_plot_agg) - score_plot_agg.mean()) #Moilanen threshold technique
        peaks, _ = find_peaks(score_plot_agg, height=threshold, distance=k)
        if(len(peaks)==0): #Occurs when no peak is detected, simply give a value to pass the exception in mean_average_precision
            preds.append([0, 0, 0, 0, 0, 0, 0]) 
        for peak in peaks:
            preds.append([peak-k, 0, peak+k, 0, 0, 0, 0]) #Extend left and right side of peak by k frames
        for samples in video:
            gt.append([samples[0]-k, 0, samples[1]-k, 0, 0, 0, 0, 0])
            if show:
                plt.axvline(x=samples[0]-k, color='r')
                plt.axvline(x=samples[1]-k+1, color='r')
                plt.axhline(y=threshold, color='g')
        if show:
            plt.show()

        prev += len(dataset[countVideo+videoIndex])
        metric.add(np.array(preds), np.array(gt)) #IoU = 0.5 according to MEGC2020 metrics
        preds_all.append(preds)
        gt_all.append(gt)
    return preds_all, gt_all, metric

def evaluation(total_gt, metric): #Get TP, FP, FN for final evaluation
    TP = int(sum(metric.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP = int(sum(metric.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN = total_gt - TP
    
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = (2 * precision * recall) / (precision + recall)
    except:
        precision = recall = F1_score = 0
        
    return TP, FP, FN, F1_score, precision, recall

def final_evaluation(total_gt, metric_fn):
    TP, FP, FN, F1_score, precision, recall = evaluation(total_gt, metric_fn)
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    print("COCO AP@[.5:.95]:", round(metric_fn.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))