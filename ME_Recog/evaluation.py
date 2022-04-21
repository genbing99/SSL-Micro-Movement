from sklearn.metrics import confusion_matrix
import numpy as np

def confusionMatrix(gt, pred, show=False):
    TN_recog, FP_recog, FN_recog, TP_recog = confusion_matrix(gt, pred).ravel()
    f1_score = (2*TP_recog) / (2*TP_recog + FP_recog + FN_recog)
    num_samples = len([x for x in gt if x==1])
    average_recall = TP_recog / (TP_recog + FN_recog)
    average_precision = TP_recog / (TP_recog + FP_recog)
    return f1_score, average_recall, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, average_precision, average_recall

def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2 }
    
    #Display recognition result
    precision_list = []
    recall_list = []
    f1_list = []
    ar_list = []
    TP_all = 0
    FP_all = 0
    FN_all = 0
    TN_all = 0
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x==emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x==emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, precision_recog, recall_recog = confusionMatrix(gt_recog, pred_recog, show)
                if(show):
                    print(emotion.title(), 'Emotion:')
                    print('TP:', TP_recog, '| FP:', FP_recog, '| FN:', FN_recog, '| TN:', TN_recog)
                TP_all += TP_recog
                FP_all += FP_recog
                FN_all += FN_recog
                TN_all += TN_recog
                precision_list.append(precision_recog)
                recall_list.append(recall_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        precision_list = [0 if np.isnan(x) else x for x in precision_list]
        recall_list = [0 if np.isnan(x) else x for x in recall_list]
        precision_all = np.mean(precision_list)
        recall_all = np.mean(recall_list)
        f1_all = (2 * precision_all * recall_all) / (precision_all + recall_all)
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        if (show):
            print('------ After adding ------')
            print('TP:', TP_all, 'FP:', FP_all, 'FN:', FN_all, 'TN:', TN_all)
            print('Precision:', round(precision_all, 4), 'Recall:', round(recall_all, 4))
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4), '| F1-Score:', round(f1_all, 4))
        return UF1, UAR
    except:
        return '', ''

def subject_evaluation(all_gt, all_pred):
    SAMM_gt = all_gt[:28]
    SAMM_gt = [i for i in SAMM_gt for i in i]
    SAMM_pred = all_pred[:28]
    SAMM_pred = [i for i in SAMM_pred for i in i]
    print('---- SAMM :----')
    UF1, UAR = recognition_evaluation(SAMM_gt, SAMM_pred, show=False)
    
    CASME2_gt = all_gt[28:52]
    CASME2_gt = [i for i in CASME2_gt for i in i]
    CASME2_pred = all_pred[28:52]
    CASME2_pred = [i for i in CASME2_pred for i in i]
    print('---- CASME 2 :----')
    UF1, UAR = recognition_evaluation(CASME2_gt, CASME2_pred, show=False)
    
    SMIC_gt = all_gt[52:]
    SMIC_gt = [i for i in SMIC_gt for i in i]
    SMIC_pred = all_pred[52:]
    SMIC_pred = [i for i in SMIC_pred for i in i]
    print('---- SMIC :----')
    UF1, UAR = recognition_evaluation(SMIC_gt, SMIC_pred, show=False)
    print('\n')