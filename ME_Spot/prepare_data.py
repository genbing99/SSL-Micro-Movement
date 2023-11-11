import pickle
import numpy as np
from collections import Counter
from tensorflow.keras.utils import to_categorical
import pandas as pd

# Load pkl
def load_data(dataset_name):
    dataset = pickle.load( open( "dataset/" + dataset_name + "_dataset.pkl", "rb" ) )
    subjects = pickle.load( open( "dataset/" + dataset_name + "_subjects.pkl", "rb" ) )
    subjectsVideos = pickle.load( open( "dataset/" + dataset_name + "_subjectsVideos.pkl", "rb" ) )
    return dataset, subjects, subjectsVideos

def load_excel(dataset_name):
    if(dataset_name == 'CASME_sq'):
        xl = pd.ExcelFile('dataset/code_final.xlsx') #Specify directory of excel file

        colsName = ['subject', 'video', 'onset', 'apex', 'offset', 'au', 'emotion', 'type', 'selfReport']
        codeFinal = xl.parse(xl.sheet_names[0], header=None, names=colsName) #Get data

        videoNames = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(videoName.split('_')[0])
        codeFinal['videoName'] = videoNames

        naming1 = xl.parse(xl.sheet_names[2], header=None, converters={0: str})
        dictVideoName = dict(zip(naming1.iloc[:,1], naming1.iloc[:,0]))
        codeFinal['videoCode'] = [dictVideoName[i] for i in codeFinal['videoName']]

        naming2 = xl.parse(xl.sheet_names[1], header=None)
        dictSubject = dict(zip(naming2.iloc[:,2], naming2.iloc[:,1]))
        codeFinal['subjectCode'] = [dictSubject[i] for i in codeFinal['subject']]
        
    elif(dataset_name=='SAMMLV'):
        xl = pd.ExcelFile('dataset/SAMM_LongVideos_V2_Release.xlsx')

        colsName = ['Subject', 'Filename', 'Inducement Code', 'Onset', 'Apex', 'Offset', 'Duration', 'Type', 'Action Units', 'Notes']
        codeFinal = xl.parse(xl.sheet_names[0], header=None, names=colsName, skiprows=[0,1,2,3,4,5,6,7,8,9])

        videoNames = []
        subjectName = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(str(videoName).split('_')[0] + '_' + str(videoName).split('_')[1])
            subjectName.append(str(videoName).split('_')[0])
        codeFinal['videoCode'] = videoNames
        codeFinal['subjectCode'] = subjectName
        
        #Synchronize the columns name with CAS(ME)^2
        codeFinal.rename(columns={'Type':'type', 'Onset':'onset', 'Offset':'offset', 'Apex':'apex'}, inplace=True) 
        
    # print('Data Columns:', codeFinal.columns) #Final data column
    return codeFinal

def load_ground_truth(dataset_name, subjects, subjectsVideos, codeFinal):
    # Due to the different column name in excel file
    dataset_expression_type = 'micro-expression'
    if(dataset_name == 'SAMMLV'):
        dataset_expression_type = 'Micro - 1/2'
        
    vid_need = []
    vid_count = 0
    ground_truth = []
    for sub_video_each_index, sub_vid_each in enumerate(subjectsVideos):
        ground_truth.append([])
        for videoIndex, videoCode in enumerate(sub_vid_each):
            on_off = []
            for i, row in codeFinal.iterrows():
                if (row['subjectCode']==subjects[sub_video_each_index]): #S15, S16... for CAS(ME)^2, 001, 002... for SAMMLV
                    if (row['videoCode']==videoCode):
                        if (row['type']==dataset_expression_type): #Micro-expression or macro-expression
                            if (row['offset']==0): #Take apex if offset is 0
                                on_off.append([int(row['onset']-1), int(row['apex']-1)])
                            else:
                                if(dataset_expression_type!='Macro' or int(row['onset'])!=0): #Ignore the samples that is extremely long in SAMMLV
                                    on_off.append([int(row['onset']-1), int(row['offset']-1)])
            if(len(on_off)>0):
                vid_need.append(vid_count) #To get the video that is needed
            ground_truth[-1].append(on_off) 
            vid_count+=1

    #Remove unused video
    final_samples = []
    final_videos = []
    final_subjects = []
    count = 0
    for subjectIndex, subject in enumerate(ground_truth):
        final_samples.append([])
        final_videos.append([])
        for samplesIndex, samples in enumerate(subject):
            if (count in vid_need):
                final_samples[-1].append(samples)
                final_videos[-1].append(subjectsVideos[subjectIndex][samplesIndex])
                final_subjects.append(subjects[subjectIndex])
            count += 1

    #Remove the empty data in array
    final_subjects = np.unique(final_subjects)
    final_videos = [ele for ele in final_videos if ele != []]
    final_samples = [ele for ele in final_samples if ele != []]

    print('Ground Truth Data')
    print('Subjects Name', final_subjects)
    print('Videos Name: ', final_videos)
    print('Samples [Onset, Offset]: ', final_samples)

    return final_subjects, final_videos, final_samples

# Pseudo-labeling
def pseudo_label(dataset_name, final_samples, dataset):
    # Set value of k
    if dataset_name == 'CASME_sq':
        k = 6
    elif dataset_name == 'SAMMLV':
        k = 37

    pseudo_y = []
    video_count = 0 

    for subject in final_samples:
        for video in subject:
            samples_arr = []
            if (len(video)==0):
                pseudo_y.append([0 for i in range(len(dataset[video_count]))])
            else:
                pseudo_y_each = [0]*(len(dataset[video_count]))
                for ME in video:
                    samples_arr.append(np.arange(ME[0]+1, ME[1]+1))
                for ground_truth_arr in samples_arr: 
                    for index in range(len(pseudo_y_each)):
                        pseudo_arr = np.arange(index, index+k) 
                        # Equivalent to if IoU>0 then y=1, else y=0
                        if (pseudo_y_each[index] < len(np.intersect1d(pseudo_arr, ground_truth_arr))/len(np.union1d(pseudo_arr, ground_truth_arr))):
                            pseudo_y_each[index] = 1 
                pseudo_y.append(pseudo_y_each)
            video_count+=1

    # Integrate all videos into one dataset
    pseudo_y = [y for x in pseudo_y for y in x]
    print('Total frames:', len(pseudo_y))

    return pseudo_y, k

# Prepare train and test sets
def prepare_dataset(dataset, pseudo_y, final_samples):
    Y = np.array(pseudo_y)
    videos_len = []
    groupsLabel = Y.copy()
    prevIndex = 0
    countVideos = 0

    #Get total frames of each video
    for video_index in range(len(dataset)):
        videos_len.append(len(dataset[video_index]))

    print('Frame Index for each subject:-')
    for video_index in range(len(final_samples)):
        countVideos += len(final_samples[video_index])
        index = sum(videos_len[:countVideos])
        groupsLabel[prevIndex:index] = video_index
        print('Subject', video_index, ':', prevIndex, '->', index)
        prevIndex = index

    X = [frame for video in dataset for frame in video]
    print('\nTotal X:', len(X), ', Total y:', len(Y))
    return X, Y, groupsLabel