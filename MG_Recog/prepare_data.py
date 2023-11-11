import pickle
import numpy as np
from collections import Counter
from tensorflow.keras.utils import to_categorical

# Load pkl
def load_data():
    final_dataset = pickle.load( open( "dataset/imigue_dataset.pkl", "rb" ) )
    final_videos = pickle.load( open( "dataset/imigue_videos.pkl", "rb" ) )
    final_subjects = pickle.load( open( "dataset/imigue_subjects.pkl", "rb" ) )
    final_emotions = pickle.load( open( "dataset/imigue_emotions.pkl", "rb" ) )

    print('Total subjects: ', len(final_subjects))
    print('Total videos:', len([ele for ele in final_videos for ele in ele]))
    print('Total emotions in each class:', Counter([ele for ele in final_emotions for ele in ele]))

    return final_subjects, final_dataset, final_emotions, final_videos

# Prepare train and test sets
def prepare_dataset(final_subjects, final_dataset, final_emotions, final_videos):
    X = [frame for frame in final_dataset]

    y = [ele for ele in final_emotions for ele in ele]
    y = np.array(y) 

    # Convert class 99 to 32
    y = [0 if ele==99 else ele for ele in y]
        
    y = to_categorical(y)
    # print('Total X :', len(X))
    # print('Total y :', len(y))
    # print('Total Class:', len(y[0]))
    groupsLabel = []

    #Get samples for each subject
    print('\nSample Index for each subject (MG Recognition):-')
    for subject_index in range(len(final_subjects)):
        for video_index in range(len(final_videos[subject_index])):
            groupsLabel.append(final_subjects[subject_index])
        # print('Subject', final_subjects[subject_index], ':', len(groupsLabel)-len(final_videos[subject_index]), '->', len(groupsLabel)-1)
    
    return X, y, groupsLabel