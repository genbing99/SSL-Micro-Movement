import pickle
import numpy as np
from collections import Counter
from tensorflow.keras.utils import to_categorical

# Load pkl
def load_data(dataset_name):
    if dataset_name == 'Composite':
        final_subjects = pickle.load( open( "dataset/Composite_subjects.pkl", "rb" ) )
        final_dataset = pickle.load( open( "dataset/Composite_dataset.pkl", "rb" ) )
        final_emotions = pickle.load( open( "dataset/Composite_emotions.pkl", "rb" ) )
        final_samples = pickle.load( open( "dataset/Composite_samples.pkl", "rb" ) )
        print('Total emotions in each class:', Counter([ele for ele in final_emotions for ele in ele for ele in ele]))
    else:
        final_subjects = pickle.load( open( "dataset/MMEW_subjects.pkl", "rb" ) )
        final_dataset = pickle.load( open( "dataset/MMEW_dataset.pkl", "rb" ) )
        final_emotions = pickle.load( open( "dataset/MMEW_emotions.pkl", "rb" ) )
        final_samples = pickle.load( open( "dataset/MMEW_samples.pkl", "rb" ) )
        print('Total emotions in each class:', Counter([ele for ele in final_emotions for ele in ele]))

    print('Total subjects: ', len(final_subjects))
    print('Total samples:', len([ele for ele in final_samples for ele in ele]))
    return final_subjects, final_dataset, final_emotions, final_samples
    
# Prepare train and test sets
def prepare_dataset(final_dataset, final_emotions, final_samples, dataset_name):
    X = [frame for frame in final_dataset]

    # Due to slightly different format in different dataset
    if dataset_name == 'Composite': 
        y = [samples for subjects in final_emotions for videos in subjects for samples in videos]
    else:
        y = [samples for videos in final_emotions for samples in videos]
    y = np.array(y) 

    # Convert 0, 1, 2 to negative, positive, surprise
    y = [0 if ele=='negative' else ele for ele in y]
    y = [1 if ele=='positive' else ele for ele in y]
    y = [2 if ele=='surprise' else ele for ele in y]
        
    y = to_categorical(y)
    groupsLabel = []

    # Get samples for each subject
    print('\nSample Index for each subject (ME Recognition):-')

    # Due to slightly different format in different dataset
    if dataset_name == 'Composite': 
        for subject_index in range(len(final_samples)):
            for video_index in range(len(final_samples[subject_index])):
                for sample_index in range(len(final_samples[subject_index][video_index])):
                    groupsLabel.append(subject_index)
    else:
        for subject_index in range(len(final_samples)):
            for video_index in range(len(final_samples[subject_index])):
                groupsLabel.append(subject_index)
                
    return X, y, groupsLabel