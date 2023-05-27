# encoding=utf8
###############
## Created by: Chitralekha Gupta
## Affiliation: NUS, Singapore
## Last edited on: 9th May 2018
## Last edited by: Chitralekha Gupta

## Please refer to the following paper for details:
## Gupta, C., Li, H. and Wang, Y., 2017, December.
## Perceptual Evaluation of Singing Quality.
## In Proceedings of APSIPA Annual Summit and Conference (Vol. 2017, pp. 12-15).
###############


from get_mfcc_dtw import get_features
import os
import numpy as np
import pickle
import time
import converter

###############
## Flags to be set by the user
single_or_group = 0  # single file: 0; group: 1
feature_extraction = 1  # when single_or_group flag is set to 1, and this flag is set to 1, it extracts 20 features for all the audio files. The features are listed here, as well as in the file: get_mfcc_dtw.py
tag = 'overall'  # this tag denotes the ground-truth (GT) score. It can be overall, pitch, rhythm, vibrato, volume, vqual, pronunciation, pitchDynamicRange, and overall
reference_singer = 'MCUR'  # this is the reference singer in case of group mode
###############
#######################################################
## Final Features are:
## raw_disturbance_features
# 1. Rhythm Disturbance (L2-norm)
# 2. Raw Pitch Disturbance (L2-norm)
# 3. Pitch-derivative Disturbance (L2-norm)
# 4. Pitch Median subtracted Disturbance (L2-norm)
# 5. Vibrato-frame-based Disturbance (L2-norm)

## perceptual_disturbance_features
# 1. Rhythm Disturbance (L6+L2-norm)
# 2. Raw Pitch Disturbance (L6+L2-norm)
# 3. Pitch-derivative Disturbance (L6+L2-norm)
# 4. Pitch Median subtracted Disturbance (L6+L2-norm)
# 5. Vibrato-frame-based Disturbance (L6+L2-norm)

## distance_features
# 1. Timbral Difference (Distance)
# 2. Raw Pitch Difference (Distance)
# 3. Pitch-derivative Difference (Distance)
# 4. Pitch-median subtracted Difference (Distance)
# 5. Vibrato-segment difference (Distance)
# 6. Vibrato: whole pitch contour frame-based evaluation (Distance)
# 7. Volume Distance
# 8. Pitch Dynamic Range
# 9. E.Molina's Rhythm Distance
# 10. E.Molina's Rhythm Distance based on pitch
#######################################################
groundtruthfolder = 'groundtruthfiles'
output_arff_folder = 'output_arffs'

if not os.path.exists(output_arff_folder):
    os.mkdir(output_arff_folder)

t = time.time()


def get_features_per_song(folder, GT_file):
    ## This function extracts the features for all the test singers of one song
    ref_files = []
    test_files = []
    data = []
    for dir, sub, files in os.walk(folder):
        for file in files:
            if '.wav' not in file: continue
            if reference_singer in file:
                ref_files.append(dir + os.sep + file)
            else:
                test_files.append(dir + os.sep + file)

    fin = open(GT_file, 'r')
    GT_file_score = np.array([])
    for line in fin.readlines():
        if GT_file_score.size == 0:
            GT_file_score = np.array(line.split('\t'))
        else:
            GT_file_score = np.vstack((GT_file_score, np.array(line.split('\t'))))

    for test in test_files:
        segment_num = test.split('_')[-1]
        original = [s for s in ref_files if segment_num in s][0]
        GT_score = float([s for s in GT_file_score if s[0] in test][0][1].rstrip('\n'))

        print(original)
        print(test)

        raw_disturbance_features, perceptual_disturbance_features, distance_features = get_features(original, test)
        data.append(list(raw_disturbance_features[0]) + list(perceptual_disturbance_features[0]) + list(
            distance_features[0]) + [GT_score])
    return data, test_files


def get_onlyGT(folder, GT_file):
    ### This function obtains the subjective ground truths from the groundtruth folder
    ref_files = []
    test_files = []
    GT = []
    for dir, sub, files in os.walk(folder):
        for file in files:
            if '.wav' not in file: continue
            if reference_singer in file:
                ref_files.append(dir + os.sep + file)
            else:
                test_files.append(dir + os.sep + file)

    fin = open(GT_file, 'r')
    GT_file_score = np.array([])
    for line in fin.readlines():
        if GT_file_score.size == 0:
            GT_file_score = np.array(line.split('\t'))
        else:
            GT_file_score = np.vstack((GT_file_score, np.array(line.split('\t'))))

    for test in test_files:
        segment_num = test.split('_')[-1]
        original = [s for s in ref_files if segment_num in s][0]
        GT_score = float([s for s in GT_file_score if s[0] in test][0][1].rstrip('\n'))
        # if GT_score>1.0 and GT_score<4.0: continue
        GT.append(GT_score)
    return GT, test_files


def getFeatures(original="wavfile_samples/reference_clip.wav", test='wavfile_samples/bad_clip.wav'):
    raw_disturbance_features, perceptual_disturbance_features, distance_features = get_features(original, test)
    result = list()
    print(raw_disturbance_features)
    print(raw_disturbance_features[0])
    print(perceptual_disturbance_features)
    for elem in raw_disturbance_features[0]:
        result.append(round(elem, 2))
    for elem in perceptual_disturbance_features[0]:
        result.append(round(elem, 2))
    for elem in distance_features[0]:
        result.append(round(elem, 2))
    return result

if __name__ == '__main__':
    #"wavfile_samples/reference_clip.wav"
    # scores = getFeatures("wavfile_samples/reference_clip.wav", "wavfile_samples/bad_clip.wav")
    # scores = getFeatures("1839262209ref.wav", "1839262209user.wav")
    sung_song_path = '/root/audioData/user/566443167user.wav'
    reference_song_path = '/root/audioData/music/566443167.mp3'
    print(sung_song_path)
    print(reference_song_path)

    begin_time = 1.001
    end_time = 10.001
    print(begin_time)
    print(end_time)

    os.system('cp ' + sung_song_path + ' ./sung.wav')
    os.system('cp ' + reference_song_path + ' ./ref.mp3')
    #converter.trans_mp3_to_wav('./sung.mp3', './sung.wav')
    converter.get_second_part_from_mp3_to_wav('./ref.mp3', float(begin_time), float(end_time), 'ref.wav')
    scores = np.array(getFeatures('sung.wav', 'ref.wav'))
    # os.system('rm -f ./sung.wav')
    # os.system('rm -f ./ref.wav')
    # os.system('rm -f ./ref.mp3')
    score = np.mean(scores)
    print(score)
