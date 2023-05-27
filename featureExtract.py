from mfcc_copy import mfcc
from mfcc_copy import delta
import scipy.io.wavfile as wav
import numpy as np
from fastdtw import fastdtw
from matplotlib import pylab as plt
from itertools import chain
import os
import math
import decimal
import csv
from scipy.spatial.distance import euclidean
from scipy.fftpack import fft
# from praatio import tgio
from praatio import pitch_and_intensity
from math import sqrt

import converter
# from praatio import tgio

from get_mfcc_dtw import *

class featureExtract:
    def __init__(self):
        self.num_mfcc = 13
        self.NFFT = 512
        self.shift = 1
        
        
    def forward_once(self,original,test,type = 'mp3'):
        if type != 'wav':
            converter.get_second_part_from_mp3_to_wav(original, 1.001, 10.001, 'origin.wav')
            converter.get_second_part_from_mp3_to_wav(test, 1.001, 10.001, 'test.wav')
            original = 'origin.wav'
            test = 'test.wav'
            
        ######### Original file reading and mfcc computation ##########
        (rate, ori_sig) = wav.read(original)
        ori_sig = ori_sig / 32768.0
        ori_sig = ori_sig - np.mean(ori_sig)  # remove DC offset
        window = NFFT / (rate * 1.0)
        hop = window / 2.0
        ori_sig = InitialFinalSilenceRemoved(ori_sig)
        num_ori_frames = int(np.floor(len(ori_sig) / (rate * hop)))

        mfcc_original = mfcc(ori_sig, rate * 1.0, winlen=window, winstep=hop, nfft=NFFT, numcep=13)
        if num_mfcc == 39:
            d_mfcc_feat1 = delta(mfcc_original, 2)
            d_mfcc_feat2 = delta(d_mfcc_feat1, 2)
            mfcc_original = np.hstack((mfcc_original, d_mfcc_feat1, d_mfcc_feat2))

        ######### Test file reading and mfcc computation ##########
        (rate, test_sig) = wav.read(test)
        test_sig = test_sig / 32768.0
        test_sig = test_sig - np.mean(test_sig)  # remove DC offset
        test_sig = InitialFinalSilenceRemoved(test_sig)
        window = NFFT / (rate * 1.0)
        Nh = np.ceil((len(test_sig) - NFFT) / (
                num_ori_frames - 1))  # number of samples in a hop or frame shift # Make the two files of (approx) equal number of frames
        hop_test = Nh / rate  # hop in time #window/2.0 #
        mfcc_test = mfcc(test_sig, rate * 1.0, winlen=window, winstep=hop_test, nfft=NFFT,
                        numcep=13)  # test sig truncated by hop size because overlap is now different from 50%
        if num_mfcc == 39:
            d_mfcc_feat1 = delta(mfcc_test, 2)
            d_mfcc_feat2 = delta(mfcc_test, 2)
            mfcc_test = np.hstack((mfcc_test, d_mfcc_feat1, d_mfcc_feat2))

        ########## Compute DTW on mfcc ################
        distance, path = fastdtw(mfcc_original, mfcc_test, radius=1, dist=mfcc_dist)
        ############ MFCC frame disturbance array ##############
        # optimal path of mfcc frames
        mfcc_frame_disturbance = FrameDisturbance(path)

        self.timbralDifference = distance
        # print("timbralDifference: ",distance)
        # L2 norm of mfcc frame disturbance
        self.rhythmDisturbance = np.linalg.norm(mfcc_frame_disturbance, ord=2)
        # print("rhythmDisturbance: ",self.rhythmDisturbance)
        # L2+L6-norm of mfcc frame disturbance
        self.perceptualRhythmDisturbance = CalcPESQnorm(mfcc_frame_disturbance)
        # print("perceptualRhythmDisturbance: ",self.perceptualRhythmDisturbance)


        #############VOLUME####################################
        volume_dist = VolumeDistance(ori_sig, test_sig, rate)
        # print("volume_dist: ",volume_dist)
        self.volumeDist = volume_dist

        #############EMOLINA's Rhythm Calc#####################
        ## E. Molina, I. Barbancho, E. Gomez, A. M. Barbancho, and
        # L. J. Tardon, "Fundamental frequency alignment vs. note-based
        # melodic similarity for singing voice assessment," IEEE ICASSP, pp.
        # 744-748, 2013.
        emolina_rhythm_mfcc_distance = EmolinaRhythm_mfcc(ori_sig, test_sig, rate, window)
        ### Emolina's rhythm distance
        self.emolinaRhythm = emolina_rhythm_mfcc_distance
        # print("emolinaRhythm: ",self.emolinaRhythm)
        
        self.distance_features = [self.timbralDifference,self.emolinaRhythm,self.volumeDist]
        self.raw_disturbance_features = [self.rhythmDisturbance]
        self.perceptual_disturbance_features = [self.perceptualRhythmDisturbance]
        
        # delete the wav file
        os.remove(original)
        os.remove(test)
        
    def total_score(self):
        dist = self.timbralDifference + self.emolinaRhythm + self.volumeDist
        score = 10000000 / dist
        score = sqrt(score*100)
        # clip the score to range [0,100]
        if score > 100:
            score = 100
        elif score < 0:
            score = 0
        return score
    
    def print_all_scores(self):
        precise_dist = (self.timbralDifference + self.rhythmDisturbance + self.perceptualRhythmDisturbance)/3
        # clip the precise to range [0,80000]
        if precise_dist > 80000:
            precise_dist = 80000
        elif precise_dist < 0:
            precise_dist = 0
        precise_score = (80000 * 10+1) / (precise_dist+1)
        precise_score = sqrt(precise_score * 100)
        precise_score = precise_score + 40
        # clip the score to range [0,100]
        if precise_score > 100:
            precise_score = 100
        elif precise_score < 0:
            precise_score = 0
        print("precise_score")
        print(precise_score)
        
        quality_dist = self.emolinaRhythm
        # clip the quality to range [0,2000000]
        if quality_dist > 2000000:
            quality_dist = 2000000
        elif quality_dist < 0:
            quality_dist = 0
        quality_score = (2000000+1) / (quality_dist+1)
        quality_score = sqrt(quality_score * 100)
        quality_score = quality_score + 40
        # clip the score to range [0,100]
        if quality_score > 100:
            quality_score = 100
        elif quality_score < 0:
            quality_score = 0
        print("quality_score")
        print(quality_score)
        
        pitch_dist = self.volumeDist
        # clip the pitch to range [0,10000]
        if pitch_dist > 10000:
            pitch_dist = 10000
        elif pitch_dist < 0:
            pitch_dist = 0
        pitch_score = (10000 * 10+1) / (pitch_dist+1)
        pitch_score = sqrt(pitch_score * 100)
        pitch_score = pitch_score + 40
        # clip the score to range [0,100]
        if pitch_score > 100:
            pitch_score = 100
        elif pitch_score < 0:
            pitch_score = 0
        print("pitch_score")
        print(pitch_score)
        # print("total score: ",self.total_score())