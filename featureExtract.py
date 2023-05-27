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
# from praatio import tgio

from get_mfcc_dtw import *

class featureExtract:
    def __init__(self):
        self.num_mfcc = 13
        self.NFFT = 512
        self.shift = 1
        
    def forward_once(self,original,test):
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

        self.mfccDTW = distance
        ############ MFCC frame disturbance array ##############
        mfcc_frame_disturbance = FrameDisturbance(path)

        self.timbralDifference = distance
        self.rhythmDisturbance = np.linalg.norm(mfcc_frame_disturbance, ord=2)
        self.perceptualRhythmDisturbance = CalcPESQnorm(mfcc_frame_disturbance)


        

        #############VOLUME####################################
        volume_dist = VolumeDistance(ori_sig, test_sig, rate)
        vibrato_section_disturbance = FrameDisturbance(path)

        self.vibratoDist = np.linalg.norm(vibrato_section_disturbance, ord=2)
        self.perceptualVibratoDist = CalcPESQnorm(vibrato_section_disturbance) # L2+L6-norm
        self.vibratoDiff = distance

        ### Append Volume Distance Feature
        self.volumeDist = volume_dist

        #############EMOLINA's Rhythm Calc#####################
        ## E. Molina, I. Barbancho, E. Gomez, A. M. Barbancho, and
        # L. J. Tardon, "Fundamental frequency alignment vs. note-based
        # melodic similarity for singing voice assessment," IEEE ICASSP, pp.
        # 744-748, 2013.
        emolina_rhythm_mfcc_distance = EmolinaRhythm_mfcc(ori_sig, test_sig, rate, window)
        ### Emolina's rhythm distance
        self.emolinaRhythm = emolina_rhythm_mfcc_distance
        
        self.distance_features = [self.timbralDifference, self.vibratoDiff,self.emolinaRhythm,self.volumeDist]
        self.raw_disturbance_features = [self.rhythmDisturbance, self.vibratoDist]
        self.perceptual_disturbance_features = [self.perceptualRhythmDisturbance, self.perceptualVibratoDist]