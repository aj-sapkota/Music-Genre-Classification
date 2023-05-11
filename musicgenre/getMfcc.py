import librosa
import torch
import math
import json
import numpy


    # print(num_segments)
SAMPLE_RATE = 22050

def getmfcc(audio_file, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    data = {
        "mfcc": []
    }
    signal,  sample_rate = librosa.load(audio_file)
    TRACK_DURATION = librosa.get_duration(
        y=signal, sr=sample_rate, S=None, n_fft=n_fft, hop_length=hop_length, center=True)

    temp = TRACK_DURATION // 30
    # print(temp)
    
    
    TRACK_DURATION = 30 * temp
    # print(TRACK_DURATION)

    num_segments = 6 * temp
    num_segments = int(num_segments)
    # print(num_segments)

    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(
            y= signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
    
    with open(json_path, "w+") as fp:
        json.dump(data, fp, indent=4)
