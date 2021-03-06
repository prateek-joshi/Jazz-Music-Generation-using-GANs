import os
import librosa
import librosa.display
import numpy as np
import json
import argparse
import cv2


parser = argparse.ArgumentParser(description='Extract mfcc from audio files.')
parser.add_argument('--data_path', required=True, help='Path to folder containing the wav files.')
parser.add_argument('--json_path', required=True, help='Save path of generated json file.')
parser.add_argument('--num_segments', required=False, default=5, help='Number of segments of division.')
arg = parser.parse_args()

DATASET_PATH = arg.data_path
JSON_PATH = arg.json_path
SAMPLE_RATE = 22050
DURATION = 30 # seconds, specific to this dataset
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_SEGMENTS = int(arg.num_segments)


def calculate_sgram(data, sr):
    specgram = librosa.stft(data)

    # use the mel-scale instead of raw frequency
    sgram_mag, _ = librosa.magphase(specgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
    # librosa.display.specshow(mel_scale_sgram)
    # plt.show()

    # use the decibel scale to get the final Mel Spectrogram
    mel_decibel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    # print('db shape ', mel_decibel_sgram.shape)
    # librosa.display.specshow(mel_decibel_sgram, sr=sr, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()

    reshaped = cv2.resize(mel_decibel_sgram,(mel_decibel_sgram.shape[0]*2,mel_decibel_sgram.shape[0]),interpolation=cv2.INTER_AREA) # Reshape sizes specific to num_segments=5
    # print('reshaped ', reshaped.shape)
    # librosa.display.specshow(reshaped, sr=sr, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()

    return reshaped

def save_sgrams(datapath, json_path, num_segments):
    data_dict = { 'sgram': [] }
    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)

    for filename in os.listdir(datapath):
        filepath = os.path.join(datapath,filename)
        print(f'Processing file: {filepath}')
        data, sr = librosa.load(filepath, sr=SAMPLE_RATE)

        # segment each track and extract specgrams
        for s in range(num_segments):
            start_sample = s * num_samples_per_segment
            finish_sample = start_sample + num_samples_per_segment
            mel_sgram = calculate_sgram(data[start_sample:finish_sample], sr)
            data_dict['sgram'].append(mel_sgram.tolist())

    # write dictionary into a json file
    with open(json_path, 'w+') as f:
        json.dump(data_dict, f, indent=4)


if __name__=='__main__':
    # print(os.getcwd())
    # os.chdir('/content')
    # print(os.getcwd())
    save_sgrams(DATASET_PATH,JSON_PATH,num_segments=NUM_SEGMENTS)