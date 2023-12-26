import librosa
import pandas as pd
import numpy as np
import os


def split_audio_using_VAD(input_file, pred, chunk_length_sec=15):
    # Load the audio file
    audio, sr = librosa.load(input_file, sr=None)
    timestamps_df=pd.DataFrame()
    timestamps_df['start']=pred['start'].apply(lambda x: x*sr/1000)
    timestamps_df['end']=pred['end'].apply(lambda x: x*sr/1000)
    print(timestamps_df)
    # Calculate the total duration of the audio in samples
    total_samples = len(audio)

    # Calculate the chunk length in samples
    chunk_length_samples = int(sr * chunk_length_sec)
    print(chunk_length_samples)
    # Create an array to store audio chunks
    audio_chunks = []
    start_sample=0
    end_sample=chunk_length_samples
    split_sec=[]
    k=0
    index=0
    for i, row in timestamps_df.iterrows():
        if (start_sample+chunk_length_samples)>=row['end']:
            end_sample=row['end']
        else:
            audio_chunks.append(audio[int(start_sample):int(end_sample)])
            split_sec.append((index, int(start_sample)/sr, int(end_sample)/sr))
            index+=1
            start_sample=end_sample
            end_sample=row['end']
            while(start_sample+chunk_length_samples<row['end']):
                end_sample=start_sample+chunk_length_samples
                audio_chunks.append(audio[int(start_sample):int(end_sample)])
                split_sec.append((index, int(start_sample)/sr, int(end_sample)/sr))
                start_sample=end_sample
                index+=1
                end_sample=row['end']
            

    last_chunk = audio[int(start_sample):int(total_samples)]
    audio_chunks.append(last_chunk)
    split_sec.append((index, start_sample/sr, total_samples/sr))
    split_sec=pd.DataFrame(split_sec, columns=['index', 'start', 'end'])
    return audio_chunks, split_sec
