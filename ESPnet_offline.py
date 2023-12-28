import numpy as np
import pandas as pd
from panns_inference import AudioTagging
import os
from espnet2.bin.enh_inference import SeparateSpeech
import time
import torch
import string
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from src.core.offline_asr_espnet.SplitAudio import split_audio_using_VAD
from src.core.offline_asr_espnet.WithTranscript import with_transcript, solo_label_2
from src.core.offline_asr_espnet.CreateDict import create_dict
def text_normalizer(text):
        text = text.upper()
        return text.translate(str.maketrans('', '', string.punctuation))

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True)
# download example
# torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')
def generate_transcription(s2t_config_file, 
                           s2t_model_file, 
                           audio_path, 
                           speech_enh_train_conf, 
                           speech_enh_model_file,
                           transcript=None,
                           lm_file=None,
                           lm_train_config=None, 
                           frame_length=7, 
                           music_tolerance=0.8, 
                           speech_limit=0.1, 
                           transcript_file=None, 
                           output_arpa_file="6gram1.arpa",
                           ):
    if transcript_file is not None:
        os.system(f'/home/suryansh/kenlm/build/bin/lmplz -o 6 --discount_fallback <{transcript_file}> {output_arpa_file}')
        os.chdir('../models/asr_train_asr_raw_hindi_bpe500')
        # It may takes a while to download and build models
        speech2text = Speech2Text(
            asr_train_config=s2t_config_file,
            asr_model_file=s2t_model_file,
            lm_train_config=lm_train_config,
            lm_file=lm_file,
            ngram_file=output_arpa_file,
            ngram_weight = 0.8,
            device="cuda",
            minlenratio=0.0,
            maxlenratio=0.5,
            ctc_weight=0.3,
            beam_size=10,
            batch_size=0,
            nbest=1
        )
    else:
        # It may takes a while to download and build models
        speech2text = Speech2Text(
            asr_train_config=s2t_config_file,
            asr_model_file=s2t_model_file,
            lm_train_config=lm_train_config,
            lm_file=lm_file,
            device="cuda",
            minlenratio=0.0,
            maxlenratio=0.5,
            ctc_weight=0.3,
            beam_size=10,
            batch_size=0,
            nbest=1
        )

    (get_speech_timestamps,
    _, read_audio,
    *_) = utils
    sampling_rate=16000

    os.chdir('../models')
    enh_model_sc = SeparateSpeech(
        train_config=speech_enh_train_conf,
        model_file=speech_enh_model_file,
        # for segment-wise process on long speech
        normalize_segment_scale=False,
        show_progressbar=True,
        ref_channel=1,
        normalize_output_wav=True,
        device="cuda:0",
    )  
    os.chdir('../models')
#     print(ground_truth)
    wav = read_audio(audio_path, sampling_rate=sampling_rate)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate, threshold=0.2)
    df = pd.DataFrame(speech_timestamps)
    df = df // 16
    # Split the audio into 15-second chunks with adjustment for the last chunk
    chunks_array, split_sec = split_audio_using_VAD(audio_path, df, frame_length)
    os.chdir('../models/asr_train_asr_raw_hindi_bpe500')
    preds = ""
    at = AudioTagging(checkpoint_path=None, device='cuda')
    timestamps_and_transcripts = []
    for i, chunk in enumerate(chunks_array):
        start_time = time.time()
        speech = np.array([])
        duration=split_sec['end'][i]*1000-split_sec['start'][i] * 1000
        for _, row in df.iterrows():
            start_sample = row['start']-split_sec['start'][i] * 1000
            end_sample = row['end']-split_sec['start'][i] * 1000
    #         print(start_sample, end_sample, duration)
            if(start_sample<0 and end_sample<0):
                continue
            if(start_sample>duration and end_sample>duration):
                break
    #         print("Y")
            speech = np.concatenate([speech, chunk[int(max(0,start_sample))*16:int(min(duration, end_sample))*16]])

        print(len(speech))
    #     nbests = speech2text(chunk)
        if len(speech) >= 4000:
            a=speech.reshape((speech.shape[0], 1))
            mixwav_sc = a[:,0]
            wave = mixwav_sc[None, :]  # (batch_size, segment_samples)
#             at = AudioTagging(checkpoint_path=None, device='cuda')
            (clipwise_output, embedding) = at.inference(wave)
            print(clipwise_output[0][137])
            if clipwise_output[0][137]>=music_tolerance and clipwise_output[0][0]>=speech_limit:
                print('Enhancement Required')
                wave = enh_model_sc(mixwav_sc[None, ...], 16000)
            a=wave[0].squeeze()
            nbests = speech2text(a)
            text, *_ = nbests[0]
    #         output_filename = f'results/output_chunk{i}.wav'
    #         sf.write(output_filename, speech, samplerate=16000)
            timestamps_and_transcripts.append(create_dict(split_sec['start'][i], split_sec['end'][i], text_normalizer(text)))
            if(preds==""):
                preds=(text_normalizer(text))
            else:
                preds += " " + (text_normalizer(text))

        print(i, "/", len(chunks_array))
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")
        if transcript is not None:
            target_key = '1_transcript'
            new_values=with_transcript(transcript, solo_label_2(timestamps_and_transcripts), 5)
            for i, d in enumerate(timestamps_and_transcripts):
                d[target_key] = new_values[i]
            preds=" ".join(new_values)
    return preds, timestamps_and_transcripts

def modified_transcription(s2t_config_file, 
                           s2t_model_file, 
                           audio_path, 
                           speech_enh_train_conf, 
                           speech_enh_model_file,
                           transcript,
                           lm_file=None,
                           lm_train_config=None, 
                           frame_length=7, 
                           music_tolerance=0.8, 
                           speech_limit=0.1, 
                           transcript_file=None, 
                           output_arpa_file="6gram1.arpa",
                           ):
    text2, _=generate_transcription(s2t_config_file,
                            s2t_model_file,
                            audio_path,
                            speech_enh_train_conf,
                            speech_enh_model_file,
                            lm_file,
                            lm_train_config,
                            frame_length,
                            music_tolerance,
                            speech_limit,
                            transcript_file,
                            output_arpa_file
                            )
    return with_transcript(transcript, text2, 6)

def runInference(language,
                 audio_path,
                frame_length=7, 
                music_tolerance=0.8, 
                speech_limit=0.1, 
                transcript_file=None, 
                output_arpa_file="6gram1.arpa",
                ):
    if language=="hindi":
        s2t_config_file = '../models/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_hindi_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_hindi_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="telugu":
        s2t_config_file = '../models/asr_train_asr_raw_telugu_bpe500/exp/asr_train_asr_raw_telugu_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_telugu_bpe500/exp/asr_train_asr_raw_telugu_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_telugu_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_telugu_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="urdu":
        s2t_config_file = '../models/asr_train_asr_raw_urdu_bpe500/exp/asr_train_asr_raw_urdu_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_urdu_bpe500/exp/asr_train_asr_raw_urdu_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_urdu_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_urdu_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="malayalam":
        s2t_config_file = '../models/asr_train_asr_raw_malayalam_bpe500/exp/asr_train_asr_raw_malayalam_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_malayalam_bpe500/exp/asr_train_asr_raw_malayalam_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_malayalam_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_malayalam_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="tamil":
        s2t_config_file = '../models/asr_train_asr_raw_tamil_bpe500/exp/asr_train_asr_raw_tamil_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_tamil_bpe500/exp/asr_train_asr_raw_tamil_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_tamil_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_tamil_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="kannada":
        s2t_config_file = '../models/asr_train_asr_raw_kannada_bpe500/exp/asr_train_asr_raw_kannada_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_kannada_bpe500/exp/asr_train_asr_raw_kannada_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_kannada_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_kannada_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="marathi":
        s2t_config_file = '../models/asr_train_asr_raw_marathi_bpe500/exp/asr_train_asr_raw_marathi_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_marathi_bpe500/exp/asr_train_asr_raw_marathi_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_marathi_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_marathi_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="odia":
        s2t_config_file = '../models/asr_train_asr_raw_odia_bpe500/exp/asr_train_asr_raw_odia_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_odia_bpe500/exp/asr_train_asr_raw_odia_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_odia_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_odia_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="bengali":
        s2t_config_file = '../models/asr_train_asr_raw_bengali_bpe500/exp/asr_train_asr_raw_bengali_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_bengali_bpe500/exp/asr_train_asr_raw_bengali_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_bengali_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_bengali_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="gujrati":
        s2t_config_file = '../models/asr_train_asr_raw_gujrati_bpe500/exp/asr_train_asr_raw_gujrati_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_gujrati_bpe500/exp/asr_train_asr_raw_gujrati_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_gujrati_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_gujrati_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="punjabi":
        s2t_config_file = '../models/asr_train_asr_raw_punjabi_bpe500/exp/asr_train_asr_raw_punjabi_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_punjabi_bpe500/exp/asr_train_asr_raw_punjabi_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_punjabi_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_punjabi_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None
    elif language=="sanskrit":
        s2t_config_file = '../models/asr_train_asr_raw_sanskrit_bpe500/exp/asr_train_asr_raw_sanskrit_bpe500/config.yaml'
        s2t_model_file = '../models/asr_train_asr_raw_sanskrit_bpe500/exp/asr_train_asr_raw_sanskrit_bpe500/valid.acc.ave_10best.pth'
        speech_enh_train_conf = '../models/enh_train_enh_conv_tasnet_raw_sanskrit_bpe500/config.yaml'
        speech_enh_model_file = '../models/enh_train_enh_conv_tasnet_raw_sanskrit_bpe500/valid.loss.best.pth'
        lm_file=None
        lm_train_config=None



    text, timestamps_and_transcripts=generate_transcription(s2t_config_file,
                            s2t_model_file,
                            audio_path,
                            speech_enh_train_conf,
                            speech_enh_model_file,
                            lm_file,
                            lm_train_config,
                            frame_length,
                            music_tolerance,
                            speech_limit,
                            transcript_file,
                            output_arpa_file
                            )
    return timestamps_and_transcripts