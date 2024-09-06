

def extract_openSMILE_features(SMILExtract, conf_file,
                               audio_path, feat_path,
                               feat_file):

    # Import libraries
    import os
    import sys
    import csv
    import pandas as pd
    import numpy as np

    import opensmile

   
    # Create the feature directory if it doesn't exist
    os.makedirs(feat_path, exist_ok = True)

    # List all .wav files in the audio directory
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

    # Initialise array to store average features for each clip
    data = []

    # for each .wav file in the audio directory
    for file in audio_files:

        # Construct the full path for each .wav file
        full_audio_path = os.path.join(audio_path, file)

        # Extract the base name without extension
        base_file_name = os.path.splitext(file)[0]

        # Construct the CSV file name
        csv_file_name = base_file_name + '.csv'

        # extract features and save them in a .csv file
        cmd = SMILExtract + ' - C ' + conf_file + ' -I ' + full_audio_path + ' -O ' + feat_path + csv_file_name
        
        # Calculate the mean for each feature and send to data array
        with open(feat_path+csv_file_name) as file_name:
            array = np.loadtxt(file_name, skiprows=1, delimiter=",")
            array = np.transpose(array)

    # replace 0 values with NaN
    array[array==0] = np.nan  

    # calculate array with mean for all features per clip ignoring nan
    np.append(data,np.nanmean(array,axis=1),axis=0)

    # get a list of the different feature names
    eg = pd.read_csv(feat_path + csv_file_name)
    list_of_feature_names = list(eg.columns)

    # save the mean data array a csv file
    df = pd.DataFrame(data)
    df.to_csv(feat_file,index=False, header=list_of_feature_names)

    return feat_file




def extract_wav2vec2_features(w2v_model_name,
                              audio_path, feat_path):


    # Import libraries
    import os
    import glob
    import pandas as pd
    import numpy as np

    import torch, torchaudio

    import tensorflow as tf
    from transformers import AutoModel, AutoTokenizer

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    from transformers import Wav2Vec2Model

    from huggingsound import SpeechRecognitionModel

    from transformers import DistilBertTokenizer, TFDistilBertModel
     
    # Set up the device for PyTorch operations.
    # If a CUDA-compatible GPU is available, use it for accelerated computing; otherwise, use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the Wav2Vec2 feature extractor.
    # This is used for processing audio data and extracting features relevant for audio analysis.
    w2v_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(w2v_model_name)

    # Load the Wav2Vec2 model, for a specific language
    # This model is designed for understanding and processing the spoken language in audio format.
    w2vec_model = Wav2Vec2Model.from_pretrained(w2v_model_name).to(device)

    # Load the Wav2Vec2 processor.
    # This processor combines the feature extractor and the model for streamlined audio data processing.
    processor = Wav2Vec2Processor.from_pretrained(w2v_model_name)

    # Create the CSV directory if it doesn't exist
    os.makedirs(feat_path, exist_ok=True)

    # List all WAV files in the audio directory
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

    for file in audio_files:
        # Construct the full path for each audio file
        full_audio_path = os.path.join(audio_path, file)

        # Load the audio file
        waveform, original_sample_rate = torchaudio.load(full_audio_path)
        
        # Your existing processing code
        resampled_waveform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=16000)(waveform)
        inputs = w2v_feature_extractor(resampled_waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = w2vec_model(inputs.input_values.squeeze(0).to(device))
 
        # For extracting features from a specific layer (e.g., the last layer):
        w2v_features = out.last_hidden_state

        # Convert tensor to numpy and then to DataFrame
        features_numpy = w2v_features.squeeze(0).cpu().numpy()
        df = pd.DataFrame(features_numpy)

        # Extract the base name without extension
        base_file_name = os.path.splitext(file)[0]

        # Construct the CSV file name
        csv_file_name = base_file_name + '.csv'
 
        # Save to CSV in the Wave2vec2_features directory
        csv_full_path = os.path.join(feat_path, csv_file_name)
        df.to_csv(csv_full_path, index=False)

















