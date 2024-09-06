# import necessary packages
import sys
import os
import io
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import Audio
import torch
from sklearn.model_selection import KFold
import keras
from tensorflow.keras.utils import plot_model
from IPython.display import Image
import xlsxwriter
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from scipy.stats import spearmanr


# Prepare a Tensorflow dataset
def dataframe_to_dataset_REG(dataframe, BATCH_SIZE):
  
    X = np.array(dataframe["values"].tolist())
    X = np.concatenate(X, axis=0)
    Y = np.array(dataframe["label"].tolist())
    Y = np.concatenate(Y, axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    print(X.shape, Y.shape)

    return dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Build the Model
def build_and_compile_model_REG(neurons, layers, l_rate, feat_dim, trait):
    if layers==1:
        inputs = Input(shape=(feat_dim), name="main_input")

        x = Dense(neurons, activation="relu", name="dense_1")(inputs)
        x = Dropout(0.1, name="dropout_1")(x)

    elif layers==2:
        inputs = Input(shape=(feat_dim), name="main_input")

        x = Dense(neurons, activation="relu", name="dense_1")(inputs)
        x = Dropout(0.1, name="dropout_1")(x)

        x = Dense(neurons, activation="relu", name="dense_2")(x)
        x = Dropout(0.2, name="dropout_2")(x)

    elif layers==3:
        inputs = Input(shape=(feat_dim), name="main_input")

        x = Dense(neurons, activation="relu", name="dense_1")(inputs)
        x = Dropout(0.1, name="dropout_1")(x)

        x = Dense(neurons, activation="relu", name="dense_2")(x)
        x = Dropout(0.2, name="dropout_2")(x)

        x = Dense(neurons, activation="relu", name="dense_3")(x)
        x = Dropout(0.2, name="dropout_3")(x)

    elif layers==4:
        inputs = Input(shape=(feat_dim), name="main_input")

        x = Dense(neurons, activation="relu", name="dense_1")(inputs)
        x = Dropout(0.1, name="dropout_1")(x)

        x = Dense(neurons, activation="relu", name="dense_2")(x)
        x = Dropout(0.2, name="dropout_2")(x)

        x = Dense(neurons, activation="relu", name="dense_3")(x)
        x = Dropout(0.2, name="dropout_3")(x)

        x = Dense(neurons, activation="relu", name="dense_4")(x)
        x = Dropout(0.2, name="dropout_4")(x)

    elif layers==5:
        inputs = Input(shape=(feat_dim), name="main_input")

        x = Dense(neurons, activation="relu", name="dense_1")(inputs)
        x = Dropout(0.1, name="dropout_1")(x)

        x = Dense(neurons, activation="relu", name="dense_2")(x)
        x = Dropout(0.2, name="dropout_2")(x)

        x = Dense(neurons, activation="relu", name="dense_3")(x)
        x = Dropout(0.2, name="dropout_3")(x)

        x = Dense(neurons, activation="relu", name="dense_4")(x)
        x = Dropout(0.2, name="dropout_4")(x)

        x = Dense(neurons, activation="relu", name="dense_5")(x)
        x = Dropout(0.2, name="dropout_5")(x)

    outputs = Dense(1, activation="linear", name="ouput")(x)

    model = Model(inputs=inputs, outputs=outputs, name=trait+"_perception_regression")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(name="mean_squared_error"), keras.metrics.MeanAbsoluteError(name="mean_absolute_error")],
    )
    
    return model


# Generalization Metrics
def get_metrics_REG(model, rep, valid_ds, loss_per_rep, mse_per_rep, mae_per_rep):
     
    scores = model.evaluate(valid_ds, verbose=0)

    names = model.metrics_names

    print(f'Scores for repetition {rep}:')

    for i in range(0, len(scores)):
        print(f'{names[i]} = {scores[i]}')

    loss_per_rep.append(scores[names.index('loss')])
    mse_per_rep.append(scores[names.index('mean_squared_error')])
    mae_per_rep.append(scores[names.index('mean_absolute_error')])
   
    return names, scores


# Calculate Spearman rank correlation between predictions and ground truth for each fold
def per_clip_metrics_REG(test, model, rep, dataframe, trait, pred_score_per_clip_overall, label_per_clip_overall):
    
    pred_score_per_clip_per_rep = []
    pred_score_per_frame_per_rep = []

    label_per_clip_per_rep = []

    test_files = []
    
    for t in test:        # for each clip t in the test set           
        res = model.predict(
            dataframe["values"][t],
            batch_size=None,
            verbose="1",
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False)

        for j in range(0,len(dataframe["values"][t])):       # for each frame j within a clip t
            pred_score_per_frame_per_rep.append(res[j])
                   
        pred_score_per_clip_per_rep.append(np.mean(pred_score_per_frame_per_rep))

        label_per_clip_per_rep.append(dataframe["label"][t][0])

        test_files.append(dataframe["filename"][t].split("/")[-1])
        
        # Overall predictions and labels
        pred_score_per_clip_overall.append(np.mean(pred_score_per_frame_per_rep))
        label_per_clip_overall.append(dataframe["label"][t][0])

    # Spearman rank Correlation 
    rho, p = spearmanr(pred_score_per_clip_per_rep, label_per_clip_per_rep)

    print(f'{trait}: repetition {rep} Spearman rank correlation rho = {rho}')
    print(f'{trait}: repetition {rep} Spearman rank correlation p = {p}')

    # Mean and stdev of error per clip
    label_per_clip_per_rep = np.array(label_per_clip_per_rep)
    pred_score_per_clip_per_rep = np.array(pred_score_per_clip_per_rep)
    
    mse_per_clip_per_rep = np.mean(np.square(label_per_clip_per_rep - pred_score_per_clip_per_rep))
    stdev_se_per_clip_per_rep = np.std(np.square(label_per_clip_per_rep - pred_score_per_clip_per_rep))
    
    mae_per_clip_per_rep = np.mean(np.absolute(label_per_clip_per_rep - pred_score_per_clip_per_rep))
    stdev_ae_per_clip_per_rep = np.std(np.absolute(label_per_clip_per_rep - pred_score_per_clip_per_rep))

    pred_per_rep = pred_score_per_clip_per_rep
    ground_truth = label_per_clip_per_rep

    scores = [mse_per_clip_per_rep, stdev_se_per_clip_per_rep, mae_per_clip_per_rep, stdev_ae_per_clip_per_rep, rho, p]

    metrics_names = ['mean_squared_error', 'stdev_squared_error', 'mean_absolute_error', 'stdev_absolute_error', 'Spearman_rho', 'Spearman_p']

    predictions = [test_files, ground_truth, pred_per_rep]
    
    return metrics_names, scores, predictions


def dnn_regression(EPOCHS, BATCH_SIZE, l_rate, TRAITS,
                   repetitions, layers, neurons, split_index,
                   feat_type, feat_path, labels_path, output_path, ):


    for trait in TRAITS:
        # Get the labels
        labs = open(labels_path+'labels_numerical_'+trait+'.txt','r')
        lab = labs.read()
        list_of_labels = lab.split('\n')


        l = 0
        # Load the data in a Dataframe
        dataframe = pd.DataFrame(columns=["filename","label"])
        for filename in os.listdir(feat_path):
            f = os.path.join(feat_path, filename)
            if os.path.isfile(f) and filename.endswith('csv'):
                if filename.split('.')[0].split('_channel_')[1]=='1':
                    # Create a dictionary with the data for the new row
                    new_row = {'filename': f, 'label': list_of_labels[l]}
                    # Inserting the new row
                    dataframe.loc[len(dataframe)] = new_row
                    # Reset the index
                    dataframe = dataframe.reset_index(drop=True)
            
                    l += 1

        # Remove leading space in filename column
        dataframe["filename"] = dataframe.apply(lambda row: row["filename"].strip(), axis=1)

        # Stack the data from channel 1 and channel 2
        dataframe["values"] = dataframe["filename"].apply(lambda filename: tf.convert_to_tensor(pd.concat([pd.read_csv(filename),pd.read_csv(filename.replace('_channel_1', '_channel_2'))], ignore_index=True, axis=0).to_numpy()))

        # Express the label column in vectors of feat dim, repeating the same label for each of the feat_dim frame values
        dataframe["label"] = [tf.repeat(float(row["label"]), row["values"].shape[0]) for _, row in dataframe.iterrows()]
        
        # Get the number of features
        feat_dim = dataframe["values"][0].shape[1]
  
        # Split the data into train and test set
        dataframe_train = dataframe.iloc[split_index:,:]
        dataframe_test = dataframe.iloc[:split_index,:]

        # Get the indices for the train and test set
        train = dataframe_train.index.values
        test = dataframe_test.index.values

        # Reset the index in the train and test dataframes
        dataframe_train = dataframe_train.reset_index(drop=True)
        dataframe_test = dataframe_test.reset_index(drop=True)

        # Create title for the output excel
        title = trait+'_Regression_'+feat_type+'_'+str(layers)+'layers_'+str(neurons)+'neurons_'+str(int(split_index/3))+'firstSpkrs_test_set'

        # Create output excel containing a sheet with the Results per Frame and one sheet with the Results per Clip
        workbook = xlsxwriter.Workbook(output_path+title+'.xlsx')

        worksheet1 = workbook.add_worksheet('per_Frame')
        worksheet2 = workbook.add_worksheet('per_Clip')
        worksheet3 = workbook.add_worksheet('predictions')
        bold = workbook.add_format({'bold':True})

        worksheet1.write(0,0,'Epochs',bold)
        worksheet1.write(0,1,EPOCHS)

        worksheet2.write(0,0,'Epochs',bold)
        worksheet2.write(0,1,EPOCHS)

        worksheet1.write(1,0,feat_type,bold)
        worksheet2.write(1,0,feat_type,bold)


        for rep in range(1,repetitions+1):
            
            # Define per-repetitions score containers
            loss_per_rep = []
            mse_per_rep = []
            mae_per_rep = []

            # Define overall predictions and ground truths
            pred_score_per_clip_overall = []
            label_per_clip_overall = []

            # Train the model on Speakers split_index/3 - 120 and test on Speakers 1 - split_index/3
            worksheet1.write(rep+2,0,'repetition_'+str(rep), bold)
            worksheet2.write(rep+2,0,'repetition_'+str(rep), bold)

            train_ds = dataframe_to_dataset_REG(dataframe_train, BATCH_SIZE)
            valid_ds = dataframe_to_dataset_REG(dataframe_test, BATCH_SIZE)

            print(f'{trait}: Training for repetition {rep} ...')

            # Compile model
            keras.backend.clear_session()
            model = build_and_compile_model_REG(neurons, layers, l_rate, feat_dim, trait)
            model.summary()

            
            # Callbacks
            early_stopping_cb = keras.callbacks.EarlyStopping(
                monitor="loss", patience=100, restore_best_weights=True)

            tensorboard_cb = keras.callbacks.TensorBoard(
                os.path.join(os.curdir, "logs", model.name))

            callbacks = [early_stopping_cb, tensorboard_cb]

            # Fit data to model
            history = model.fit(
                train_ds,
                epochs=EPOCHS,
                batch_size = BATCH_SIZE,
                validation_data=valid_ds,
                callbacks=callbacks,
                verbose=2,)

            # Generate generalization metrics
            names_perframe, scores_perframe = get_metrics_REG(model, rep, valid_ds, loss_per_rep, mse_per_rep, mae_per_rep)

            # Calculate the clip level Accuracy of each repetition
            names_perclip, scores_perclip, predictions = per_clip_metrics_REG(test, model, rep, dataframe, trait, pred_score_per_clip_overall, label_per_clip_overall)


            # Write the results into the spreadsheet
            for i in range(0, len(names_perframe)):
                worksheet1.write(2, i+1, names_perframe[i], bold)
                worksheet1.write(rep+2, i+1, scores_perframe[i])

            worksheet1.write(1, 1, trait, bold)
            worksheet2.write(1, 1, trait, bold)

            for j in range(0, len(names_perclip)):
                worksheet2.write(2, j+1, names_perclip[j], bold)
                worksheet2.write(rep+2, j+1, scores_perclip[j])

            worksheet3.write(0, 1, 'Ground Truth', bold)
            worksheet3.write(0, 1+rep, 'repetition_'+str(rep), bold)
            for k in range(0, len(test)):
                worksheet3.write(k+1,0, predictions[0][k])
                worksheet3.write(k+1,1, predictions[1][k])
                worksheet3.write(k+1,1+rep, predictions[2][k])
  

        # Overall Spearman rank Correlation 
        rho_overall, p_overall = spearmanr(pred_score_per_clip_overall, label_per_clip_overall)
        print(f'{trait}: Spearman rho over all repetitions = {rho_overall}')
        print(f'{trait}: Spearman p value over all repetitions = {p_overall}')
       
        worksheet2.write(2, len(names_perclip)+1, 'Spearman_rho_overall', bold)
        worksheet2.write(3, len(names_perclip)+1, rho_overall)

        worksheet2.write(2, len(names_perclip)+2, 'Spearman_p_overall', bold)
        worksheet2.write(3, len(names_perclip)+2, p_overall) 


        workbook.close()


dnn_regression(100, 64, 1.9644e-5,['Openness'],
               2, 1, 20, 72,
               'wav2vec2', 'Wav2Vec_features_spatialised/last_hidden_state/', '', '', )
