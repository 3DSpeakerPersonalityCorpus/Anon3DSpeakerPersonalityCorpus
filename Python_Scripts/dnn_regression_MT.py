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
def dataframe_to_dataset_REG_MT(dataframe, traits):
  
    X = np.array(dataframe["values"].tolist())
    X = np.concatenate(X, axis=0)
    Y_trait = np.array(dataframe['label_'+traits[0]].tolist())
    Y_trait = np.concatenate(Y_trait, axis=0)
    Y_D = np.array(dataframe['label_'+traits[1]].tolist())
    Y_D = np.concatenate(Y_D, axis=0)
        
    print(X.shape, Y_trait.shape, Y_D.shape)

    return X, Y_trait, Y_D

# Build the Model
def build_and_compile_model_REG_MT(neurons, layers, l_rate, feat_dim, traits):
    if layers==1:
        inputs = Input(shape=feat_dim, name='main_input')
    
        main_branch = Dense(neurons, activation="relu", name="dense_1")(inputs)
        main_branch = Dropout(0.1, name="dropout_1")(main_branch)

    elif layers==2:
        inputs = Input(shape=feat_dim, name='main_input')
    
        main_branch = Dense(neurons, activation="relu", name="dense_1")(inputs)
        main_branch = Dropout(0.1, name="dropout_1")(main_branch)

        main_branch = Dense(neurons, activation="relu", name="dense_2")(main_branch)
        main_branch = Dropout(0.2, name="dropout_2")(main_branch)

    elif layers==3:
        inputs = Input(shape=feat_dim, name='main_input')
    
        main_branch = Dense(neurons, activation="relu", name="dense_1")(inputs)
        main_branch = Dropout(0.1, name="dropout_1")(main_branch)

        main_branch = Dense(neurons, activation="relu", name="dense_2")(main_branch)
        main_branch = Dropout(0.2, name="dropout_2")(main_branch)
    
        main_branch = Dense(neurons, activation="relu", name="dense_3")(main_branch)
        main_branch = Dropout(0.2, name="dropout_3")(main_branch)

    elif layers==4:
        inputs = Input(shape=feat_dim, name='main_input')
    
        main_branch = Dense(neurons, activation="relu", name="dense_1")(inputs)
        main_branch = Dropout(0.1, name="dropout_1")(main_branch)

        main_branch = Dense(neurons, activation="relu", name="dense_2")(main_branch)
        main_branch = Dropout(0.2, name="dropout_2")(main_branch)
    
        main_branch = Dense(neurons, activation="relu", name="dense_3")(main_branch)
        main_branch = Dropout(0.2, name="dropout_3")(main_branch)

        main_branch = Dense(neurons, activation="relu", name="dense_4")(main_branch)
        main_branch = Dropout(0.2, name="dropout_4")(main_branch)

    elif layers==5:
        inputs = Input(shape=feat_dim, name='main_input')
    
        main_branch = Dense(neurons, activation="relu", name="dense_1")(inputs)
        main_branch = Dropout(0.1, name="dropout_1")(main_branch)

        main_branch = Dense(neurons, activation="relu", name="dense_2")(main_branch)
        main_branch = Dropout(0.2, name="dropout_2")(main_branch)
    
        main_branch = Dense(neurons, activation="relu", name="dense_3")(main_branch)
        main_branch = Dropout(0.2, name="dropout_3")(main_branch)

        main_branch = Dense(neurons, activation="relu", name="dense_4")(main_branch)
        main_branch = Dropout(0.2, name="dropout_4")(main_branch)
        
        main_branch = Dense(neurons, activation="relu", name="dense_5")(main_branch)
        main_branch = Dropout(0.2, name="dropout_5")(main_branch)
    
    trait_branch = Dense(1, activation='linear', name='trait_output')(main_branch)
    D_branch = Dense(1, activation='linear', name='D_output')(main_branch)
    
    model = Model(inputs = inputs, outputs = [trait_branch, D_branch], name="MT_"+traits[0]+"_"+traits[1]+"_perception_regression")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate),
        loss={'trait_output': keras.losses.MeanSquaredError(), 'D_output': keras.losses.MeanSquaredError()},
        metrics=[keras.metrics.MeanSquaredError(name="mean_squared_error"), keras.metrics.MeanAbsoluteError(name="mean_absolute_error")],)
    
    return model



# Generalization Metrics
def get_metrics_REG_MT(model, rep, valid_ds, trait_loss_per_rep, trait_mse_per_rep, trait_mae_per_rep,
                D_loss_per_rep, D_mse_per_rep, D_mae_per_rep):

    scores = model.evaluate(valid_ds[0], (valid_ds[1], valid_ds[2]), verbose=0)

    names = model.metrics_names

    print(f'Scores for repetition {rep}:')

    for i in range(0, len(scores)):
        print(f'{names[i]} = {scores[i]}')

    trait_loss_per_rep.append(scores[names.index('trait_output_loss')])
    trait_mse_per_rep.append(scores[names.index('trait_output_mean_squared_error')])
    trait_mae_per_rep.append(scores[names.index('trait_output_mean_absolute_error')])

    D_loss_per_rep.append(scores[names.index('D_output_loss')])
    D_mse_per_rep.append(scores[names.index('D_output_mean_squared_error')])
    D_mae_per_rep.append(scores[names.index('D_output_mean_absolute_error')])

    return names, scores



# Calculate Spearman rank correlation between predictions and ground truth for each fold
def per_clip_metrics_REG_MT(test, model, rep, dataframe, traits, trait_pred_score_per_clip_overall, D_pred_score_per_clip_overall, trait_label_per_clip_overall, D_label_per_clip_overall):
 
    trait_pred_score_per_clip_per_rep = []
    trait_pred_score_per_frame_per_rep = []

    D_pred_score_per_clip_per_rep = []
    D_pred_score_per_frame_per_rep = []

    trait_label_per_clip_per_rep = []
    D_label_per_clip_per_rep = []

    pred_per_rep = [[0 for col in range(len(test))] for row in range(len(traits))]
    ground_truth = [[0 for col in range(len(test))] for row in range(len(traits))]
    test_files = [[0 for col in range(len(test))] for row in range(len(traits))]


    for t in test:        # for each speaker t in the test set          
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
            trait_pred_score_per_frame_per_rep.append(res[0][j])
            D_pred_score_per_frame_per_rep.append(res[1][j])
 
        trait_pred_score_per_clip_per_rep.append(np.mean(trait_pred_score_per_frame_per_rep))
        D_pred_score_per_clip_per_rep.append(np.mean(D_pred_score_per_frame_per_rep))

        trait_label_per_clip_per_rep.append(dataframe["label_"+traits[0]][t][0])
        D_label_per_clip_per_rep.append(dataframe["label_"+traits[1]][t][0])

        test_files[0][t] = (dataframe["filename"][t].split("/")[-1])
        test_files[1][t] = (dataframe["filename"][t].split("/")[-1])

        pred_per_rep[0][t] = trait_pred_score_per_clip_per_rep[t]
        pred_per_rep[1][t] = D_pred_score_per_frame_per_rep[t]

        ground_truth[0][t] = trait_label_per_clip_per_rep[t]
        ground_truth[1][t] = D_label_per_clip_per_rep[t]
        
        # Overall predictions and labels
        trait_pred_score_per_clip_overall.append(np.mean(trait_pred_score_per_frame_per_rep))
        trait_label_per_clip_overall.append(dataframe["label_"+traits[0]][t][0])

        D_pred_score_per_clip_overall.append(np.mean(D_pred_score_per_frame_per_rep))
        D_label_per_clip_overall.append(dataframe["label_"+traits[1]][t][0])

    # Spearman rank Correlation 
    trait_rho, trait_p = spearmanr(trait_pred_score_per_clip_per_rep, trait_label_per_clip_per_rep)
    D_rho, D_p = spearmanr(D_pred_score_per_clip_per_rep, D_label_per_clip_per_rep)

    print(f'{traits[0]}: repetition {rep} Spearman rank correlation rho = {trait_rho}')
    print(f'{traits[0]}: repetition {rep} Spearman rank correlation p = {trait_p}')

    print(f'{traits[1]}: repetition {rep} Spearman rank correlation rho = {D_rho}')
    print(f'{traits[1]}: repetition {rep} Spearman rank correlation p = {D_p}')

    # Mean and stdev of error per clip
    trait_label_per_clip_per_rep = np.array(trait_label_per_clip_per_rep)
    trait_pred_score_per_clip_per_rep = np.array(trait_pred_score_per_clip_per_rep)

    trait_mse_per_clip_per_rep = np.mean(np.square(trait_label_per_clip_per_rep - trait_pred_score_per_clip_per_rep))
    trait_stdev_se_per_clip_per_rep = np.std(np.square(trait_label_per_clip_per_rep - trait_pred_score_per_clip_per_rep))
 
    trait_mae_per_clip_per_rep = np.mean(np.absolute(trait_label_per_clip_per_rep - trait_pred_score_per_clip_per_rep))
    trait_stdev_ae_per_clip_per_rep = np.std(np.absolute(trait_label_per_clip_per_rep - trait_pred_score_per_clip_per_rep))


    D_label_per_clip_per_rep = np.array(D_label_per_clip_per_rep)
    D_pred_score_per_clip_per_rep = np.array(D_pred_score_per_clip_per_rep)

    D_mse_per_clip_per_rep = np.mean(np.square(D_label_per_clip_per_rep - D_pred_score_per_clip_per_rep))
    D_stdev_se_per_clip_per_rep = np.std(np.square(D_label_per_clip_per_rep - D_pred_score_per_clip_per_rep))
 
    D_mae_per_clip_per_rep = np.mean(np.absolute(D_label_per_clip_per_rep - D_pred_score_per_clip_per_rep))
    D_stdev_ae_per_clip_per_rep = np.std(np.absolute(D_label_per_clip_per_rep - D_pred_score_per_clip_per_rep))


    scores = [trait_mse_per_clip_per_rep, trait_stdev_se_per_clip_per_rep, trait_mae_per_clip_per_rep, trait_stdev_ae_per_clip_per_rep, trait_rho, trait_p,
              D_mse_per_clip_per_rep, D_stdev_se_per_clip_per_rep, D_mae_per_clip_per_rep, D_stdev_ae_per_clip_per_rep, D_rho, D_p]

    metrics_names = ['trait_mean_squared_error', 'trait_stdev_squared_error', 'trait_mean_absolute_error', 'trait_stdev_absolute_error', 'trait_Spearman_rho', 'trait_Spearman_p',
                     'D_mean_squared_error', 'D_stdev_squared_error', 'D_mean_absolute_error', 'D_stdev_absolute_error', 'D_Spearman_rho', 'D_Spearman_p']

    predictions = [test_files, ground_truth, pred_per_rep]

    return metrics_names, scores, predictions
 



def dnn_regression_MT(EPOCHS, BATCH_SIZE, l_rate, TRAITS,
                      repetitions, layers, neurons, split_index,
                      feat_type, feat_path, labels_path, output_path, ):


    for trait in TRAITS:
        traits = [trait, 'Distance']

        # Get the labels
        list_of_labels = []
        for j in range(0,len(traits)):
            labs = open('labels_numerical_'+traits[j]+'.txt','r')
            lab = labs.read()
            list_of_labels.append(lab.split('\n'))


        l = 0
        # Load the data in a Dataframe
        dataframe = pd.DataFrame(columns=["filename","label_"+traits[0],"label_"+traits[1]])

        for filename in os.listdir(feat_path):
            f = os.path.join(feat_path, filename)
            if os.path.isfile(f) and filename.endswith('csv'):
                if filename.split('.')[0].split('_channel_')[1]=='1':
                    # Create a dictionary with the data for the new row
                    new_row = {'filename': f, 'label_'+traits[0]: list_of_labels[0][l], 'label_'+traits[1]: list_of_labels[1][l]}
                    # Inserting the new row
                    dataframe.loc[len(dataframe)] = new_row
                    # Reset the index
                    dataframe = dataframe.reset_index(drop=True)
            
                    l += 1
                    

        # Remove leading space in filename column
        dataframe["filename"] = dataframe.apply(lambda row: row["filename"].strip(), axis=1)

        # Stack the data from channel 1 and channel 2
        dataframe["values"] = dataframe["filename"].apply(lambda filename: tf.convert_to_tensor(pd.concat([pd.read_csv(filename),pd.read_csv(filename.replace('_channel_1', '_channel_2'))], ignore_index=True, axis=0).to_numpy()))

        # Express the label columns in vectors of feat_dim dim, repeating the same label for each of the feat_dim frame values
        dataframe["label_"+traits[0]] = [tf.repeat(float(row["label_"+traits[0]]), row["values"].shape[0]) for _, row in dataframe.iterrows()]
        dataframe["label_"+traits[1]] = [tf.repeat(float(row["label_"+traits[1]]), row["values"].shape[0]) for _, row in dataframe.iterrows()]

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
        title = trait+'_Distance_Regression_'+feat_type+'_'+str(layers)+'layers_'+str(neurons)+'neurons_'+str(int(split_index/3))+'firstSpkrs_test_set'

        # Create output excel containing a sheet with the Results per Frame and one sheet with the Results per Clip
        workbook = xlsxwriter.Workbook(output_path+title+'.xlsx')

        worksheet1 = workbook.add_worksheet('per_Frame')
        worksheet2 = workbook.add_worksheet('per_Clip')
        worksheet3 = workbook.add_worksheet('predictions')
        worksheet4 = workbook.add_worksheet('predictions_Distance') 
        bold = workbook.add_format({'bold':True})

        worksheet1.write(0,0,'Epochs',bold)
        worksheet1.write(0,1,EPOCHS)

        worksheet2.write(0,0,'Epochs',bold)
        worksheet2.write(0,1,EPOCHS)

        worksheet1.write(1,0,feat_type,bold)
        worksheet2.write(1,0,feat_type,bold)


        for rep in range(1,repetitions+1):

            # Define per-repetition score containers
            trait_loss_per_rep = []
            trait_mse_per_rep = []
            trait_mae_per_rep = []

            D_loss_per_rep = []
            D_mse_per_rep = []
            D_mae_per_rep = []

            # Define overall predictions and ground truths
            trait_pred_score_per_clip_overall = []
            trait_label_per_clip_overall = []

            D_pred_score_per_clip_overall = []
            D_label_per_clip_overall = []


            # Train the model on Speakers split_index/3 - 120 and test on Speakers 1 - split_index/3
            worksheet1.write(rep+2,0,'repetition_'+str(rep), bold)
            worksheet2.write(rep+2,0,'repetition_'+str(rep), bold)

            train_ds = dataframe_to_dataset_REG_MT(dataframe_train, traits)
            valid_ds = dataframe_to_dataset_REG_MT(dataframe_test, traits)

            print(f'{trait}: Training for repetition {rep} ...')

            # Compile model
            keras.backend.clear_session()
            model = build_and_compile_model_REG_MT(neurons, layers, l_rate, feat_dim, traits)
            model.summary()

            # Callbacks
            early_stopping_cb = keras.callbacks.EarlyStopping(
                monitor="loss", patience=100, restore_best_weights=True)

            tensorboard_cb = keras.callbacks.TensorBoard(
                os.path.join(os.curdir, "logs", model.name))

            callbacks = [early_stopping_cb, tensorboard_cb]


            # Fit data to model        
            history = model.fit({'main_input': train_ds[0]},
                                {'trait_output': train_ds[1], 'D_output': train_ds[2]},
                                validation_data = ({'main_input': valid_ds[0]},
                                                   {'trait_output': valid_ds[1], 'D_output': valid_ds[2]}),
                                epochs=EPOCHS,
                                batch_size = BATCH_SIZE,
                                callbacks=callbacks,
                                verbose=2,)

            # Generate generalization metrics
            names_perframe, scores_perframe = get_metrics_REG_MT(model, rep, valid_ds,
                                                                 trait_loss_per_rep, trait_mse_per_rep, trait_mae_per_rep,
                                                                 D_loss_per_rep, D_mse_per_rep, D_mae_per_rep)


            # Calculate the clip level Accuracy of each repetition
            names_perclip, scores_perclip, predictions = per_clip_metrics_REG_MT(test, model, rep, dataframe, traits,
                                                                    trait_pred_score_per_clip_overall, D_pred_score_per_clip_overall,
                                                                    trait_label_per_clip_overall, D_label_per_clip_overall)
 
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
            worksheet4.write(0, 1, 'Ground Truth', bold)
            worksheet4.write(0, 1+rep, 'repetition_'+str(rep), bold)

            for q in range(0,len(test)):
                worksheet3.write(q+1,0, predictions[0][0][q])
                worksheet3.write(q+1,1, predictions[1][0][q])
                worksheet3.write(q+1,1+rep, predictions[2][0][q])

                worksheet4.write(q+1,0, predictions[0][1][q])
                worksheet4.write(q+1,1, predictions[1][1][q])
                worksheet4.write(q+1,1+rep, predictions[2][1][q])


        # Overall Spearman rank Correlation 
        trait_rho_overall, trait_p_overall = spearmanr(trait_pred_score_per_clip_overall, trait_label_per_clip_overall)
        print(f'{traits[0]}: Spearman rho over all repetitions = {trait_rho_overall}')
        print(f'{traits[0]}: Spearman p value over all repetitions = {trait_p_overall}')
 
        D_rho_overall, D_p_overall = spearmanr(D_pred_score_per_clip_overall, D_label_per_clip_overall)
        print(f'{traits[1]}: Spearman rho over all repetitions = {D_rho_overall}')
        print(f'{traits[1]}: Spearman p value over all repetitions = {D_p_overall}')
       
        worksheet2.write(2, len(names_perclip)+1,'trait_Spearman_rho_overall', bold)
        worksheet2.write(3, len(names_perclip)+1, trait_rho_overall)

        worksheet2.write(2, len(names_perclip)+3,'D_Spearman_rho_overall', bold)
        worksheet2.write(3, len(names_perclip)+3, D_rho_overall)
          
        worksheet2.write(2, len(names_perclip)+2, 'trait_Spearman_p_overall', bold)
        worksheet2.write(3, len(names_perclip)+2, trait_p_overall)
        
        worksheet2.write(2, len(names_perclip)+4, 'D_Spearman_p_overall', bold)
        worksheet2.write(3, len(names_perclip)+4, D_p_overall)


        workbook.close()




dnn_regression_MT(100, 64, 1.9644e-5, ['Openness'],
                  2, 1, 20, 72,
                  'wav2vec2', 'Wav2Vec_features_spatialised/last_hidden_state/', '', '', )


        




















