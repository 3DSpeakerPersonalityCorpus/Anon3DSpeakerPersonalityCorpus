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


# Prepare a Tensorflow dataset
def dataframe_to_dataset_MT(dataframe, traits):
  
    X = np.array(dataframe["values"].tolist())
    X = np.concatenate(X, axis=0)
    Y_trait = np.array(dataframe['label_'+traits[0]].tolist())
    Y_trait = np.concatenate(Y_trait, axis=0)
    Y_D = np.array(dataframe['label_'+traits[1]].tolist())
    Y_D = np.concatenate(Y_D, axis=0)
    
    
    print(X.shape, Y_trait.shape, Y_D.shape)

    return X, Y_trait, Y_D

# Build the Model
def build_and_compile_model_MT(neurons, layers, l_rate, feat_dim, traits, class_names):
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
    
    trait_branch = Dense(len(class_names), activation='softmax', name='trait_output')(main_branch)
    D_branch = Dense(len(class_names), activation='softmax', name='D_output')(main_branch)
    
    model = Model(inputs = inputs, outputs = [trait_branch, D_branch], name="MT_"+traits[0]+"_"+traits[1]+"_perception")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate),
        loss={'trait_output': keras.losses.CategoricalCrossentropy(), 'D_output': keras.losses.CategoricalCrossentropy()},
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],)
    
    return model



# Generalization Metrics
def get_metrics_MT(model, rep, valid_ds,
                trait_acc_per_rep, trait_loss_per_rep, trait_auc_per_rep, trait_prec_per_rep, trait_rec_per_rep, 
                D_acc_per_rep, D_loss_per_rep, D_auc_per_rep, D_prec_per_rep, D_rec_per_rep):
    
    scores = model.evaluate(valid_ds[0], (valid_ds[1], valid_ds[2]), verbose=0)
    
    names = model.metrics_names
    
    print(f'Scores for repetition {rep}:')
    
    for i in range(0, len(scores)):
        print(f'{names[i]} = {scores[i]}')
        
    trait_acc_per_rep.append(scores[names.index('trait_output_accuracy')])
    trait_loss_per_rep.append(scores[names.index('trait_output_loss')])
    trait_auc_per_rep.append(scores[names.index('trait_output_auc')])
    trait_prec_per_rep.append(scores[names.index('trait_output_precision')])
    trait_rec_per_rep.append(scores[names.index('trait_output_recall')])
        
   
    D_acc_per_rep.append(scores[names.index('D_output_accuracy')])
    D_loss_per_rep.append(scores[names.index('D_output_loss')])
    D_auc_per_rep.append(scores[names.index('D_output_auc')])
    D_prec_per_rep.append(scores[names.index('D_output_precision')])
    D_rec_per_rep.append(scores[names.index('D_output_recall')])
    
    return names, scores



# Calculate the clip level Accuracy of each fold
def per_clip_metrics_MT(test, model, dataframe, traits, class_names, rep):
    
    correct_by_majority = [0] * len(traits)
    
    tp = [0] * len(traits)
    fp = [0] * len(traits)
    tn = [0] * len(traits)
    fn = [0] * len(traits)
    
    accuracy = [0] * len(traits)
    precision = [0] * len(traits)
    recall = [0] * len(traits)
    F1 = [0] * len(traits)

    pred_per_rep = [[0 for col in range(len(test))] for row in range(len(traits))]
    ground_truth = [[0 for col in range(len(test))] for row in range(len(traits))]
    test_files = [[0 for col in range(len(test))] for row in range(len(traits))]
    
    for t in test:        # for each clip t in the test set
        correct = [0] * len(traits)
        incorrect = [0] * len(traits)

        res = model.predict(
        dataframe["values"][t],
        batch_size=None,
        verbose="1",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False)
        
        for k in range(0,len(traits)):                           # for each of the 2 traits (trait, D)
            for j in range(0,len(dataframe["values"][t])):       # for each frame j within a clip t 
               
                if res[k][j][0]>0.5:
                    to_compare = tf.constant([1.0, 0.0])
                else:
                    to_compare = tf.constant([0.0, 1.0])
        

                if str(to_compare) == str(dataframe["label_"+traits[k]][t][j]):     # Calculate all the 
                    correct[k] += 1                                                 # Correct ones
                else:                                                               # &
                    incorrect[k] +=1                                                # Incorrect ones
                        
            # if the majority of frames per clip are correctly predicted                           # arbitrarily take [1.0, 0.0] --> low to be negative
            if correct[k] > incorrect[k]:                                                          # arbitrarily take [0.0, 1.0] --> high to be positive
                correct_by_majority[k] +=1                                                         
                                                                                                  
                if str(dataframe["label_"+traits[k]][t][0]) == str(tf.constant([1.0, 0.0])):
                    tn[k] += 1                                                               # calculate true negatives    
                    pred_per_rep[k][t] = class_names[0]
                    ground_truth[k][t] = class_names[0]
                    test_files[k][t] = dataframe["filename"][t].split("/")[-1]
                elif str(dataframe["label_"+traits[k]][t][0]) == str(tf.constant([0.0, 1.0])):
                    tp[k] += 1                                                               # calculate true positives
                    pred_per_rep[k][t] = class_names[1]
                    ground_truth[k][t] = class_names[1]
                    test_files[k][t] = dataframe["filename"][t].split("/")[-1]

            # if the majority of frames per clip are incorrectly predicted       
            elif correct[k] <= incorrect[k]:      
                
                if str(dataframe["label_"+traits[k]][t][0]) == str(tf.constant([1.0, 0.0])):
                    fp[k] += 1                                                               # calcultate false positives
                    pred_per_rep[k][t] = class_names[1]
                    ground_truth[k][t] = class_names[0]
                    test_files[k][t] = dataframe["filename"][t].split("/")[-1]
                elif str(dataframe["label_"+traits[k]][t][0]) == str(tf.constant([0.0, 1.0])):
                    fn[k] += 1                                                               # calculate false negatives
                    pred_per_rep[k][t] = class_names[0]
                    ground_truth[k][t] = class_names[1]
                    test_files[k][t] = dataframe["filename"][t].split("/")[-1]
                
    for k in range(0, len(traits)):        
        print(f'{traits[k]}: repetition {rep} Correctly classified clips = {correct_by_majority[k]}')   
        print(f'{traits[k]}: repetition {rep} Total number of clips in the test set = {len(test)}')
        print(f'{traits[k]}: repetition {rep} per Clip Accuracy = {correct_by_majority[k]/len(test)}')
        print(f'{traits[k]}: repetition {rep} True Positives = {tp[k]}')
        print(f'{traits[k]}: repetition {rep} True Negatives = {tn[k]}')
        print(f'{traits[k]}: repetition {rep} False Positives = {fp[k]}')
        print(f'{traits[k]}: repetition {rep} False Negatives = {fn[k]}')
        
        
        accuracy[k] = correct_by_majority[k]/len(test)
        
        if tp[k]+fp[k] != 0:
            precision[k] = tp[k]/(tp[k]+fp[k])
        else:
            precision[k] = 0
        
        if tp[k]+fn[k] != 0:
            recall[k] = tp[k]/(tp[k]+fn[k])
        else:
            recall[k] = 0
            
        if precision[k]+recall[k] != 0:
            F1[k] = 2*precision[k]*recall[k]/(precision[k]+recall[k])
        else:
            F1[k] = 0
        
        print(f'{traits[k]}: repetition {rep} per file Precision = {precision[k]}')
        print(f'{traits[k]}: repetition {rep} per file Recall = {recall[k]}')
        print(f'{traits[k]}: repetition {rep} per file F1 = {F1[k]}')
        
    scores = [correct_by_majority, tp, tn, fp, fn, accuracy, recall, precision, F1]
        
    metrics_names = ['correct_clips', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'recall', 'precision', 'F1']

    predictions = [test_files, ground_truth, pred_per_rep]
    
    return metrics_names, scores, predictions



def dnn_classification_MT(EPOCHS, BATCH_SIZE, l_rate,
                          class_names, TRAITS,
                          repetitions, layers, neurons, split_index,
                          feat_type, feat_path, labels_path, output_path, ):


    for trait in TRAITS:
        traits = [trait, 'Distance']

        # Get the labels
        list_of_labels = []
        for j in range(0,len(traits)):
            labs = open(labels_path+'labels_'+traits[j]+'.txt','r')
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


        # Replace string labels with numerical values (their indices in the label list)
        for j in range(0,len(traits)):
            dataframe['label_'+traits[j]].replace('low', class_names.index('low'), inplace=True)
            dataframe['label_'+traits[j]].replace('high', class_names.index('high'), inplace=True)


        # Remove leading space in filename column
        dataframe["filename"] = dataframe.apply(lambda row: row["filename"].strip(), axis=1)
    
        # Stack the data from channel 1 and channel 2
        dataframe["values"] = dataframe["filename"].apply(lambda filename: tf.convert_to_tensor(pd.concat([pd.read_csv(filename),pd.read_csv(filename.replace('_channel_1', '_channel_2'))], ignore_index=True, axis=0).to_numpy()))

        # Express the label column in one hot vector form
        for j in range(0,len(traits)):
            dataframe['label_'+traits[j]] = [tf.one_hot(tf.repeat(row['label_'+traits[j]], row["values"].shape[0]), len(class_names)) for _, row in dataframe.iterrows()]

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
        title = trait+'_Distance_Classification_'+feat_type+'_'+str(layers)+'layers_'+str(neurons)+'neurons_'+str(int(split_index/3))+'firstSpkrs_test_set'

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
            trait_acc_per_rep = []
            trait_loss_per_rep = []
            trait_auc_per_rep = []
            trait_prec_per_rep = []
            trait_rec_per_rep =[]

            D_acc_per_rep = []
            D_loss_per_rep = []
            D_auc_per_rep = []
            D_prec_per_rep = []
            D_rec_per_rep =[]


            # Train the model on Speakers split_index/3 - 120 and test on Speakers 1 - split_index/3
            worksheet1.write(rep+2,0,'repetition_'+str(rep), bold)
            worksheet2.write(rep+2,0,'repetition_'+str(rep), bold)

            train_ds = dataframe_to_dataset_MT(dataframe_train, traits)
            valid_ds = dataframe_to_dataset_MT(dataframe_test, traits)

            print(f'{trait}: Training for repetition {rep} ...')

            # Compile model
            keras.backend.clear_session()
            model = build_and_compile_model_MT(neurons, layers, l_rate, feat_dim, traits, class_names)
            model.summary()
 
            # Callbacks
            early_stopping_cb = keras.callbacks.EarlyStopping(monitor="loss", patience=100, restore_best_weights=True)

            tensorboard_cb = keras.callbacks.TensorBoard(os.path.join(os.curdir, "logs", model.name))

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
            names_perframe, scores_perframe = get_metrics_MT(model, rep, valid_ds,
                                                             trait_acc_per_rep, trait_loss_per_rep, trait_auc_per_rep, trait_prec_per_rep, trait_rec_per_rep,
                                                             D_acc_per_rep, D_loss_per_rep, D_auc_per_rep, D_prec_per_rep, D_rec_per_rep)
 

            # Calculate the clip level Accuracy of each repetition
            names_perclip, scores_perclip, predictions = per_clip_metrics_MT(test, model, dataframe, traits, class_names, rep)


            # Write the results into the spreadsheet
            for i in range(0, len(names_perframe)):
                worksheet1.write(2, i+1, names_perframe[i], bold)
                worksheet1.write(rep+2, i+1, scores_perframe[i])
    
            for m, l in zip(range(0, len(traits)*len(scores_perclip), len(scores_perclip)), traits):
                worksheet2.write(1, m+1, l, bold)
        
            for j in range(0, len(names_perclip)):
                for k in range(0, len(traits)):
                    inc = k*len(names_perclip)
                    worksheet2.write(2, j+1+inc, names_perclip[j], bold)
                    worksheet2.write(rep+2, j+1+inc, scores_perclip[j][k])


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

                
    
    

        workbook.close()

        

dnn_classification_MT(100, 64, 1.9644e-5,
                          ["low","high"],['Openness'],
                          1, 1, 20, 72,
                          'wav2vec2', 'Wav2Vec_features_spatialised/last_hidden_state/', '', '', )

















    
