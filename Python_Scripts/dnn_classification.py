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

# Prepare a Tensorflow dataset
def dataframe_to_dataset(dataframe, BATCH_SIZE):
  
    X = np.array(dataframe["values"].tolist())
    X = np.concatenate(X, axis=0)
    Y = np.array(dataframe["label"].tolist())
    Y = np.concatenate(Y, axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    print(X.shape, Y.shape)

    return dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Build the Model
def build_and_compile_model(neurons, layers, l_rate, feat_dim, trait, class_names):
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

    outputs = Dense(len(class_names), activation="softmax", name="ouput")(x)

    model = Model(inputs=inputs, outputs=outputs, name=trait+"_perception")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    
    return model


# Generalization Metrics
def get_metrics(model, rep, valid_ds, acc_per_rep, loss_per_rep, 
                auc_per_rep, prec_per_rep, rec_per_rep):
     
    scores = model.evaluate(valid_ds, verbose=0)

    names = model.metrics_names

    print(f'Scores for repetition {rep}:')

    for i in range(0, len(scores)):
        print(f'{names[i]} = {scores[i]}')

    acc_per_rep.append(scores[names.index('accuracy')])
    loss_per_rep.append(scores[names.index('loss')])
    auc_per_rep.append(scores[names.index('auc')])
    prec_per_rep.append(scores[names.index('precision')])
    rec_per_rep.append(scores[names.index('recall')])
   
    return names, scores


# Calculate the clip level Accuracy of each fold
def per_clip_metrics(test, model, dataframe, trait, class_names, pred_per_rep, ground_truth, test_files, rep):

    correct_by_majority = 0

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    accuracy = 0
    precision = 0
    recall = 0
    F1 = 0

    for t in test:        # for each clip t in the test set
        correct = 0
        incorrect = 0

        res = model.predict(
        dataframe["values"][t],
        batch_size=None,
        verbose="1",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False)


        for j in range(0,len(dataframe["values"][t])):       # for each frame j within a clip t (each clip has 1098 frames)
            if res[j][0]>0.5:
                to_compare = tf.constant([1.0, 0.0])
            else:
                to_compare = tf.constant([0.0, 1.0])

            if str(to_compare) == str(dataframe["label"][t][j]):     # Calculate all the 
                correct += 1                                         # Correct ones
            else:                                                    # &
                incorrect +=1                                        # Incorrect ones

        # if the majority of frames per clip are correctly predicted                     # arbitrarily take [1.0, 0.0] --> low to be negative
        if correct > incorrect:                                                          # arbitrarily take [0.0, 1.0] --> high to be positive
            correct_by_majority +=1     

            if str(dataframe["label"][t][0]) == str(tf.constant([1.0, 0.0])):
                tn += 1                                                               # calculate true negatives
                pred_per_rep.append(class_names[0])
                ground_truth.append(class_names[0])
                test_files.append(dataframe["filename"][t].split("/")[-1])
            elif str(dataframe["label"][t][0]) == str(tf.constant([0.0, 1.0])):
                tp += 1                                                               # calculate true positives
                pred_per_rep.append(class_names[1])
                ground_truth.append(class_names[1])
                test_files.append(dataframe["filename"][t].split("/")[-1])
                    

        # if the majority of frames per clip are incorrectly predicted       
        elif correct <= incorrect:      
                
            if str(dataframe["label"][t][0]) == str(tf.constant([1.0, 0.0])):
                fp += 1                                                               # calcultate false positives
                pred_per_rep.append(class_names[1])
                ground_truth.append(class_names[0])
                test_files.append(dataframe["filename"][t].split("/")[-1])
            elif str(dataframe["label"][t][0]) == str(tf.constant([0.0, 1.0])):
                fn += 1                                                               # calculate false negatives
                pred_per_rep.append(class_names[0])
                ground_truth.append(class_names[1])
                test_files.append(dataframe["filename"][t].split("/")[-1])

    print(f'{trait}: repetition {rep} Correctly classified clips = {correct_by_majority}')   
    print(f'{trait}: repetition {rep} Total number of clips in the test set = {len(test)}')
    print(f'{trait}: repetition {rep} per Clip Accuracy = {correct_by_majority/len(test)}')
    print(f'{trait}: repetition {rep} True Positives = {tp}')
    print(f'{trait}: repetition {rep} True Negatives = {tn}')
    print(f'{trait}: repetition {rep} False Positives = {fp}')
    print(f'{trait}: repetition {rep} False Negatives = {fn}')

    accuracy = correct_by_majority/len(test)

    if tp+fp != 0:
        precision = tp/(tp+fp)
    else:
        precision = 0
        
    if tp+fn != 0:
        recall = tp/(tp+fn)
    else:
        recall = 0
            
    if precision+recall != 0:
        F1 = 2*precision*recall/(precision+recall)
    else:
        F1 = 0    

    print(f'{trait}: repetition {rep} per file Precision = {precision}')
    print(f'{trait}: repetition {rep} per file Recall = {recall}')
    print(f'{trait}: repetition {rep} per file F1 = {F1}')
        
    scores = [correct_by_majority, tp, tn, fp, fn, accuracy, recall, precision, F1]
        
    metrics_names = ['correct_clips', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'recall', 'precision', 'F1']

    predictions = [test_files, ground_truth, pred_per_rep]
    
    return metrics_names, scores, predictions



def dnn_classification(EPOCHS, BATCH_SIZE, l_rate,
                       class_names, TRAITS,
                       repetitions, layers, neurons, split_index,
                       feat_type, feat_path, labels_path, output_path, ):


    for trait in TRAITS:
        # Get the labels
        labs = open(labels_path+'labels_'+trait+'.txt','r')
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


        # Replace string labels with numerical values (their indices in the label list)
        dataframe["label"].replace('low', class_names.index('low'), inplace=True)
        dataframe["label"].replace('high', class_names.index('high'), inplace=True)


        # Remove leading space in filename column
        dataframe["filename"] = dataframe.apply(lambda row: row["filename"].strip(), axis=1)

        # Stack the data from channel 1 and channel 2
        dataframe["values"] = dataframe["filename"].apply(lambda filename: tf.convert_to_tensor(pd.concat([pd.read_csv(filename),pd.read_csv(filename.replace('_channel_1', '_channel_2'))], ignore_index=True, axis=0).to_numpy()))
 
        # Express the label column in one hot vector form
        dataframe["label"] = [tf.one_hot(tf.repeat(row["label"], row["values"].shape[0]), len(class_names)) for _, row in dataframe.iterrows()]

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
        title = trait+'_Classification_'+feat_type+'_'+str(layers)+'layers_'+str(neurons)+'neurons_'+str(int(split_index/3))+'firstSpkrs_test_set'

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
            
            # Define per-repetition score containers
            acc_per_rep = []
            loss_per_rep = []
            auc_per_rep = []
            prec_per_rep = []
            rec_per_rep = []

            # Define per-repetition prediction array
            pred_per_rep = []

            # Define ground truth array
            ground_truth = []

            # Define test files array
            test_files = []

            # Train the model on Speakers split_index/3 - 120 and test on Speakers 1 - split_index/3
            worksheet1.write(rep+2,0,'repetition_'+str(rep), bold)
            worksheet2.write(rep+2,0,'repetition_'+str(rep), bold)

            train_ds = dataframe_to_dataset(dataframe_train, BATCH_SIZE)
            valid_ds = dataframe_to_dataset(dataframe_test, BATCH_SIZE)

            print(f'{trait}: Training for repetition {rep} ...')

            # Compile model
            keras.backend.clear_session()
            model = build_and_compile_model(neurons, layers, l_rate, feat_dim, trait, class_names)
            model.summary()

            # Callbacks
            early_stopping_cb = keras.callbacks.EarlyStopping(monitor="loss", patience=EPOCHS, restore_best_weights=True)

            tensorboard_cb = keras.callbacks.TensorBoard(os.path.join(os.curdir, "logs", model.name))

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
            names_perframe, scores_perframe = get_metrics(model, rep, valid_ds, acc_per_rep, loss_per_rep, auc_per_rep, prec_per_rep, rec_per_rep)


            # Calculate the clip level Accuracy of each repetition
            names_perclip, scores_perclip, predictions = per_clip_metrics(test, model, dataframe, trait, class_names, pred_per_rep, ground_truth, test_files, rep)



            # Write the results into the spreadsheet
            for i in range(0, len(names_perframe)):
                worksheet1.write(2, i+1, names_perframe[i], bold)
                worksheet1.write(rep+2, i+1, scores_perframe[i])
        
            worksheet1.write(1, 1, trait, bold)
            worksheet2.write(1, 1, trait, bold)

            inc = 1
            for j in range(0, len(names_perclip)):
                worksheet2.write(2, j+1, names_perclip[j], bold)
                worksheet2.write(rep+2, j+1, scores_perclip[j])

            worksheet3.write(0, 1, 'Ground Truth', bold)
            worksheet3.write(0, 1+rep, 'repetition_'+str(rep), bold)
            for k in range(0, len(test)):
                worksheet3.write(k+1,0, predictions[0][k])
                worksheet3.write(k+1,1, predictions[1][k])
                worksheet3.write(k+1,1+rep, predictions[2][k])
            
        workbook.close()







dnn_classification(100, 64, 1.9644e-5,
                   ["low","high"],['Openness'],
                   1, 1, 20, 72,
                   'wav2vec2', 'Wav2Vec_features_spatialised/last_hidden_state/', '', '', )



                


    
