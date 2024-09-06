def ml_classification(step, S, TRAITS, feat_file, labels_path, output_path,):

    # set the number of cross validations
    crossval = int(S/step)

    # define models
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    # create output excel containing a table of accuracy scores per ml model for every personality trait
    workbook = xlsxwriter.Workbook(out_name + '_' + str(crossval) + 'foldCV.xlsx')

    worksheet1 = workbook.add_worksheet('accuracy')
    worksheet2 = workbook.add_worksheet('precision')
    worksheet3 = workbook.add_worksheet('recall')
    worksheet4 = workbook.add_worksheet('harmonic_mean')
    worksheet5 = workbook.add_worksheet('successes')
    bold = workbook.add_format({'bold':True})

    # define mean and stdev tables in output sheet
    worksheet1.write(0,0,'MEAN')
    worksheet2.write(0,0,'MEAN')
    worksheet3.write(0,0,'MEAN')
    worksheet4.write(0,0,'MEAN')

    worksheet1.write(7,0,'STDEV')
    worksheet2.write(7,0,'STDEV')
    worksheet3.write(7,0,'STDEV')
    worksheet4.write(7,0,'STDEV')

    worksheet1.write(14,0,'T test Chance Vs Accuracy')

    worksheet5.write(0,0,'Total successful predictions')


    # print model name in output sheet
    for name in models:
        worksheet1.write(0,models.index(name)+1,name[0],bold)
        worksheet2.write(0,models.index(name)+1,name[0],bold)
        worksheet3.write(0,models.index(name)+1,name[0],bold)
        worksheet4.write(0,models.index(name)+1,name[0],bold)
        worksheet5.write(0,models.index(name)+1,name[0],bold)

        worksheet1.write(7,models.index(name)+1,name[0],bold)
        worksheet2.write(7,models.index(name)+1,name[0],bold)
        worksheet3.write(7,models.index(name)+1,name[0],bold)
        worksheet4.write(7,models.index(name)+1,name[0],bold)

        worksheet1.write(14,models.index(name)+1,name[0],bold)

    # add 'high' and 'low' labels in output sheet
    worksheet1.write(0,len(models)+1,'high',bold)
    worksheet1.write(0,len(models)+2,'low',bold)

    # initialise acc, prec, rec, f1 mean and stdev arrays for all traits
    acc_results_mean = [[0 for x in range(len(models))] for y in range(len(traits))]
    acc_results_stdev = [[0 for x in range(len(models))] for y in range(len(traits))]

    prec_results_mean = [[0 for x in range(len(models))] for y in range(len(traits))]
    prec_results_stdev = [[0 for x in range(len(models))] for y in range(len(traits))]

    rec_results_mean = [[0 for x in range(len(models))] for y in range(len(traits))]
    rec_results_stdev = [[0 for x in range(len(models))] for y in range(len(traits))]

    harmonic_results_mean = [[0 for x in range(len(models))] for y in range(len(traits))]
    harmonic_results_stdev = [[0 for x in range(len(models))] for y in range(len(traits))]
    
    # for every trait
    for trait in TRAITS:

        # print trait name in output sheet
        worksheet1.write(traits.index(trait)+1,0,trait,bold)
        worksheet2.write(traits.index(trait)+1,0,trait,bold)
        worksheet3.write(traits.index(trait)+1,0,trait,bold)
        worksheet4.write(traits.index(trait)+1,0,trait,bold)
        worksheet5.write(traits.index(trait)+1,0,trait,bold)

        worksheet1.write(traits.index(trait)+8,0,trait,bold)
        worksheet2.write(traits.index(trait)+8,0,trait,bold)
        worksheet3.write(traits.index(trait)+8,0,trait,bold)
        worksheet4.write(traits.index(trait)+8,0,trait,bold)

        worksheet1.write(traits.index(trait)+15,0,trait,bold)

        # print count of 'high' and 'low' labels in output sheet
        labels = open(lab_path + 'labels_'+trait+'.txt','r')
        lab = labels.read()
        list_of_labels = lab.split('\n')

        counts = pd.Series(list_of_labels).value_counts()
    
        worksheet1.write(traits.index(trait)+1,len(models)+1,counts['high'])
        worksheet1.write(traits.index(trait)+1,len(models)+2,counts['low'])

        # read in the data set (features + corresponding labels) and create dataset vector
        dataset = pd.read_csv(feat_file)
        dataset = dataset.assign(Labels = list_of_labels)

        speaker = dataset.groupby('Speaker').Speaker.count()

        array_data = dataset.values
        array_data = array_data[np.argsort(array_data[:,0])]
        n = dataset.shape[1]

        # initialise result array per model
        acc_res = [[0 for x in range(len(models))] for y in range(int(len(speaker)))]
        acc_res = np.array(acc_res).astype(float)

        prec_res = [[0 for x in range(len(models))] for y in range(int(len(speaker)/step))]
        prec_res = np.array(prec_res).astype(float)

        rec_res = [[0 for x in range(len(models))] for y in range(int(len(speaker)/step))]
        rec_res = np.array(rec_res).astype(float)

        harmonic_res = [[0 for x in range(len(models))] for y in range(int(len(speaker)/step))]
        harmonic_res = np.array(harmonic_res).astype(float)

        chance = [[0 for y in range(int(len(speaker)/step))] for x in range(len(models))]
        chance = np.array(chance).astype(float)

        num_successes = [[0 for y in range(int(len(speaker)/step))] for x in range(len(models))]
        num_successes = np.array(num_successes).astype(float)

        
        # for every n=step speakers:
        g = 0
        for i in range(0,len(speaker),step):
            # for each data point per speaker:
            ind_first = np.where(array_data == speaker.index[i])[0][0]
            ind_last = np.where(array_data == speaker.index[i])[0][speaker[i]-1]

            ##### Test set #####
            X_test = array_data[ind_first:ind_last+((step-1)*speaker[i]+1),3:n-1]
            Y_test = array_data[ind_first:ind_last+((step-1)*speaker[i]+1),n-1]
        
            ##### Train set #####
            index = []
            for j in range(ind_first,ind_last+((step-1)*speaker[i]+1)):
                index = np.append(index,j)
                index = index.astype(int)
            train_data = np.delete(array_data,index,axis=0)
            X_train = train_data[:,3:n-1]
            Y_train = train_data[:,n-1]

            ##### Fit models #####
            k = 0
            for name, model in models:
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)

                test_labels = pd.Series(Y_test).value_counts()
    
                # calculate chance probability based on priors in the test set
                if 'low' in test_labels.keys() and 'high' in test_labels.keys():
                    chance[k][g] = pow(test_labels['high']/len(Y_test),2)+pow(test_labels['low']/len(Y_test),2)
                else:
                    chance[k][g] = 1

                # calculate results: accuracy, precision, recall, f1 score           
                acc_score = accuracy_score(Y_test,Y_pred)
                prec_score = precision_score(Y_test,Y_pred,average='macro')
                rec_score = recall_score(Y_test,Y_pred,average='macro')
                harmonic_score = f1_score(Y_test,Y_pred,average='macro')
                
                acc_res[g][k] = round(acc_score,2)
                prec_res[g][k] = round(prec_score,2)
                rec_res[g][k] = round(rec_score,2)
                harmonic_res[g][k] = round(harmonic_score,2)

                # calculate number of successfully predicted instances
                for z in range(0,len(Y_pred)):
                    if Y_pred[z] == Y_test[z]:
                        num_successes[k][g] += 1

                k += 1
            g += 1

        # calculate the mean and standard deviation for accuracy, precision, recall and f1 score over all cross validations    
        acc_results_mean[traits.index(trait)] = np.mean(acc_res, axis = 0)
        acc_results_stdev[traits.index(trait)] = np.std(acc_res, axis = 0)

        prec_results_mean[traits.index(trait)] = np.mean(prec_res, axis = 0)
        prec_results_stdev[traits.index(trait)] = np.std(prec_res, axis = 0)

        rec_results_mean[traits.index(trait)] = np.mean(rec_res, axis = 0)
        rec_results_stdev[traits.index(trait)] = np.std(rec_res, axis = 0)

        harmonic_results_mean[traits.index(trait)] = np.mean(harmonic_res, axis = 0)
        harmonic_results_stdev[traits.index(trait)] = np.std(harmonic_res, axis = 0)

    
        acc_res = np.transpose(acc_res)

        # print the results in the output sheet
        for t in range(1,len(models)+1):
            worksheet1.write(traits.index(trait)+1,t,round(acc_results_mean[traits.index(trait)][t-1],2))
            worksheet2.write(traits.index(trait)+1,t,round(prec_results_mean[traits.index(trait)][t-1],2))
            worksheet3.write(traits.index(trait)+1,t,round(rec_results_mean[traits.index(trait)][t-1],2))
            worksheet4.write(traits.index(trait)+1,t,round(harmonic_results_mean[traits.index(trait)][t-1],2))

            worksheet1.write(traits.index(trait)+8,t,round(acc_results_stdev[traits.index(trait)][t-1],2))
            worksheet2.write(traits.index(trait)+8,t,round(prec_results_stdev[traits.index(trait)][t-1],2))
            worksheet3.write(traits.index(trait)+8,t,round(rec_results_stdev[traits.index(trait)][t-1],2))
            worksheet4.write(traits.index(trait)+8,t,round(harmonic_results_stdev[traits.index(trait)][t-1],2))

            tstat_chance_acc, tpval_chance_acc = stats.ttest_ind(chance[t-1], acc_res[t-1])

            worksheet1.write(traits.index(trait)+15,t,round(tpval_chance_acc,2))

            worksheet5.write(traits.index(trait)+1,t,np.sum(num_successes,axis=1)[t-1])

    workbook.close()
