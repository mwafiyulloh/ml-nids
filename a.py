def convergence_time_test(epochs, opt, batch_size, roc_list, time_list, acc_list ):
    print('==============================')
    print('optimizer =', opt)
    print('batch size =', batch_size)
    print('KFold = 10')
    kfold = KFold(n_splits=10, shuffle=True)
    total_time1 = 0
    total_time2 = 0
    total_acc = 0
    i = 1
    
    train_x, train_y, _, _ = load_train_test(df_train, df_test)
    for train_idx, valid_idx in kfold.split(train_x):
        print('\nCurrent KFold = ', i)
        Xtrain, Xvalid = train_x.iloc[train_idx], train_x.iloc[valid_idx]
        Ytrain, Yvalid = train_y.iloc[train_idx], train_y.iloc[valid_idx]
        
        scaler_kfold = StandardScaler()
        scaler_kfold.fit(Xtrain)
        Xtrain_scaled = scaler_kfold.transform(Xtrain)
        Xvalid_scaled = scaler_kfold.transform(Xvalid)
    
        time_before_training_AE_1 = datetime.now()
        AE_1.fit(Xtrain_scaled, Xtrain_scaled, epochs=epochs, batch_size=batch_size)
        time_after_training_AE_1 = datetime.now()
        delta_time_AE_1 = time_after_training_AE_1 - time_before_training_AE_1
        
        AE_1_encoded_train = Encoder_1.predict(Xtrain_scaled)
        AE_1_encoded_valid = Encoder_1.predict(Xvalid_scaled)

        time_before_training_AE_2 = datetime.now()
        AE_2.fit(AE_1_encoded_train, AE_1_encoded_train, epochs=epochs, batch_size=batch_size)
        time_after_training_AE_2 = datetime.now()
        delta_time_AE_2 = time_after_training_AE_2 - time_before_training_AE_2
        
        AE_2_encoded_train = Encoder_2.predict(AE_1_encoded_train)
        AE_2_encoded_valid = Encoder_2.predict(AE_1_encoded_valid)

        rfc.fit(AE_2_encoded_train, Ytrain)
            
        print(time_before_training_AE_1)
        print(time_after_training_AE_1)
        print('time taken for training 1st autoencoder', delta_time_AE_1.total_seconds())
        print()
        print(time_before_training_AE_2)
        print(time_after_training_AE_2)
        print('time taken for training 2nd autoencoder', delta_time_AE_2.total_seconds())
        print()
        print('total time =', delta_time_AE_1.total_seconds() + delta_time_AE_2.total_seconds())

        total_time1 += delta_time_AE_1.total_seconds()
        total_time2 += delta_time_AE_2.total_seconds()
        time_list.append([delta_time_AE_1.total_seconds(), delta_time_AE_2.total_seconds()])
        
        print('Data Train')
        trainResult = rfc.predict(AE_2_encoded_train)
        print(rfc.score(AE_2_encoded_train, Ytrain))
        print(classification_report(Ytrain, trainResult))
        
        print('Data Valid')
        validResult = rfc.predict(AE_2_encoded_valid)
        valid_acc = rfc.score(AE_2_encoded_valid, Yvalid)
        total_acc += valid_acc
        acc_list.append(valid_acc)
        print(valid_acc)
        print(classification_report(Yvalid, validResult))
        
        print()
              
        fpr, tpr, _ = roc_curve(Yvalid, validResult)
        roc_auc = auc(fpr, tpr)
        roc_list.append([fpr, tpr, roc_auc])
        print()
        i+=1
        
    avg_time_AE_1 = total_time1 / 10 
    avg_time_AE_2 = total_time2 / 10
    avg_total_val_acc = total_acc / 10
    print('Average training time for AE_1:', avg_time_AE_1, 'seconds')
    print('Average training time for AE_2:', avg_time_AE_2, 'seconds')
    print('Average accuracy of 10 folds :', avg_total_val_acc)