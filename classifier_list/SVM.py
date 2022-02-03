import pandas as pd
import numpy as np 



def svm(features,l_path):

    
    labels_df = pd.read_csv(l_path)
    features = np.concatenate((features,labels_df["labels"].to_numpy().reshape(-1,1)), axis=1)
    features_df = pd.DataFrame(features)
    
    #Train-Test Splitting
    
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(features_df, features_df[features_df.columns.size-1]):
        strat_train_raw = features_df.loc[train_index]
        strat_test_raw = features_df.loc[test_index]


    #Drop Label

    strat_train = strat_train_raw.drop([strat_train_raw.columns.size-1], axis=1)
    strat_train_label = strat_train_raw[strat_train_raw.columns.size-1]
    strat_test = strat_test_raw.drop([strat_test_raw.columns.size-1], axis=1)
    strat_test_label = strat_test_raw[strat_test_raw.columns.size-1]


    #Feature Scaling
    
    from sklearn.preprocessing import StandardScaler
    std_scaler = StandardScaler()
    std_scaler.fit(strat_train)
    scaled_features_train = std_scaler.transform(strat_train)
    scaled_features_test = std_scaler.transform(strat_test)



    #Prediction using SVM

    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix

    #model
    
    model = SVC(kernel='rbf')
    #train model
    
    model.fit(scaled_features_train, strat_train_label)

    #use model for prediction
    
    predict=model.predict(scaled_features_test)

    #calculate accuracy of this model
    
    acc=accuracy_score(predict,strat_test_label)
    txt = "The accuracy is {:.0f}% ."
    return "{} \n {}\n {}".format(txt.format(acc*100),confusion_matrix(predict,strat_test_label),classification_report(predict,strat_test_label))