from classifier_list import SVM
from classifier_list import KNN
from classifier_list import MLP



def classifier_model(features_raw,label_path,feature_type):
    if feature_type == "svm":
        print (SVM.svm(features_raw,label_path))
    elif feature_type == "knn":
        print (KNN.knn(features_raw,label_path))
    elif feature_type == "mlp":
        print (MLP.mlp(features_raw,label_path))
    else:
        print("For svm:\n"+SVM.svm(features_raw,label_path))
        print("For knn:\n"+KNN.knn(features_raw,label_path))
        print("For mlp:\n"+MLP.mlp(features_raw,label_path))