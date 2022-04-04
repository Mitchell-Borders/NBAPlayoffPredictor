import numpy as np
import sklearn as sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utils.MLUtils import *


def main():
    X, y = get_input_output_split('./Team_Stats_Per_Game.csv')

    model_list = [
        DecisionTreeClassifier(),
        SVC(probability=True),
        GaussianNB(),
        RandomForestClassifier(),
        LogisticRegression(max_iter=10000),
        GradientBoostingClassifier(n_estimators=150, max_depth=3)
    ]

    current_model = [
        'DecisionTree',
        'Support Vector Machine',
        'GaussianNB',
        'RandomForestClassifier',
        'LogisticRegression',
        'GradientBoostingClassifier'
    ]

    for i in range(0, len(model_list)):
        print('\n' + current_model[i])
        model = model_list[i]

        actual_y = []
        predicted_y = []
        # implement 5-fold cross-validation
        fold_no = 1
        skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X, y):
            train_x = X[train_index]
            train_y = y[train_index]
            test_x = X[test_index]
            test_y = y[test_index]

            model.fit(train_x, train_y)

            pred = model.predict(test_x)

            # put values of this fold into the array
            actual_y.append(test_y)

            # use for calculations
            predicted_y.append(pred)
            # pred = np.argmax(pred, axis=1)

            fold_accuracy_score = sklearn.metrics.accuracy_score(test_y, pred)
            print('\nFold', fold_no, 'score (accuracy):', fold_accuracy_score)

            fold_no += 1

        # calculate data for overall model
        actual_y = np.concatenate(actual_y)
        predicted_y = np.concatenate(predicted_y)

        overall_accuracy_score = sklearn.metrics.accuracy_score(actual_y, predicted_y)
        print('\nOverall score (accuracy):', overall_accuracy_score)

        print(sklearn.metrics.classification_report(actual_y, predicted_y, zero_division=0, digits=5))

        # plot confusion matrix
        matrix = sklearn.metrics.confusion_matrix(actual_y, predicted_y)
        cmPlot = sklearn.metrics.ConfusionMatrixDisplay(matrix)
        cmPlot.plot(cmap='YlGn')
        plt.title(current_model[i])
        plt.show()


if __name__ == "__main__":
    main()
