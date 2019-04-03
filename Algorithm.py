from sklearn.model_selection import train_test_split # version 0.18.1
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import Model

class Algos:
    data_train=0
    data_test=0
    target_train=0
    target_test=0


    def setDataTarget(self, data, target):
        d_train, d_test, t_train, t_test = train_test_split(data, target, train_size=0.95, test_size=0.05)
        self.data_train=d_train
        self.data_test=d_test
        self.target_train=t_train
        self.target_test=t_test

    def Bayes(self, modelName):
        #train
        clf = GaussianNB()
        clf.fit(self.data_train, self.target_train)
        Model.save_Model(modelName,clf)
        #predict
        predict = clf.predict(self.data_test)


        #print("BAYES : ",accuracy_score(self.target_test, predict))
        return predict
        return (accuracy_score(self.target_test,predict))

    def Ada_boost(self, modelName):
        clf = AdaBoostClassifier(n_estimators = 20)
        clf.fit(self.data_train, self.target_train)
        Model.save_Model(modelName,clf)
        predict = clf.predict(self.data_test)

        #print("ADA : ",accuracy_score(self.target_test, predict))
        return predict
        #return accuracy_score(self.target_test, predict)

    def svmImgvec(self, modelName):
        clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

        clf.fit(self.data_train, self.target_train)
        Model.save_Model(modelName,clf)
        predict = clf.predict(self.data_test)


        #print("SVM : ",accuracy_score(self.target_test, predict))
        return predict
        return accuracy_score(self.target_test, predict)
