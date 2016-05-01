__author__ = 'Deyang'

import numpy as np


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


def get_boosted_decision_stumps(n_estimators):
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                              n_estimators=n_estimators)


def get_calibrated_classifier(clf, cv=3):
    return CalibratedClassifierCV(clf,  method='sigmoid', cv=cv)


X, y = make_gaussian_quantiles(n_samples=5000, n_features=6,
                               n_classes=3, random_state=1)

n_split = 4000

X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]


bds_classifier = get_boosted_decision_stumps(100)
bds_classifier.fit(X_train, y_train)

bds_prob = bds_classifier.predict_proba(X_test)
bds_predicted_labels = bds_classifier.predict(X_test)


bds_log_Loss = log_loss(y_test, bds_prob)
bds_accuracy = accuracy_score(y_test, bds_predicted_labels)


print "Raw classifier: log loss:%f, accuracy: %f" % (bds_log_Loss, bds_accuracy)

raw_clf = get_boosted_decision_stumps(100)
calibrated_classifier = get_calibrated_classifier(raw_clf)
calibrated_classifier.fit(X_train, y_train)


cali_prob = calibrated_classifier.predict_proba(X_test)
cali_predicted_labels = calibrated_classifier.predict(X_test)

cali_log_Loss = log_loss(y_test, cali_prob)
cali_accuracy = accuracy_score(y_test, cali_predicted_labels)
print "Calibrated classifier: log loss:%f, accuracy: %f" % (cali_log_Loss, cali_accuracy)



