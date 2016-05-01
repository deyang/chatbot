__author__ = 'Deyang'

import os
from sklearn.externals import joblib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from ir_query_engine import engine_logger

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(DIR_PATH, "..", "saved_models", 'cali_boosted_decision_stumps.md')


def get_md_path():
    return MODEL_FILE_PATH


def get_boosted_decision_stumps(n_estimators):
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                              n_estimators=n_estimators)


def get_calibrated_classifier(clf, cv=3):
    return CalibratedClassifierCV(clf,  method='sigmoid', cv=cv)


class CalibratedBoostedDecisionStumpsFactory(object):

    @classmethod
    def get_new_model(cls, n_estimators, train_data, train_labels, cross_validation_folds, save=True):
        engine_logger.info("Training new calibrated boosted decision stumps")
        md_file_path = get_md_path()
        boosted_decision_stumps = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                     n_estimators=n_estimators)
        calibrated_bds = CalibratedClassifierCV(boosted_decision_stumps,  method='sigmoid', cv=cross_validation_folds)
        calibrated_bds.fit(train_data, train_labels)
        if save:
            joblib.dump(calibrated_bds, md_file_path)
        return calibrated_bds

    @classmethod
    def load_existing_model(cls):
        md_file_path = get_md_path()
        engine_logger.info("Loading existing calibrated boosted decision stumps")
        model = joblib.load(md_file_path)
        return model


if __name__ == "__main__":
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

    model = CalibratedBoostedDecisionStumpsFactory.get_new_model(100, X_train, y_train, 3)
    new_model = CalibratedBoostedDecisionStumpsFactory.load_existing_model()

    cali_prob = new_model.predict_proba(X_test)
    print cali_prob[0]
    cali_predicted_labels = new_model.predict(X_test)

    cali_log_Loss = log_loss(y_test, cali_prob)
    cali_accuracy = accuracy_score(y_test, cali_predicted_labels)
    print "Calibrated classifier: log loss:%f, accuracy: %f" % (cali_log_Loss, cali_accuracy)




