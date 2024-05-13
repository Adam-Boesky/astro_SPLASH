import numpy as np

from sklearn.ensemble import RandomForestClassifier
from typing import Union


class TwoStepClassifier():

    def __init__(self, random_state: int = 22, ia_thresh: float = 0.5, cc_class_weights: Union[dict, str] = 'balanced') -> None:
        """
        Initialize a two step classifier consisting of a RF that classifies into Ia vs. CC, and then
        classify the CC into Ib/c, II (P/L), IIn, SLSN. ia_thesh is the minimum probability of being a Ia to call
        something a Ia.
        """
        # Useful attributes
        self.ia_thresh = ia_thresh
        self.random_state = random_state
        self._trained = False

        # Classifiers
        self.base_classifier = RandomForestClassifier(n_estimators=1000, random_state=self.random_state)
        self.cc_classifier = RandomForestClassifier(n_estimators=1000, random_state=self.random_state, class_weight=cc_class_weights)


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit our models to given data.
        Parameters:
            X: The predictor values for each object.
            y: Array  of the classes of each SN. We will treat 0 as the SN class label.
        """
        # Train the Ia v. CC classifier
        y_Ia_v_rest = (y != 0).astype(int)  # np.array with 0=Ia, I=Other
        self.base_classifier.fit(X, y_Ia_v_rest)

        # Train the non-Ia model
        non_ia_mask = (y != 0)
        self.cc_classifier.fit(X[non_ia_mask], y[non_ia_mask])
        
        self._trained = True


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the classes of the Xs.
        Parameters:
            X: The predictors values of each object.
        Returns:
            1. Predictions of the classes for each object.
        """
        if self._trained is False:
            raise RuntimeError("You must train your TwoStepClassifier model before you predict with it!")

        # Predict Ia v. CC probability
        preds = np.zeros(X.shape[0])
        ia_preds_proba = self.base_classifier.predict_proba(X)[:, 0]
        cc_mask = (ia_preds_proba < self.ia_thresh)
        if np.sum(cc_mask) > 0:
            cc_preds_proba = self.cc_classifier.predict_proba(X[cc_mask])
            preds[cc_mask] = np.argmax(cc_preds_proba, axis=1) + 1

        return preds

        # # Predict Ia v. CC, then predict the CC subclasses
        # preds = self.base_classifier.predict(X)
        # CC_mask = preds != 0
        # preds[CC_mask] = self.cc_classifier.predict(X[CC_mask])

        # return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probabilities that each of the Xs are a given class.
        Parameters:
            X: The predictors values of each object.
        Returns:
            1. Predictions of the probabilities of the classes for each object.
        """
        if self._trained is False:
            raise RuntimeError("You must train your TwoStepClassifier model before you predict with it!")

        # Predict probas for Ia v. CC, then for the CC subclasses
        y_proba_ia = self.base_classifier.predict_proba(X)
        y_proba_cc = self.cc_classifier.predict_proba(X)
        y_proba = np.hstack([np.atleast_2d(y_proba_ia[:, 0]).reshape(-1, 1), y_proba_ia[:, 1].reshape(-1, 1) * y_proba_cc])

        return y_proba