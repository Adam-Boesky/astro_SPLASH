from sklearn.ensemble import RandomForestClassifier


class SnClassifier:
    def __init__(self, classifier: RandomForestClassifier = None) -> None:

        # Import domain transfer and host prop predicting NNs
        self.domain_transfer_net = domain_transfer_net
        self.property_predicting_net = property_predicting_net
        self.random_forest = random_forest


    def predict(self, X):
        # Transform data with domain transfer network
        X_transformed = self.domain_transfer_net.transform(X)

        # Get features for random forest
        features_for_rf = self.property_predicting_net.predict(X_transformed)

        # Predict with random forest
        return self.random_forest.predict(features_for_rf)
