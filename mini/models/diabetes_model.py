from utils.data_utils import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DiabetesModel:
    def __init__(self):
        self.dataset = load_dataset('data/diabetes.csv')
        self.X = self.dataset.drop('Outcome', axis=1)
        self.y = self.dataset['Outcome']
        self.train_model()

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model accuracy: {accuracy * 100:.2f}%')

    def predict(self, user_input):
        prediction = self.model.predict([user_input])
        if prediction[0] == 0:
            return 'You are not at risk of developing Type 2 Diabetes.'
        else:
            return 'Based on the provided information, you are at risk of developing Type 2 Diabetes.'