import pandas as pd
import chardet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import scipy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class FakeNewsDetector:
    def __init__(self, max_features = 500):
        self.max_features = max_features

        self.title_vectorizer = self.text_vectorizer = CountVectorizer(
            max_features = self.max_features,
            ngram_range =(1,2)
        )

        self.classifier = MultinomialNB()

        self.is_trained = False
        self.accuracy = None
        self.feature_names = None

    def load_data(self):
        df = pd.read_csv('dataset/news.csv', on_bad_lines = 'warn',
                         encoding = 'utf-8', encoding_errors = 'replace')
        return df

    def preprocess_data(self, df):
        df = df.drop(columns='Unnamed: 0', axis=1)
        df['label'] = df['label'].map({"REAL": 1, "FAKE": 0}).astype(float)

        title_features = self.title_vectorizer.fit_transform(df['title'])
        text_features = self.text_vectorizer.fit_transform(df['text'])

        X = scipy.sparse.hstack([title_features, text_features])
        y = df['label']
        return X, y

    def train(self, X, y):
        train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
        val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size = 0.5, random_state = 42, stratify = temp_y)

        self.train_X, self.train_y = train_X, train_y
        self.val_X, self.val_y = val_X, val_y
        self.test_X, self.test_y = test_X, test_y

        self.model = LogisticRegression(n_jobs = -1, max_iter = 1000,
                                        solver = 'liblinear',
                                        class_weight = 'balanced',
                                        random_state = 42)
        self.model.fit(train_X, train_y)

        val_pred = self.model.predict(self.val_X)
        test_pred = self.model.predict(self.test_X)

        val_acc = accuracy_score(self.val_y, val_pred)
        test_acc = accuracy_score(self.test_y, test_pred)

        self.is_trained = True
        self.val_acc = val_acc
        self.test_acc = test_acc

        return val_acc, test_acc

    def evaluate(self):
        val_pred = self.model.predict(self.val_X)
        test_pred = self.model.predict(self.test_X)

        print("=== CLASSIFICATION REPORT OF VAL ===\n")
        print(classification_report(self.val_y, val_pred, target_names = ['FAKE', 'REAL']))

        print('=== TEST REPORT ===\n')
        print(classification_report(self.test_y, test_pred, target_names = ['FAKE', "REAL"]))

        return classification_report(self.test_y, test_pred, output_dict = True)

    def get_names_out(self, n=20):
        if not self.is_trained:
            raise ValueError("Train the model first.")

        title_feat = self.title_vectorizer.get_feature_names_out()
        text_feat = self.text_vectorizer.get_feature_names_out()
        all_feat = list(title_feat) + list(text_feat)

        coef = self.model.coef_[0]

        fake_indices = coef.argsort()[:n]
        fake_words = [(all_feat[i], coef[i]) for i in fake_indices]

        real_indices = coef.argsort()[-n:][::-1]
        real_words = [(all_feat[i], coef[i]) for i in real_indices]

        print("\n=== TOP FAKE-INDICATING WORDS ===\n")
        for word, score in fake_words:
            print(f"{word}: {score:.3f}")

        print("\n\n=== TOP REAL-INDICATING WORDS ===\n")
        for word, score in real_words:
            print(f"{word}: {score:.3f}")

    def predict(self, title, text):
        if not self.is_trained:
            raise ValueError("Train the model first.")

        title_feat = self.title_vectorizer.transform([title])
        text_feat = self.text_vectorizer.transform([text])

        X_new = scipy.sparse.hstack([title_feat, text_feat])
        pred = self.model.predict(X_new)[0]
        prob = f"{max(self.model.predict_proba(X_new)[0]) * 100:.2f}%"
        
        return {"prediction": "REAL" if pred == 1 else "FAKE",
                "confidence": prob}

if __name__ == "__main__":
    detector = FakeNewsDetector()
    df = detector.load_data()
    X, y = detector.preprocess_data(df)

    val_acc, test_acc = detector.train(X, y)
    print(f"Training completed!\n VAL: {val_acc:.3f}, TEST: {test_acc:.3f}")

    eval_res = detector.evaluate()
    top_feat = detector.get_names_out()

    sample_title = "Breaking: Scientist discovered a cure for cancer."
    sample_text = "This incredible discover will change everything for everyone in the world ..."

    res = detector.predict(sample_title, sample_text)
    print(f"\nPrediction result: {res}")
