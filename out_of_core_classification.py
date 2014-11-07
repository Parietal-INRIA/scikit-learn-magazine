from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 16) # A vectorizer

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers'))
X_test = vectorizer.fit_transform(dataset.data[:1000])
y_test = dataset.target[:1000]

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()     # SVM classifier that learns online using SGD.

# Artificially make long train data
train_data = 10 * dataset.data[1000:]
train_target = 10 * list(dataset.target[1000:])

for i in range(0, len(train_data), 1000):    # Iterate on mini-batchs of 1000 samples
    y_train = train_target[i:i + 1000]
    X_train = vectorizer.fit_transform(train_data[i:i + 1000])
    # update estimator with examples in the current mini-batch
    classifier.partial_fit(X_train, y_train, classes=range(len(dataset.target_names)))
    print classifier.score(X_test, y_test)
