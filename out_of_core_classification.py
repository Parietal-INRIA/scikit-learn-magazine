from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer

# Create the hasher
hasher = HashingVectorizer(decode_error='ignore', n_features=2 ** 16)

dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))

X_test = hasher.fit_transform(dataset.data[:1000])
y_test = dataset.target[:1000]

from sklearn.linear_model import SGDClassifier
# SVM classifier that learns online using SGD.
classifier = SGDClassifier()

# Artificially make long train data
train_data = 15 * dataset.data[1000:]
train_target = 15 * list(dataset.target[1000:])

# Main loop : iterate on mini-batchs of 1000 samples
i = 0
while i < len(train_data) - 1000:
    y_train = train_target[i:i + 1000]
    X_train = hasher.fit_transform(train_data[i:i + 1000])
    # update estimator with examples in the current mini-batch
    classifier.partial_fit(X_train, y_train,
                           classes=range(len(dataset.target_names)))
    print classifier.score(X_test, y_test)
    i += 1000
