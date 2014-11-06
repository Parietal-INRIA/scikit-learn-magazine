import urllib, tarfile  # From the Python standard library
from sklearn import linear_model, pipeline, datasets, feature_extraction

# Download, unpack and load the dataset.
urllib.urlretrieve('http://bit.ly/1juuXIr', 'tmp.tgz')
tarfile.open('tmp.tgz').extractall(path='.')
data = datasets.load_files('txt_sentoken')

clf = pipeline.Pipeline([('step1', feature_extraction.text.TfidfVectorizer(min_df=2,
                                                        sublinear_tf=True, ngram_range=(1, 2))),
                            ('step2', linear_model.LogisticRegression(C=5000))])
clf.fit(data.data, data.target)

# Demo
clf.predict_proba(["Worst movie I ever saw"])[0, 0]  # Negative probability
# 0.99929299972040175
clf.predict_proba(["Godzilla, eat your heart out!"])[0, 1]  # Positive probability
# 0.64533559029803611
print clf.predict(["This movie is the worst I ever saw.",
"Shawshank Redemption, eat your heart out!"])

# Measure the prediction score by cross validation
from sklearn import cross_validation
print cross_validation.cross_val_score(clf, data.data, data.target)
