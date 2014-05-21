import urllib, tarfile  # Standard library imports
from sklearn import linear_model, pipeline, datasets, feature_extraction

urllib.urlretrieve('http://bit.ly/1juuXIr', 'tmp.tgz')
tarfile.open('tmp.tgz').extractall(path='.')
data = datasets.load_files('txt_sentoken')

clf = pipeline.Pipeline([('step1', feature_extraction.text.TfidfVectorizer(min_df=2,
                                                 sublinear_tf=True, ngram_range=(1, 2))),
                         ('step2', linear_model.LogisticRegression(C=5000))])
clf.fit(data.data, data.target)

# Demo
clf.predict_proba(["This movie is the worst I ever saw."])[0, 0]  # negative probability
# 0.99929299972040175
clf.predict_proba(["Shawshank Redemption, eat your heart out!"])[0, 1]  # positive probability
# 0.83124224591605844

# Measure the prediction score by cross validation
from sklearn import cross_validation
print cross_validation.cross_val_score(clf, data.data, data.target, cv=5,
verbose=10)
