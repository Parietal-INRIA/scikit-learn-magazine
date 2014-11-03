Let's avoid the table/box diagram given what you feel.

An extended version of the code example and the big data code example
would be fine. I guess I'm not sure about the sparse coded feature vector
either... maybe you need a vision-based example for it, and sparse coding
the TFIDF vectors makes no sense, so forget about it. Are there other
features that make sense for the text example?

> https://github.com/scikit-image/skimage-demos/pull/3

What you said below about the latest trends would actually be excellent
to include as a paragraph on managing an open source model (you almost
have a full paragraph). If you're OK with it, we can massage it into a
paragraph for your approval or if you want to, you can craft it. The
overall message being that you have to strike a balance between
supporting powerful techniques and utilizing your limited project cycles
on proven technologies.

________________________________________________________________________________

Not much will happen before Nov 15th, as I am travelling in Argentina
post conference. Supposedly I am off-line, on vacation. In practice, I am
checking e-mail way too often :). I am getting back on the 4th, but I
will have a lot of very urgent matters, and thus I will probably not be
able to handle your requests, unless I do them while traveling.

1. I am a bit sceptical about the box diagram, for several reasons, the
first one being that what you are suggesting is mixing two different kind
of information: what libraries we use, and what we provide. I am not
really sure how to avoid being confusing. I can see a two-column layout
where list functionality on one side, and how to solve it. I think that,
the way I see it, it would mainly be a table. Maybe slightly more
complex, but this is not obvious to me. Anyhow, such figure takes a while
to do, and I don't think that I'll do it before a couple of weeks.

2. I'll work on the code example a bit more. I am surprised that you
mention sparse coding: what exactly to you have in mind? A soft
thresholding function? Coding on a specific dictionary, in which case
which one?

3. I'll work on a "big-data" code example.

4. With regards to your last remark on how we handle the latest trends,
one problem is that we tend to stick away from them. Neural nets are
trendy, but very much a technology in flux that doesn't match the goals
of scikit-learn to provide mostly turn-key code. We have found that GPUs
give a lot of user-level problems. And new plateform, like Android, are
out of scope for scikit-learn if them don't come with a good CPython
implementation, as this is our basis. Though a persistence layer to save
models that would then be loaded by predictors on Android would be in the
scope.

Sorry for the answer, that somewhat pushes back on some of your
suggestions. I am trying to find a middle point between your desire to
move fast, and the fact that I don't see by myself an easy way to address
some of these suggestions.

________________________________________________________________________________

2. Can you extend your code example (Figure 4) slightly so that it shows
off how easy it is to mix and match various features? E.g. (a) use
alternate classifiers (i.e. show how easy it is to  substitute SVM or
random forests ) (b) use alternate feature extractors (e.g. maybe add a
sparse coder in frontnt of the TFIDF), and (c) try various evaluation
metrics (e.g. cross validation, precision/recall and confusion matrix).

3.  If you can add a code snippet to the big data section to show the
partial_fit estimator and HashingVectorizer in use, that would be great.

4. Maybe as a short extra paragraph to the "Nurturing Open Source"
section, can you reflect on how you guys, as an Open Source project, plan
to handle/usually handle trends such as Deep Networks, GPU acceleration,
distributed learning, and classification infrastructure that run on new
platforms such as (Android/iOS) phones?

