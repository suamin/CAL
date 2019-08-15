"""
=============================================================
Demo of Enron Spam-Ham Classification in CAL Simulation
=============================================================
"""

import os
join = os.path.join

DATA_DIR = "data"
OUTPUT_DIR = "output"
FEATURES_DIR = "features"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

PREPROCEES = False
FEATURES = False

#============================================================
#                 I - Raw Files Preprocessing
#============================================================
# This is generic step and not specific to CAL application,
# its included for the sake of complete, standalone and 
# reproducible results from data files.
#
#   * Input: data dir containing "spam/", "ham/" folders
#   * Preprocessing: check scripts/preprocess.py for details
#   * Output: 3 files under output dir
#
#       - processed_enron_docs_as_lines.txt: each processed
#         file is converted to single line and considered as
#         one document (used for tfidf and doc2vec features)
#
#       - docid2path.pkl: mapping of each processed document
#         id to their original file path on system (note: ids
#         will not having any gaps, sequentially from 1 to N)
#
#       - processed_enron_sents_as_lines.txt: here documents
#         are processed at sentences level and only useful 
#         for training a word2vec like model to generate word
#         embeddings
#------------------------------------------------------------

if PREPROCEES:
    from scripts.preprocess import main as preprocess_main
    
    # TODO: add progress reporting
    preprocess_main(DATA_DIR, OUTPUT_DIR)

#============================================================
#                 II - Features Generation
#============================================================
# Like preprocessing this step of features creation is not
# trivial to CAL but included for the sake of reproducibility
#
#   * Input: processed files from I
#
#   * Features: TF-IDF, Vanilla word2vec, doc2vec
#     only required features can be set as needed
#
#   * Output: several files under FEATURES_DIR depending on
#     features selected
#
#       - (tfidf_vocab, tfidf_model, tfidf_X.npz):
#         TF-IDF vocab, model and features files
#
#       - (w2v_model, w2v_X.npy):
#         word2vec model and features files
#
#       - (d2v_model, d2v_X.npy):
#         doc2vec model and features files
#------------------------------------------------------------

if FEATURES:
    from scripts.prepare_features import main as features_main
    
    features_main(
        join(OUTPUT_DIR, 'processed_enron_docs_as_lines.txt'),
        join(OUTPUT_DIR, 'processed_enron_sents_as_lines.txt'),
        FEATURES_DIR,
        True, True, False
    )

#============================================================
#                III - Supervised Learning
#============================================================
# Without going into too many details of supervised learning
# or performing exhasutive experiments, we will show the 
# performance of default model (checks models.py) and 
# learning curves.
#
# note: we will use tf-idf features created in previous step
#------------------------------------------------------------

import models
import metrics
import plots
import numpy as np

from sklearn.model_selection import train_test_split
from gensim.utils import unpickle
from scipy import sparse


def load_X(X_path, is_sparse):
    if is_sparse:
        return sparse.load_npz(X_path)
    else:
        return np.load(X_path)


def load_y(docid2path_path):
    docid2path = unpickle(docid2path_path)
    y = list()
    for path in docid2path.values():
        if 'ham' in path.split(os.sep):
            y.append(1)
        else:
            y.append(0)
    return np.asarray(y, dtype=int)


data_X = load_X(join(FEATURES_DIR, 'tfidf_X.npz'), True)
data_y = load_y(join(OUTPUT_DIR, 'docid2path.pkl')) #np.load('features\\ynew.npy')

# create train-test splits
X, X_test, y, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=2018)

# plot learning curves with metrics of interest
lc_scoring = ['accuracy', 'precision', 'recall', 'roc_auc']

for scoring in lc_scoring:
    # load default model: Linear SVM with SGD
    clf_supervised = models.default_model()
    plt_handle = plots.plot_learning_curve(
        clf_supervised, 'Supervised Learning Curve (Scorer: {})'.format(scoring),
        X, y, cv=5, scoring=scoring
    )
    plt_handle.show()

# train and report test results
clf_supervised = models.default_model()
clf_supervised.fit(X, y)
sup_y_test_preds = clf_supervised.predict(X_test)
supervised_results = {
    'accuracy': metrics.accuracy(y_test, sup_y_test_preds),
    'precision': metrics.precision(y_test, sup_y_test_preds),
    'recall': metrics.recall(y_test, sup_y_test_preds),
    'gmeans': metrics.g_means(y_test, sup_y_test_preds),
    'auc': metrics.auc(y_test, sup_y_test_preds),
    'cohen-kappa': metrics.user_machine_agreement(y_test, sup_y_test_preds)
}

#============================================================
#                IV (a) - Active Learning
#============================================================
# Part IV of this demo is divided into two sub-parts:
#
#   (a) - Here we demonstrate the active "learning phase" of
#         a typical predictive coding life cycle, and since
#         demo will be in simulation mode, we will not
#         require an interactive session to get user labels
#
#   (b) - In this part we will simulate the review phase.
#
# note: we are not labeling our approach as TAR 1.0, 2.0 or
# 3.0 primarily because we don't know for sure which tech
# it is and secondly 1.0-3.0 gives an impression that 3.0 is 
# SOTA which truly isn't the case and we certain algorithmic
# modifications our can be called as STOA but here we will
# avoid addressing these issues and simply call it CAL
#------------------------------------------------------------

import matplotlib.pyplot as plt
plt.style.use('bmh')

from simulation import ActiveSimulation as CAL

T = 500 # queries budget

clf_cal = models.default_model()
cal = CAL()
cal.initialize(X, clf_cal, None, 'US', 'simulate', y, 10, True, 5)

# run queries
cal.run(T)

# check metrics:

# validation
val_X, val_y, _ = cal.dataset.get_labeled_data('valid')
val_X_scores = cal.model.decision_function(val_X)

plots.pr_curve(val_y, val_X_scores, 'valid', False)
val_precision, val_recall, val_accuracy, val_gmeans, val_auc = zip(*cal.metrics['valid'])
plots.plot_cal_metrics(val_precision, val_recall, val_accuracy, val_gmeans, val_auc, cal.eval_xs['valid'], 'Validation')

# performance on rest of data (unlabele/not queried)
# note: in real setting this won't be possible to use as quality / performance measure
unlabeled_X, unlabeled_indexes = cal.dataset.get_unlabeled_data()
unlabeled_y = cal.dataset.y_ideal[unlabeled_indexes]
unlabeled_X_scores = cal.model.decision_function(unlabeled_X)

plots.pr_curve(unlabeled_y, unlabeled_X_scores, 'unlabeled', False)
ul_precision, ul_recall, ul_accuracy, ul_gmeans, ul_auc = zip(*cal.metrics['simulate'])
plots.plot_cal_metrics(ul_precision, ul_recall, ul_accuracy, ul_gmeans, ul_auc, cal.eval_xs['simulate'], 'Unlabeled')

print("discovered {} relevant documents from an estimated number of {}".format(cal.relevant_found, cal.est_relevant))

#============================================================
#                IV (b) - Active Learning
#============================================================
# Review phase.
#------------------------------------------------------------

from simulation import ReviewStageSimulation as Review

# number of documents to review, set it to override default behavior
# which reviews #(total estimated relevant - relevant found in CAL)docs
R = None

# review instantiation will immediately (internally) rank the unlabeled
# documents based on dists from decision boundary
review = Review(
    cal.dataset, cal.model, cal.est_prevelance,
    cal.est_relevant, cal.relevant_found,
    num_docs_to_review=R
)

# number of top-k docs to review per batch
K = 10

review.review(K) # prepare batches

# document rank distribution as Relativity (set scale to True to get
# like their relevance scale from 0 to 1, which ideally is not good
# thing to do with SVM but keep raw dists as -neg ones tell that those
# instances are predicted as irrelevant rather than saying 0.2 relevant
# further, the magnitude tells how much effort to put in review, the
# farther the document is on left side, the higher is their irrelevance)
review.drd(False)

review.prp(50) # prioritized review progress like Relativity

_ = review.pr_at_k() # precision-recall @ K in predictive coding setting

#============================================================
#                 V - Test Evaluation
#============================================================
# Run model on test set and record metrics.
#------------------------------------------------------------

# train and report test results
cal_y_test_preds = cal.model.predict(X_test)
cal_results = {
    'accuracy': metrics.accuracy(y_test, cal_y_test_preds),
    'precision': metrics.precision(y_test, cal_y_test_preds),
    'recall': metrics.recall(y_test, cal_y_test_preds),
    'gmeans': metrics.g_means(y_test, cal_y_test_preds),
    'auc': metrics.auc(y_test, cal_y_test_preds),
    'cohen-kappa': metrics.user_machine_agreement(y_test, cal_y_test_preds)
}


print("Test set evaluation")
print("Supervised |    CAL       | Metric")
print("--------------------------------------")

for metric in cal_results.keys():
    print("%0.4f     |    %0.4f    | %s" % (supervised_results[metric], cal_results[metric], metric))

#============================================================
#                       Additional
#============================================================
#   *   Here we first glance at classifier's top features and 
#       show case a utility to visualize features role in 
#       predictive decision making making of the trained
#       model for any document
#
#   *   We show a hybrid methodology of dimensionality 
#       reduction and clustering techniques in combination
#       with document vectors (averaged over word2vec vectors
#       ) to perform initial sampling at much higher success
#       rate than random sampling
#------------------------------------------------------------

# let's look at top 30 features for both spam and ham classes
import features
from gensim.corpora import Dictionary

# we will need vocab object to get features names
vocab = Dictionary.load(join(FEATURES_DIR, 'tfidf_vocab'))

id2token = {id_:token for token, id_ in vocab.token2id.items() if id_ != -1}

# sanity check
assert len(id2token) == data_X.shape[1]

features_names = [id2token[i] for i in range(len(id2token))]
topfeats = features.clf_topn_coef(cal.model, features_names, 30, True)

# now let's probe into features of one example prediction from each class
pos_ind = np.random.choice(np.argwhere(y_test == 1).flatten())
neg_ind = np.random.choice(np.argwhere(y_test == 0).flatten())

_ = features.features_predictive_contrib(cal.model, X_test[pos_ind, :], np.array(features_names), True)
_ = features.features_predictive_contrib(cal.model, X_test[neg_ind, :], np.array(features_names), True)

# now lets look into LDDS and how it enhances of chance to sample
# more relevant documents random sampling with a simple prior
# knowledge bias i.e. we often have very small number of relevant
# documents compared to a large-set of irrelevant documents
#
# note: empirical results shows this approach workds best with
# prior knowledge holding true and so far only word2vec based
# document features show good representation of such sampling
#
# we call this LDDS: Low-dimensional unbalanced density sampling

import ldds

w2v_X = np.load(join(FEATURES_DIR, 'w2v_X.npy'))

ldds_output = ldds.LDDS(w2v_X, data_y, show=True)
