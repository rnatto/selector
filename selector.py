from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn.decomposition import PCA
from IPython.display import Image
from sklearn.svm import LinearSVC

from sklearn import preprocessing
import inspect
from scipy.io import arff
import pandas as pd
import numpy as np
import pydotplus
import arff
import sys
path = 'data/dataset-040919.arff'

result = arff.load(open(path, 'rb').read())

reload(sys)
sys.setdefaultencoding('utf-8')
result['data'] = np.array(result['data'])

col = [i[0] for i in result['attributes']]
stat = [i[0] for i in result['attributes'] if i[1] == 'STRING']

df = pd.DataFrame.from_dict(result['data'])

df.columns = col
# print(df.shape)
# drop not used features
drop_features = [
    'URL',
    'id',
    'tagName',
    'childsNumber',
    # 'textLength',
    'basePlatform',
    'targetPlatform',
    'baseBrowser',
    'targetBrowser',
    # 'baseDPI',
    # 'targetDPI',
    'baseScreenshot',
    'targetScreenshot',
    # 'baseX',
    # 'targetX',
    # 'baseY',
    # 'targetY'
    # 'baseHeight',
    # 'targetHeight',
    # 'baseWidth',
    # 'targetWidth',
    # 'baseParentX',
    # 'targetParentX',
    # 'baseParentY',
    # 'targetParentY',
    # 'imageDiff',
    # 'chiSquared',
    # 'baseDeviceWidth',
    # 'targetDeviceWidth',
    # 'baseViewportWidth',
    # 'targetViewportWidth',
    'xpath',
    'baseXpath',
    'targetXpath',
    # 'phash',
    # 'basePreviousSiblingLeft',
    # 'targetPreviousSiblingLeft',
    # 'basePreviousSiblingTop',
    # 'targetPreviousSiblingTop',
    # 'baseNextSiblingLeft',
    # 'targetNextSiblingLeft',
    # 'baseNextSiblingTop',
    # 'targetNextSiblingTop',
    'baseTextNodes',
    'targetTextNodes',
    'baseFontFamily',
    'targetFontFamily',
    'Result']
X = df.drop(drop_features, axis=1)
Y = df.Result
min_max_scaler = preprocessing.MinMaxScaler()
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# print(min_max_scaler.fit_transform(X))
# X = min_max_scaler.fit_transform(X)

X = pd.get_dummies(X)
k_values = [
    10, 20
    300, 400,
    500,
    1000, 2000, 3000, 4000, 5000
    # 10000, 15000, 20000, 'all'
]
methods = [f_classif, chi2]
classifiers = [
    LogisticRegression(random_state=0, solver='saga',
                       multi_class='multinomial'),
    DecisionTreeClassifier(random_state=0),
    LinearSVC(random_state=0, tol=1e-5)
]
# region TODO:
# Remover features textuais - done
# Rodar com os mesmos algoritmos
# Rodar com RFECV
# endregion
print('\n')
for k_value in k_values:
    # print('K')
    # print(k_value)
    # for method in methods:
        # print('Method')
        # print(method.func_name)
    # X_new = SelectKBest(method, k=k_value).fit_transform(X, Y)
    for classify in classifiers:
        selector = RFECV(classify, step=1, cv=5,
            min_features_to_select=k_value, scoring="f1_macro")
        selector = selector.fit(X, Y)
        scores = selector.grid_scores_
        print(selector.support_)
        print(selector.ranking_)
        print(scores)
        # scores = cross_val_score(
        # classify, X_new, Y, cv=5, scoring='f1_macro')
        # print(type(classify).__name__)
        # print(scores)
        mtd = {}
        KResult = {}
        clf = {}
        clf[type(classify).__name__ + str(k_value)] = np.amax(scores)
        # clf[type(classify).__name__ + str(k_value)] = np.mean(scores)
        # mtd[method.func_name] = clf
        mtd['RFECV'] = clf
        # KResult[k_value] = mtd
        print(mtd)
