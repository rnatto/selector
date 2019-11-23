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
print(df.shape)
# drop not used features
drop_features = [
    'URL',
    'id',
    'tagName',
    # 'childsNumber',
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
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(min_max_scaler.fit_transform(X, Y))
