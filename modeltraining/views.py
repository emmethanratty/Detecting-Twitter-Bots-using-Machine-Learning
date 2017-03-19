import django
from django.shortcuts import render
from django.db import models
from modeltraining.models import *
from django.shortcuts import render
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib.backends.backend_agg import FigureCanvasAgg, FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pickle

# Create your views here.
from django.http import HttpResponse


def index(request):
    all_users_entries = users_app.objects.all()

    #nearest_neighbor(all_users_entries)

    #predict =  linear_model(all_users_entries)

    #predict = support_vector_machines(all_users_entries)

    predict = random_forest(all_users_entries)

    names = ['ID', 'Name', 'Screen Name', 'Status Count', 'Followers Count', 'Friend Count', 'Favourites Count',
             'Listed Count', 'Created At', 'Url', 'Language', 'Time Zone', 'Location', 'Default Profile',
             'Default Profile Image', 'Geo Enabled', 'Profile Image URL', 'Profile Banner URL',
             'Profile User Background Image', 'Profile Background image url', 'Profile Text Colour',
             'Profile Image url https','Profile sidebar border color', 'Profile background tile',
             'Profile Sidebar Fill Colour', 'Profile Background Image url', 'Profile background colour',
             'Profile link colour', 'Utc Offset', 'Protected', 'Verified', 'Updated', 'Dataset', 'Bot']

    return HttpResponse(predict)


def random_forest(all_users_entries):
    userData_X_Django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userData_Y_Django = all_users_entries.values_list('bot', flat=True)

    userData_Y_Bool = []

    for user in userData_Y_Django:
        if user == True:
            userData_Y_Bool.append(1)
        else:
            userData_Y_Bool.append(0)

    userData_X = np.core.records.fromrecords(userData_X_Django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userData_Y = np.fromiter(userData_Y_Bool, np.dtype('int_'))

    from sklearn.ensemble import RandomForestClassifier

    df = pd.DataFrame(userData_X, columns=['Statuses Count', 'Followers_Count', 'Friends Count', 'Favourite Count'])

    print(len(userData_X))
    print(len(userData_Y))

    df['Bot'] = pd.Categorical.from_array(userData_Y)
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

    print(df.head())

    train, test = df[df['is_train'] == True], df[df['is_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    userDataNames = df.columns[:4]

    y = train['Bot']

    rf = RandomForestClassifier(n_jobs=2)

    rf.fit(train[userDataNames], y)

    predict = rf.predict(test[userDataNames])

    test_y = test['Bot']

    #predict=rf.predict_proba(test[userDataNames])

    count = 0

    for i in range(0, len(predict)):
        print(predict[i], '=', test_y.iloc[i])
        if predict[i] == test_y.iloc[[i]]:
            count += 1

    print((count / len(predict)*100))

    filename = 'random_forest_model.sav'
    pickle.dump(rf, open(filename, 'wb'))

    return predict


def support_vector_machines(all_users_entries):

    userData_X_Django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userData_Y_Django = all_users_entries.values_list('bot', flat=True)

    userData_Y_Bool = []

    for user in userData_Y_Django:
        if user == True:
            userData_Y_Bool.append(1)
        else:
            userData_Y_Bool.append(0)

    userData_X = np.core.records.fromrecords(userData_X_Django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userData_Y = np.fromiter(userData_Y_Bool, np.dtype('int_'))
    unique = np.unique(userData_Y)

    np.random.seed(0)
    indices = np.random.permutation(len(userData_X))
    userData_X_train = userData_X[indices[:-.1*len(userData_X)]]
    userData_Y_train = userData_Y[indices[:-.1*len(userData_Y)]]
    userData_X_test = userData_X[indices[-.1*len(userData_X):]]
    userData_Y_test = userData_Y[indices[-.1*len(userData_Y):]]

    userData_X_train = userData_X_train.reshape(len(userData_X_train), 1)
    userData_X_test = userData_X_test.reshape(len(userData_X_test), 1)

    # n_sample = len(userData_X_train)
    #
    # np.random.seed(0)
    # order = np.random.permutation(n_sample)
    # userData_X_train = userData_X_train[order]
    # userData_Y_train = userData_Y_train[order].astype(np.float)

    from sklearn import svm

    svc = svm.SVC(kernel='linear')

    SVC(cache_size=7000)

    svc.fit(userData_X_train, userData_Y_train)


    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)

    #predict = "Worked"
    predict = svc.predict(userData_X_test)

    count = 0

    for i in range(0, len(predict)):
        if predict[i] == userData_Y_test[i]:
            count += 1

    print(predict)
    print(userData_Y_test)
    print(len(userData_Y_test))

    print((count / len(userData_X_test)))

    return predict

def linear_model(all_users_entries):
    userData_X_Django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userData_Y_Django = all_users_entries.values_list('bot', flat=True)

    userData_Y_Bool = []

    for user in userData_Y_Django:
        if user == True:
            userData_Y_Bool.append(1)
        else:
            userData_Y_Bool.append(0)

    userData_X = np.core.records.fromrecords(userData_X_Django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userData_Y = np.fromiter(userData_Y_Bool, np.dtype('int_'))
    unique = np.unique(userData_Y)

    np.random.seed(0)
    indices = np.random.permutation(len(userData_X))
    userData_X_train = userData_X[indices[:-.1*len(userData_X)]]
    userData_Y_train = userData_Y[indices[:-.1*len(userData_Y)]]
    userData_X_test = userData_X[indices[-.1*len(userData_X):]]
    userData_Y_test = userData_Y[indices[-.1*len(userData_Y):]]
    # userData_X_train = userData_X[indices[:-20]]
    # userData_Y_train = userData_Y[indices[:-20]]
    # userData_X_test = userData_X[indices[-20:]]
    # userData_Y_test = userData_Y[indices[-20:]]

    userData_X_train = userData_X_train.reshape(len(userData_X_train), 1)
    userData_X_test = userData_X_test.reshape(len(userData_X_test), 1)

    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(userData_X_train, userData_Y_train)

    print(regr.coef_)

    predict = np.mean((regr.predict(userData_X_test.astype(float))-userData_Y_test.astype(float))**2)
    #print(predict)
    print(regr.score(userData_X_test.astype(float), userData_Y_test.astype(float)))

    # count = 0
    #
    # for i in range(0, len(predict)):
    #     if predict[i] == userData_Y_test[i]:
    #         count += 1
    #
    # print(predict)
    # print(userData_Y_test)
    # print(len(userData_Y_test))
    #
    # print((count / len(userData_X_test)))

    return predict


def nearest_neighbor(all_users_entries):
    userData_X_Django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userData_Y_Django = all_users_entries.values_list('bot', flat=True)

    userData_Y_Bool = []

    for user in userData_Y_Django:
        if user == True:
            userData_Y_Bool.append(1)
        else:
            userData_Y_Bool.append(0)

    userData_X = np.core.records.fromrecords(userData_X_Django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userData_Y = np.fromiter(userData_Y_Bool, np.dtype('int_'))
    unique = np.unique(userData_Y)

    np.random.seed(0)
    indices = np.random.permutation(len(userData_X))
    userData_X_train = userData_X[indices[:-.1*len(userData_X)]]
    userData_Y_train = userData_Y[indices[:-.1*len(userData_Y)]]
    userData_X_test = userData_X[indices[-.1*len(userData_X):]]
    userData_Y_test = userData_Y[indices[-.1*len(userData_Y):]]
    # userData_X_train = userData_X[indices[:-10]]
    # userData_Y_train = userData_Y[indices[:-10]]
    # userData_X_test = userData_X[indices[-10:]]
    # userData_Y_test = userData_Y[indices[-10:]]

    userData_X_train = userData_X_train.reshape(len(userData_X_train), 1)
    userData_X_test = userData_X_test.reshape(len(userData_X_test), 1)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(userData_X_train, userData_Y_train)

    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                         weights='uniform')

    predict = knn.predict(userData_X_test)

    print(predict)
    print(userData_Y_test)
    print(len(userData_Y_test))

    count = 0

    for i in range(0, len(predict)):
        if predict[i] == userData_Y_test[i]:
            count += 1

    print((count / len(userData_X_test)))













    # values = all_users_entries.values('id', 'name', 'screen_name', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'created_at',
    #                                   'url', 'lang', 'time_zone', 'location', 'default_profile', 'default_profile_image', 'geo_enabled', 'profile_image_url',
    #                                   'profile_banner_url', 'profile_use_background_image', 'profile_background_image_url_https', 'profile_text_color',
    #                                   'profile_image_url_https','profile_sidebar_fill_color', 'profile_background_image_url', 'profile_background_color',
    #                                   'profile_link_color', 'utc_offset', 'protected', 'verified', 'description', 'updated', 'dataset', 'bot')


    #values = all_users_entries.values('statuses_count', 'followers_count', 'friends_count',
    #                                  'favourites_count', 'listed_count', 'bot')





    # print(dataset.shape)
    # print(dataset.head(20))
    #
    # print(dataset.describe())

    # fig = Figure(figsize=(30, 10))
    # ax = fig.add_subplot(111)
    # ax2 = fig.add_subplot(111)
    #dataset.plot(kind='box', subplots=True, layout=(6, 6), sharex=False, sharey=False, ax=ax)
    #dataset.hist(ax=ax)
    #scatter_matrix(dataset, ax=ax)

    # array = dataset.values
    # X = array[:, 0:4]
    # Y = array[:, 4]
    #
    # validation_size = 0.20
    # seed = 71
    #
    # X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
    #                                                                                 random_state=seed)
    # seed = 7
    # scoring = 'accuracy'
    #
    # dsModels = []
    # dsModels.append(('LR', LogisticRegression()))
    # dsModels.append(('LDA', LinearDiscriminantAnalysis()))
    # dsModels.append(('KNN', KNeighborsClassifier()))
    # dsModels.append(('CART', DecisionTreeClassifier()))
    # dsModels.append(('NB', GaussianNB()))
    # dsModels.append(('SVM', SVC()))
    #
    # results = []
    # names = []
    # for name, model in dsModels:
    #     kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #     cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)
    #
    # fig.suptitle('Algorithm Comparison')
    #
    # ax.set_xticklabels(names)
    #
    # canvas = FigureCanvas(fig)
    # response = django.http.HttpResponse(content_type='image/png')
    # canvas.print_png(response)

    #return response

