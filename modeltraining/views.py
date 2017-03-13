import django
from django.shortcuts import render
from django.db import models
from modeltraining.models import *
from django.shortcuts import render
import pandas
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

# Create your views here.
from django.http import HttpResponse


def index(request):
    all_users_entries = users_app.objects.all()[:50]
    names = ['ID', 'Name', 'Screen Name', 'Status Count', 'Followers Count', 'Friend Count', 'Favourites Count', 'Listed Count', 'Created At', 'Url',
             'Language', 'Time Zone', 'Location', 'Default Profile', 'Default Profile Image', 'Geo Enabled', 'Profile Image URL', 'Profile Banner URL',
             'Profile User Background Image', 'Profile Background image url', 'Profile Text Colour', 'Profile Image url https',
             'Profile sidebar border color','Profile background tile', 'Profile Sidebar Fill Colour', 'Profile Background Image url', 'Profile background colour',
             'Profile link colour', 'Utc Offset', 'Protected', 'Verified', 'Updated', 'Dataset', 'Bot']

    # values = all_users_entries.values('id', 'name', 'screen_name', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'created_at',
    #                                   'url', 'lang', 'time_zone', 'location', 'default_profile', 'default_profile_image', 'geo_enabled', 'profile_image_url',
    #                                   'profile_banner_url', 'profile_use_background_image', 'profile_background_image_url_https', 'profile_text_color',
    #                                   'profile_image_url_https','profile_sidebar_fill_color', 'profile_background_image_url', 'profile_background_color',
    #                                   'profile_link_color', 'utc_offset', 'protected', 'verified', 'description', 'updated', 'dataset', 'bot')

    values = all_users_entries.values('statuses_count', 'followers_count', 'friends_count',
                                      'favourites_count', 'listed_count')

    dataset = pandas.DataFrame.from_records(values)
    print(dataset.shape)
    print(dataset.head(20))

    print(dataset.describe())

    fig = Figure(figsize=(30, 10))
    ax = fig.add_subplot(999)
    # ax2 = fig.add_subplot(111)
    dataset.plot(kind='box', subplots=True, layout=(6, 6), sharex=False, sharey=False, ax=ax)
    #dataset.hist(ax=ax)
    #scatter_matrix(dataset, ax=ax)

    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    canvas.print_png(response)

    # array = dataset.values
    # X = array[:, 0:4]
    # Y = array[:, 4]
    #
    # validation_size = 0.20
    # seed = 71
    #
    # X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
    #                                                                                 random_state=seed)
    #
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

    return response
    #return HttpResponse(all_users_entries.values('id', 'name'))


