from modeltraining.models import *
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from django.http import HttpResponse
from textblob import TextBlob
import re
from sklearn.metrics import accuracy_score
from yandex_translate import YandexTranslate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import datetime


def index(request):
    all_users_entries = users_app.objects.all()
    all_tweet_entries = tweets_app.objects.all()
    bots = tweets_app.objects.all().filter(bot=True)
    real = tweets_app.objects.all().filter(bot=False)

#    nearest_neighbor(all_users_entries)

#    predict =  linear_model(all_users_entries)

#    predict = support_vector_machines(all_users_entries)

    #predict = random_forest(all_users_entries)
    #predict = random_forest_tweets(bots, real)

    #sorted_tweets = sentiment_analyses(all_tweet_entries)

    timing = time_analyses(all_tweet_entries)

    # names = ['ID', 'Name', 'Screen Name', 'Status Count', 'Followers Count', 'Friend Count', 'Favourites Count',
    #          'Listed Count', 'Created At', 'Url', 'Language', 'Time Zone', 'Location', 'Default Profile',
    #          'Default Profile Image', 'Geo Enabled', 'Profile Image URL', 'Profile Banner URL',
    #          'Profile User Background Image', 'Profile Background image url', 'Profile Text Colour',
    #          'Profile Image url https', 'Profile sidebar border color', 'Profile background tile',
    #          'Profile Sidebar Fill Colour', 'Profile Background Image url', 'Profile background colour',
    #          'Profile link colour', 'Utc Offset', 'Protected', 'Verified', 'Updated', 'Dataset', 'Bot']

    return HttpResponse(timing)


def random_forest_tweets(bots, real):

    upscaled = []
    upscaled_data = []

    for i in range(0, 5):
        upscaled.extend(bots)

    print(len(upscaled))

    upscaled_data.extend(upscaled)
    upscaled_data.extend(real)

    print(len(upscaled_data))

    tweetdata_x_django = []
    tweetdata_y_django = []

    for tweet in upscaled_data:
        tweetdata_x_django.append([tweet.retweet_count, tweet.num_hashtags, tweet.num_urls, tweet.num_mentions])
        tweetdata_y_django.append(tweet.bot)

    # tweetdata_x_django = upscaled_data.values_list('retweet_count', 'num_hashtags', 'num_urls', 'num_mentions')
    # tweetdata_y_django = upscaled_data.values_list('bot', flat=True)

    tweetdata_y_bool = []

    for tweet in tweetdata_y_django:
        if tweet is True:
            tweetdata_y_bool.append(1)
        else:
            tweetdata_y_bool.append(0)

    tweetdata_x = np.core.records.fromrecords(tweetdata_x_django, names=['retweet_count', 'num_hashtags', 'num_urls',
                                                                         'num_mentions'])
    tweetdata_y = np.fromiter(tweetdata_y_bool, np.dtype('int_'))

    df = pd.DataFrame(tweetdata_x, columns=['retweet_count', 'num_hashtags', 'num_urls', 'num_mentions'])

    df['Bot'] = pd.Categorical.from_array(tweetdata_y)
    df['to_train'] = np.random.uniform(0, 1, len(df)) <= .75

    print(df.head())

    train, test = df[df['to_train'] == True], df[df['to_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    tweet_data_names = df.columns[:4]

    y = train['Bot']

    rf = RandomForestClassifier(n_jobs=2, n_estimators=1000, max_features="log2")

    rf.fit(train[tweet_data_names], y)

    predict = rf.predict(test[tweet_data_names])

    print('rf done')

    test_y = test['Bot']

    gnb = GaussianNB()
    gnb.fit(train[tweet_data_names], y)

    predict2 = gnb.predict(test[tweet_data_names])
    print('neural done')

    #    predict=rf.predict_proba(test[user_data_names])

    filename = 'random_forest_tweet_model.sav'
    pickle.dump(rf, open(filename, 'wb'))

    filename = 'neural_tweet_model.sav'
    pickle.dump(gnb, open(filename, 'wb'))

    # print(len(predict))
    # print((count / len(predict) * 100))

    # print(precision_recall_fscore_support(predict, true))

    print(accuracy_score(test_y, predict))
    print(accuracy_score(test_y, predict2))

    return predict


def random_forest(all_users_entries):
    userdata_x_django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userdata_y_django = all_users_entries.values_list('bot', flat=True)

    userdata_y_bool = []

    for user in userdata_y_django:
        if user is True:
            userdata_y_bool.append(1)
        else:
            userdata_y_bool.append(0)

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userdata_y = np.fromiter(userdata_y_bool, np.dtype('int_'))

    df = pd.DataFrame(userdata_x, columns=['Statuses Count', 'Followers_Count', 'Friends Count', 'Favourite Count'])

    print(len(userdata_x))
    print(len(userdata_y))

    df['Bot'] = pd.Categorical.from_array(userdata_y)
    df['to_train'] = np.random.uniform(0, 1, len(df)) <= .75

    print(df.head())

    train, test = df[df['to_train'] == True], df[df['to_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    user_data_names = df.columns[:4]

    y = train['Bot']

    rf = RandomForestClassifier(n_jobs=2)

    rf.fit(train[user_data_names], y)

    predict = rf.predict(test[user_data_names])

    test_y = test['Bot']

#    predict=rf.predict_proba(test[user_data_names])

    count = 0
    true = []
    for i in range(0, len(predict)):
        print(predict[i], '=', test_y.iloc[i])
        true.extend(test_y.iloc[[i]])
        if predict[i] == test_y.iloc[[i]]:
            count += 1

    print(len(predict))
    print((count / len(predict)*100))

    print(precision_recall_fscore_support(predict, true))
    filename = 'random_forest_user_model.sav'
    pickle.dump(rf, open(filename, 'wb'))

    #confusion matrix
    #oversample data
    return predict


def support_vector_machines(all_users_entries):

    userdata_x_django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userdata_y_django = all_users_entries.values_list('bot', flat=True)

    userdata_y_bool = []

    for user in userdata_y_django:
        if user is True:
            userdata_y_bool.append(1)
        else:
            userdata_y_bool.append(0)

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userdata_y = np.fromiter(userdata_y_bool, np.dtype('int_'))

    np.random.seed(0)
    indices = np.random.permutation(len(userdata_x))
    userdata_x_train = userdata_x[indices[:-.1*len(userdata_x)]]
    userdata_y_train = userdata_y[indices[:-.1*len(userdata_y)]]
    userdata_x_test = userdata_x[indices[-.1*len(userdata_x):]]
    userdata_y_test = userdata_y[indices[-.1*len(userdata_y):]]

    userdata_x_train = userdata_x_train.reshape(len(userdata_x_train), 1)
    userdata_x_test = userdata_x_test.reshape(len(userdata_x_test), 1)

    # n_sample = len(userdata_x_train)
    #
    # np.random.seed(0)
    # order = np.random.permutation(n_sample)
    # userdata_x_train = userdata_x_train[order]
    # userdata_y_train = userdata_y_train[order].astype(np.float)

    svc = svm.SVC(kernel='linear', cache_size=7000)

    svc.fit(userdata_x_train, userdata_y_train)

    predict = svc.predict(userdata_x_test)

    count = 0

    for i in range(0, len(predict)):
        if predict[i] == userdata_y_test[i]:
            count += 1

    print(predict)
    print(userdata_y_test)
    print(len(userdata_y_test))

    print((count / len(userdata_x_test)))

    return predict


def linear_model(all_users_entries):
    userdata_x_django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userdata_y_django = all_users_entries.values_list('bot', flat=True)

    userdata_y_bool = []

    for user in userdata_y_django:
        if user is True:
            userdata_y_bool.append(1)
        else:
            userdata_y_bool.append(0)

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userdata_y = np.fromiter(userdata_y_bool, np.dtype('int_'))

    np.random.seed(0)
    indices = np.random.permutation(len(userdata_x))
    userdata_x_train = userdata_x[indices[:-.1*len(userdata_x)]]
    userdata_y_train = userdata_y[indices[:-.1*len(userdata_y)]]
    userdata_x_test = userdata_x[indices[-.1*len(userdata_x):]]
    userdata_y_test = userdata_y[indices[-.1*len(userdata_y):]]
    # userdata_x_train = userdata_x[indices[:-20]]
    # userdata_y_train = userdata_y[indices[:-20]]
    # userdata_x_test = userdata_x[indices[-20:]]
    # userdata_y_test = userdata_y[indices[-20:]]

    userdata_x_train = userdata_x_train.reshape(len(userdata_x_train), 1)
    userdata_x_test = userdata_x_test.reshape(len(userdata_x_test), 1)

    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(userdata_x_train, userdata_y_train)

    print(regr.coef_)

    predict = np.mean((regr.predict(userdata_x_test.astype(float))-userdata_y_test.astype(float))**2)

    print(regr.score(userdata_x_test.astype(float), userdata_y_test.astype(float)))

    return predict


def nearest_neighbor(all_users_entries):
    userdata_x_django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userdata_y_django = all_users_entries.values_list('bot', flat=True)

    userdata_y_bool = []

    for user in userdata_y_django:
        if user is True:
            userdata_y_bool.append(1)
        else:
            userdata_y_bool.append(0)

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userdata_y = np.fromiter(userdata_y_bool, np.dtype('int_'))

    np.random.seed(0)
    indices = np.random.permutation(len(userdata_x))
    userdata_x_train = userdata_x[indices[:-.1*len(userdata_x)]]
    userdata_y_train = userdata_y[indices[:-.1*len(userdata_y)]]
    userdata_x_test = userdata_x[indices[-.1*len(userdata_x):]]
    userdata_y_test = userdata_y[indices[-.1*len(userdata_y):]]

    userdata_x_train = userdata_x_train.reshape(len(userdata_x_train), 1)
    userdata_x_test = userdata_x_test.reshape(len(userdata_x_test), 1)

    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                         weights='uniform')
    knn.fit(userdata_x_train, userdata_y_train)


    predict = knn.predict(userdata_x_test)

    print(predict)
    print(userdata_y_test)
    print(len(userdata_y_test))

    count = 0

    for i in range(0, len(predict)):
        if predict[i] == userdata_y_test[i]:
            count += 1

    print((count / len(userdata_x_test)))


def sentiment_analyses(all_tweet_entries):

    tweets = all_tweet_entries.values_list('user_id', 'text', 'bot', 'lang', 'id')

    sorted_tweets = sorted(tweets, key=lambda tw: tw[0])

    tweet_id = sorted_tweets[0][0]
    sentiment = [0, 0, 0, 0]
    sentiment_list = []
    count = 0
    print(tweet_id)
    batch_update = ''

    for tweet in sorted_tweets:
        if tweet[0] != tweet_id:
            if tweet[2]:
                sentiment[3] = 1
            else:
                sentiment[3] = 0

            #print(batch_update)

            if batch_update != '':
                print(batch_update)
                translated_string = translate_string(batch_update)

                #print(translated_string)

            #insert_sentiment(tweet[0], sentiment)
            sentiment_list.append(sentiment)
            sentiment = [0, 0, 0, 0]
            count = 0
            batch_update = ''
            tweet_id = tweet[0]

        if count < 1000:

            passed_tweet = strip_tweet(tweet[1])

            if tweet[3] != 'en':
                batch_update += str(tweet[4]) + ':::;:::' + passed_tweet + ';;;:;;;'

            tweet_sentiment = get_sentiment(tweet[1])

            if tweet_sentiment == 'positive':
                sentiment[0] += 1
            elif tweet_sentiment == 'neutral':
                sentiment[1] += 1
            elif tweet_sentiment == 'negative':
                sentiment[2] += 1
            count += 1

    predict = random_forest_sentiment(sentiment_list)
    predict = nearest_neighbor_sentiment(sentiment_list)
    print(len(sentiment_list))
    return predict


def nearest_neighbor_sentiment(sentiment_list):
    sentiment = np.core.records.fromrecords(sentiment_list, names=['positive', 'neutral', 'negative', 'bot'])

    df = pd.DataFrame(sentiment, columns=['positive', 'neutral', 'negative', 'bot'])

    df['to_train'] = np.random.uniform(0, 1, len(df)) <= .80

    print(df.head())

    train, test = df[df['to_train'] == True], df[df['to_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    sentiment_names = df.columns[:3]

    y = train['bot']

    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='uniform')
    knn.fit(train[sentiment_names], y)

    predict = knn.predict(test[sentiment_names])

    test_y = test['bot']

    svc = svm.SVC(cache_size=7000, kernel='linear')
    svc.fit(train[sentiment_names], y)

    predict2 = svc.predict(test[sentiment_names])

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train[sentiment_names], y)

    predict3 = clf.predict(test[sentiment_names])

    gnb = GaussianNB()
    gnb.fit(train[sentiment_names], y)

    predict4 = gnb.predict(test[sentiment_names])

    print('Nearest : ', accuracy_score(test_y, predict))
    print('SVM: ', accuracy_score(test_y, predict2))
    print('Neural: ', accuracy_score(test_y, predict3))
    print('GNB: ', accuracy_score(test_y, predict4))

    return predict


def get_sentiment(text):
    tweet_sentiment = TextBlob(text)

    if tweet_sentiment.sentiment.polarity > 0:
        return 'positive'
    elif tweet_sentiment.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def random_forest_sentiment(sentiment_list):
    print('in')

    sentiment = np.core.records.fromrecords(sentiment_list, names=['positive', 'neutral', 'negative', 'bot'])

    df = pd.DataFrame(sentiment, columns=['positive', 'neutral', 'negative', 'bot'])

    df['to_train'] = np.random.uniform(0, 1, len(df)) <= .75

    print(df.head())

    train, test = df[df['to_train'] == True], df[df['to_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    sentiment_names = df.columns[:3]

    y = train['bot']
    test_y = test['bot']

    rf = RandomForestClassifier(n_jobs=2, n_estimators=10000, max_features="sqrt")
    rf.fit(train[sentiment_names], y)
    predict = rf.predict(test[sentiment_names])


    #    predict=rf.predict_proba(test[user_data_names])

    # print(len(predict))
    # print((count / len(predict) * 100))

    filename = 'random_forest_sentiment_model.sav'
    pickle.dump(rf, open(filename, 'wb'))

    print('Random: ', accuracy_score(test_y, predict))
    # confusion matrix
    # oversample data
    return predict


def strip_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w +:\ / \ / \S +)", " ", tweet).split())


def translate_string(text_passed):
    translate = YandexTranslate('trnsl.1.1.20170322T133238Z.691da702cdb5dbe6.e00850c5473b4c71346d4fcb0f5fff807b8c4f43')#emmet.hanratty
    #translate = YandexTranslate('trnsl.1.1.20170322T162921Z.24bcd1a440ac1ff1.4ae8817644cf4f0773b5134f2440b780af7044b7')#emmethanratty
    #translate = YandexTranslate('trnsl.1.1.20170322T165243Z.cc4baa633b54ee81.0e9596dca723df99fb296e20846bc47b66bb678f')#hanrattyemmet
    #translate = YandexTranslate('trnsl.1.1.20170322T133603Z.e0a9d5b997f7a2be.c687257bea28bf9d7cf7a7fb9eca4185e7a807a5')#hanratty.emmet
    #translate = YandexTranslate('trnsl.1.1.20170322T171859Z.0d7715e4714685b8.06cf2e36ed1bf7f690a849c257ef708f0bc5b9d4')#hanratty.emmet2017
    #translate = YandexTranslate('trnsl.1.1.20170322T173856Z.03bf0f193d8e4190.59e0b3e30b5055a4cb042664261e0d28dcdb0723')#emmet.hanratty2017
    #translate = YandexTranslate('trnsl.1.1.20170322T185516Z.decb48e6428fa93b.3a1dd757f63e0c4b010dc8d3badb2d3838c94fe5') # e.hanratty
    #translate = YandexTranslate('trnsl.1.1.20170322T192725Z.17b6e0c431553d40.92b0884a635191e901f2cb79253239e6688d6f41')#hanratty.e
    #translate = YandexTranslate('trnsl.1.1.20170322T194833Z.866d740e343c356a.3021ce8a6ba2e802ca0543cc8c42310b91ea6533')# em.hanratty
    #translate = YandexTranslate('trnsl.1.1.20170322T200249Z.aeb9bc1f1bdb5093.c1d999f94d59f1216bd2a4f8325e846033113c6f')  # e.hanratty2017
    #translate = YandexTranslate('trnsl.1.1.20170322T200516Z.8a278bab73cd7892.e5cdbb610699b45f69490ba1c04b5bd8d83a0f2d')  # hanratty.emmet2018
    #translate = YandexTranslate('trnsl.1.1.20170322T222615Z.e2e1e4b1b09df158.e4aef95a40a49b42f48f50857a21d15ec46ba5f1') # emmet.hanratty2018
    #translate = YandexTranslate('trnsl.1.1.20170322T224837Z.e0c8b2987ddf8ad7.d29c185d939ad5e9aae7354ccd029accdcf7596c')#ya.hanratty
    #translate = YandexTranslate('trnsl.1.1.20170322T231048Z.3de42387c73276fc.21beab3c4a0c5fc05f6f6871e51aeeffe53a2efd')#hanratty.em

    translated = translate.translate(text_passed, 'en')

    batch_update(translated['text'][0])

    return translated['text'][0]


def insert_sentiment(passed_id, sentiment):
    sent = sentiment_app(id=passed_id, positive=sentiment[0], neutral=sentiment[1], negative=[2], bot=sentiment[3])
    sent.save()


def translated_tweet_update(tweet_id, text):
    t = tweets_app.objects.get(id=tweet_id)

    t.lang = 'en'
    t.text = text
    t.save()
    #print('Tweet: ', tweet_id, ' Text: ', text, ' updated')


def batch_update(text):
    tweets_and_ids = text.split(';;;:;;;')

    for tweets in tweets_and_ids:

        split_tweet = tweets.split(':::;:::')
        if split_tweet[0] != '' and split_tweet[1] != '':
            #print('id: ', split_tweet[0], 'text: ', split_tweet[1])
            tweet_id = int(split_tweet[0])
            t = tweets_app.objects.get(id=tweet_id)
            t.lang = 'en'
            t.text = split_tweet[1]
            t.save()
        elif split_tweet[0] != '':
            tweet_id = int(split_tweet[0])
            t = tweets_app.objects.get(id=tweet_id)
            t.lang = 'en'
            t.save()

    print("50 in")




def time_analyses(all_tweet_entries):
    tweets = all_tweet_entries.values_list('user_id', 'created_at', 'bot')

    timing_array = []
    tweet_timing = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count = 0
    previous_day = ['', '', '']

    sorted_tweets = sorted(tweets, key=lambda tw: tw[0])
    tweet_id = sorted_tweets[0][0]

    for tweet in sorted_tweets:
        if tweet[0] != tweet_id:
            if tweet[2]:
                tweet_timing[9] = 1
            else:
                tweet_timing[9] = 0

            timing_array.append(tweet_timing)

            tweet_timing = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            tweet_id = tweet[0]
            count = 0
            previous_day = ['', '', '']

        #print(tweet[1], tweet[0], tweet[2])

        if count <= 1000:
            full_date = tweet[1]

            split_date = full_date.split(' ')

            day = split_date[0]
            month = split_date[1]
            day_num = split_date[2]
            time = split_date[3]
            year = split_date[5]


            datetime_object = datetime.datetime.strptime(time, "%H:%M:%S")

            # print(day, month, datetime_object)

            if datetime.time(0, 0, 0) <= datetime_object.time() <= datetime.time(2, 59, 0):
                tweet_timing[0] += 1
                # print(time)
            elif datetime.time(3, 0, 0) <= datetime_object.time() <= datetime.time(5, 59, 0):
                tweet_timing[1] += 1
            elif datetime.time(6, 0, 0) <= datetime_object.time() <= datetime.time(8, 59, 0):
                tweet_timing[2] += 1
            elif datetime.time(9, 0, 0) <= datetime_object.time() <= datetime.time(11, 59, 0):
                tweet_timing[3] += 1
            elif datetime.time(12, 0, 0) <= datetime_object.time() <= datetime.time(14, 59, 0):
                tweet_timing[4] += 1
            elif datetime.time(15, 0, 0) <= datetime_object.time() <= datetime.time(17, 59, 0):
                tweet_timing[5] += 1
            elif datetime.time(18, 0, 0) <= datetime_object.time() <= datetime.time(20, 59, 0):
                tweet_timing[6] += 1
            elif datetime.time(21, 0, 0) <= datetime_object.time() <= datetime.time(23, 59, 0):
                tweet_timing[7] += 1

            if month == previous_day[0] and day_num == previous_day[1] and year == previous_day[2]:
                tweet_timing[8] += 1

            previous_day[0] = month
            previous_day[1] = day_num
            previous_day[2] = year
            count += 1


    predict = timing_model_creation(timing_array)
    return predict


def timing_model_creation(timing_array):

    bots = []
    real = []

    scaled = []
    full_scaled = []

    for user in timing_array:
        if user[9] == 1:
            bots.append(user)
        else:
            real.append(user)

    for i in range(0, 2):
        scaled.extend(real)

    full_scaled.extend(bots)
    full_scaled.extend(scaled)

    timing = np.core.records.fromrecords(full_scaled, names=['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21',
                                                              '21-24', 'same day', 'bot'])

    df = pd.DataFrame(timing, columns=['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21', '21-24', 'same day',
                                       'bot'])

    df['to_train'] = np.random.uniform(0, 1, len(df)) <= .75

    print(df.head())

    train, test = df[df['to_train'] == True], df[df['to_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    timing_names = df.columns[:8]

    y = train['bot']
    test_y = test['bot']

    rf = RandomForestClassifier(n_jobs=-1, n_estimators=10000, max_features=.2, min_samples_leaf=100, oob_score=True, random_state=50)
    rf.fit(train[timing_names], y)
    predict = rf.predict(test[timing_names])

    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1,  n_neighbors=5, p=2, weights='uniform')
    knn.fit(train[timing_names], y)
    predict2 = knn.predict(test[timing_names])

    svc = svm.SVC(cache_size=7000, kernel='linear')
    svc.fit(train[timing_names], y)
    predict3 = svc.predict(test[timing_names])

    clf = MLPClassifier()
    clf.fit(train[timing_names], y)
    predict4 = clf.predict(test[timing_names])

    gnb = GaussianNB()
    gnb.fit(train[timing_names], y)
    predict5 = gnb.predict(test[timing_names])


    #    predict=rf.predict_proba(test[user_data_names])

    # print(len(predict))
    # print((count / len(predict) * 100))

    filename = 'random_forest_timing_model.sav'
    pickle.dump(rf, open(filename, 'wb'))

    print('rf: ', accuracy_score(test_y, predict))
    print('knn: ', accuracy_score(test_y, predict2))
    print('svm: ', accuracy_score(test_y, predict3))
    print('clf: ', accuracy_score(test_y, predict4))
    print('gnb: ', accuracy_score(test_y, predict5))
    # confusion matrix
    # oversample data
    return predict, predict2, predict3, predict4, predict5





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

