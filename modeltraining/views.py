from modeltraining.models import *
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import re
from sklearn.metrics import accuracy_score
from yandex_translate import YandexTranslate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import datetime
from django.shortcuts import render


# function for the creation of the models
def index(request):
    # get all user, tweets and separate them by bots and real users
    all_users_entries = users_app.objects.all()
    all_tweet_entries = tweets_app.objects.all()
    bots = tweets_app.objects.all().filter(bot=True)
    real = tweets_app.objects.all().filter(bot=False)

    # creation of models for the user and return the prediction
    nearest_neighbor(all_users_entries)
    print('nn done')
    predict_linear = linear_model(all_users_entries)
    print('lm done', predict_linear)
    # Takes a long time to run, run at own risk
    # predict_svm = support_vector_machines(all_users_entries)
    # print('svm done', predict_svm)
    predict_rf = random_forest(all_users_entries)
    print('rf done', predict_rf)

    # creation of the models for tweets and return the tweet prediction
    predict_rf_tweet = random_forest_tweets(bots, real)
    print('tweet rf done', predict_rf_tweet)

    # creation of the sentiment model and the return of the sentiment prediction
    sorted_tweets = sentiment_analyses(all_tweet_entries)
    print('sentiment done', sorted_tweets)

    # creation of the time model nd the return of the timing prediction
    timing = time_analyses(all_tweet_entries)
    print('time done', timing)

    return render(request, 'modeltraining/models_finished.html')


# function to take the real and bot tweets and create the models for the tweets
def random_forest_tweets(bots, real):

    upscaled = []
    upscaled_data = []

    # upscale the bots so there are the same amount of bots as there are users
    for i in range(0, 5):
        upscaled.extend(bots)

    print(len(upscaled))

    # add upscaled and real bots
    upscaled_data.extend(upscaled)
    upscaled_data.extend(real)
    print(len(upscaled_data))

    tweetdata_x_django = []
    tweetdata_y_django = []

    # for loop to split the data and the classifier variable
    for tweet in upscaled_data:
        tweetdata_x_django.append([tweet.retweet_count, tweet.num_hashtags, tweet.num_urls, tweet.num_mentions])
        tweetdata_y_django.append(tweet.bot)

    tweetdata_y_bool = []

    # for loop to format the boolean as 1's and 0's
    for tweet in tweetdata_y_django:
        if tweet is True:
            tweetdata_y_bool.append(1)
        else:
            tweetdata_y_bool.append(0)

    # change the lists into data frames
    tweetdata_x = np.core.records.fromrecords(tweetdata_x_django, names=['retweet_count', 'num_hashtags', 'num_urls',
                                                                         'num_mentions'])
    tweetdata_y = np.fromiter(tweetdata_y_bool, np.dtype('int_'))
    df = pd.DataFrame(tweetdata_x, columns=['retweet_count', 'num_hashtags', 'num_urls', 'num_mentions'])

    # add the bot and to train columns to specify which ones to split into test and train 75:25
    df['Bot'] = pd.Categorical.from_array(tweetdata_y)
    df['to_train'] = np.random.uniform(0, 1, len(df)) <= .75
    print(df.head())

    # split the data into train and test data
    train, test = df[df['to_train'] == True], df[df['to_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    # get the names of the data columns
    tweet_data_names = df.columns[:4]

    # get the train classifier and the test classifier
    y = train['Bot']
    test_y = test['Bot']

    # initialize, fit and predict the random forest user data
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(train[tweet_data_names], y)
    predict = rf.predict(test[tweet_data_names])
    print('rf done')

    # initialize, fit and predict the Gaussian algorithm
    gnb = GaussianNB()
    gnb.fit(train[tweet_data_names], y)
    predict2 = gnb.predict(test[tweet_data_names])
    print('neural done')

    # get accuracy of the models
    print(accuracy_score(test_y, predict))
    print(accuracy_score(test_y, predict2))

    # create the gaussian and random forest models and save to the disk
    filename = 'random_forest_tweet_model.sav'
    pickle.dump(rf, open(filename, 'wb'))

    return predict


# function to take in the users and create the random forest models to predict the user data
def random_forest(all_users_entries):
    # retrieve the values from the django retrieved objects
    userdata_x_django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userdata_y_django = all_users_entries.values_list('bot', flat=True)

    userdata_y_bool = []

    # get the categorical class field from the user (bot)
    for user in userdata_y_django:
        if user is True:
            userdata_y_bool.append(1)
        else:
            userdata_y_bool.append(0)

    # get the data from the django objects and create the dataframe
    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userdata_y = np.fromiter(userdata_y_bool, np.dtype('int_'))
    df = pd.DataFrame(userdata_x, columns=['Statuses Count', 'Followers_Count', 'Friends Count', 'Favourite Count'])

    print(len(userdata_x))
    print(len(userdata_y))

    # add the classification class and to train column to the data frame for splitting the data into train and test
    df['Bot'] = pd.Categorical.from_array(userdata_y)
    df['to_train'] = np.random.uniform(0, 1, len(df)) <= .75

    print(df.head())

    # split the data in train and test data
    train, test = df[df['to_train'] == True], df[df['to_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    # get the test and train data column names
    user_data_names = df.columns[:4]

    # get the classification class column
    test_y = test['Bot']
    y = train['Bot']

    # initialize, train and predict the random forest model
    rf = RandomForestClassifier(n_jobs=2)
    rf.fit(train[user_data_names], y)
    predict = rf.predict(test[user_data_names])

    count = 0
    true = []

    # for loop to check the accuracy of the prediction
    for i in range(0, len(predict)):
        print(predict[i], '=', test_y.iloc[i])
        true.extend(test_y.iloc[[i]])
        if predict[i] == test_y.iloc[[i]]:
            count += 1

    print(len(predict))
    print((count / len(predict)*100))

    # create the random forest model
    filename = 'random_forest_user_model.sav'
    pickle.dump(rf, open(filename, 'wb'))

    return predict


# function to take in the users and create the support vectors models to predict the user data
def support_vector_machines(all_users_entries):

    userdata_x_django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userdata_y_django = all_users_entries.values_list('bot', flat=True)

    userdata_y_bool = []

    # for loop to change the bot boolean to 1 and 0
    for user in userdata_y_django:
        if user is True:
            userdata_y_bool.append(1)
        else:
            userdata_y_bool.append(0)

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userdata_y = np.fromiter(userdata_y_bool, np.dtype('int_'))

    # split the data into train and test data
    np.random.seed(0)
    indices = np.random.permutation(len(userdata_x))
    userdata_x_train = userdata_x[indices[:-.1*len(userdata_x)]]
    userdata_y_train = userdata_y[indices[:-.1*len(userdata_y)]]
    userdata_x_test = userdata_x[indices[-.1*len(userdata_x):]]
    userdata_y_test = userdata_y[indices[-.1*len(userdata_y):]]

    # reshape the data to use in the algorithms
    userdata_x_train = userdata_x_train.reshape(len(userdata_x_train), 1)
    userdata_x_test = userdata_x_test.reshape(len(userdata_x_test), 1)

    # initialize, train and predict the support vector machine
    svc = svm.SVC(kernel='linear', cache_size=7000)
    svc.fit(userdata_x_train, userdata_y_train)
    predict = svc.predict(userdata_x_test)

    count = 0

    # for loop to check the accuracy of the model
    for i in range(0, len(predict)):
        if predict[i] == userdata_y_test[i]:
            count += 1

    print(predict)
    print(userdata_y_test)
    print(len(userdata_y_test))

    print((count / len(userdata_x_test)))

    return predict


# function to create linear model
def linear_model(all_users_entries):
    userdata_x_django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userdata_y_django = all_users_entries.values_list('bot', flat=True)

    userdata_y_bool = []

    # loop to change the classification variable to 1 and 0
    for user in userdata_y_django:
        if user is True:
            userdata_y_bool.append(1)
        else:
            userdata_y_bool.append(0)

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userdata_y = np.fromiter(userdata_y_bool, np.dtype('int_'))

    # split the data into train and test data
    np.random.seed(0)
    indices = np.random.permutation(len(userdata_x))
    userdata_x_train = userdata_x[indices[:-.1*len(userdata_x)]]
    userdata_y_train = userdata_y[indices[:-.1*len(userdata_y)]]
    userdata_x_test = userdata_x[indices[-.1*len(userdata_x):]]
    userdata_y_test = userdata_y[indices[-.1*len(userdata_y):]]

    # reshape the data to fit the algorithms
    userdata_x_train = userdata_x_train.reshape(len(userdata_x_train), 1)
    userdata_x_test = userdata_x_test.reshape(len(userdata_x_test), 1)

    # initialize, fit and predict the linear regression model
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(userdata_x_train, userdata_y_train)
    predict = np.mean((regr.predict(userdata_x_test.astype(float))-userdata_y_test.astype(float))**2)

    print(regr.score(userdata_x_test.astype(float), userdata_y_test.astype(float)))

    return predict


# function to create the model for nearest neighbors for users
def nearest_neighbor(all_users_entries):
    userdata_x_django = all_users_entries.values_list('statuses_count', 'followers_count', 'friends_count',
                                                      'favourites_count', 'listed_count')
    userdata_y_django = all_users_entries.values_list('bot', flat=True)

    userdata_y_bool = []

    # change the classification class to 1 and 0
    for user in userdata_y_django:
        if user is True:
            userdata_y_bool.append(1)
        else:
            userdata_y_bool.append(0)

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])
    userdata_y = np.fromiter(userdata_y_bool, np.dtype('int_'))

    # split the data into train and test data
    np.random.seed(0)
    indices = np.random.permutation(len(userdata_x))
    userdata_x_train = userdata_x[indices[:-.1*len(userdata_x)]]
    userdata_y_train = userdata_y[indices[:-.1*len(userdata_y)]]
    userdata_x_test = userdata_x[indices[-.1*len(userdata_x):]]
    userdata_y_test = userdata_y[indices[-.1*len(userdata_y):]]

    userdata_x_train = userdata_x_train.reshape(len(userdata_x_train), 1)
    userdata_x_test = userdata_x_test.reshape(len(userdata_x_test), 1)

    # initialize, fit and predict the user data using knn
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


# function to create the sentiment model for the tweets
def sentiment_analyses(all_tweet_entries):

    tweets = all_tweet_entries.values_list('user_id', 'text', 'bot', 'lang', 'id')
    
    # sort the tweets by the keys
    sorted_tweets = sorted(tweets, key=lambda tw: tw[0])

    tweet_id = sorted_tweets[0][0]
    sentiment = [0, 0, 0, 0]
    sentiment_list = []
    count = 0
    print(tweet_id)
    batch_update_string = ''

    # loop to loop through the tweets and create the sentiment data
    for tweet in sorted_tweets:

        # check to see if the user changed to reset the data and add the users sentiment to the list
        if tweet[0] != tweet_id:
            if tweet[2]:
                sentiment[3] = 1
            else:
                sentiment[3] = 0

            # check to see if there is user data to be translated
            if batch_update_string != '':
                # translate the string using a batch update
                print(batch_update_string)
                translated_string = translate_string(batch_update_string)
                print(translated_string)

            # insert the tweet data into the sentiment database
            # insert_sentiment(tweet[0], sentiment)
            sentiment_list.append(sentiment)
            sentiment = [0, 0, 0, 0]
            count = 0
            batch_update_string = ''
            tweet_id = tweet[0]

        # check to see limit the sentiment prediction to 1000 tweets
        if count < 1000:

            # strip the tweet of non ascii characters
            passed_tweet = strip_tweet(tweet[1])

            # if the tweet language is not in english add to a variable to be passed into batch update
            if tweet[3] != 'en':
                batch_update_string += str(tweet[4]) + ':::;:::' + passed_tweet + ';;;:;;;'

            # get the sentiment of the tweet, either positive, neutral or negative
            tweet_sentiment = get_sentiment(tweet[1])

            # create the sentiment data
            if tweet_sentiment == 'positive':
                sentiment[0] += 1
            elif tweet_sentiment == 'neutral':
                sentiment[1] += 1
            elif tweet_sentiment == 'negative':
                sentiment[2] += 1
            count += 1

    # predict the sentiment data
    predict = random_forest_sentiment(sentiment_list)
    predict2 = sentiment_model_training(sentiment_list)
    print(len(predict2))
    return predict


# create the nearest neighbor model for the sentiment list
def sentiment_model_training(sentiment_list):

    # add the sentiment to a dataframe
    sentiment = np.core.records.fromrecords(sentiment_list, names=['positive', 'neutral', 'negative', 'bot'])
    df = pd.DataFrame(sentiment, columns=['positive', 'neutral', 'negative', 'bot'])

    # add a to train column to split the data
    df['to_train'] = np.random.uniform(0, 1, len(df)) <= .80

    print(df.head())

    # split the data into train and test data
    train, test = df[df['to_train'] == True], df[df['to_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    sentiment_names = df.columns[:3]

    # get the classification columns for the train and test data
    y = train['bot']
    test_y = test['bot']

    # initialize, train and predict the sentiment models
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='uniform')
    knn.fit(train[sentiment_names], y)
    predict = knn.predict(test[sentiment_names])

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

    filename = 'sentiment_model.sav'
    pickle.dump(gnb, open(filename, 'wb'))

    return predict


# function to get the sentiment of a test string
def get_sentiment(text):
    tweet_sentiment = TextBlob(text)

    # returns positive, neutral or negative depending on the polarity
    if tweet_sentiment.sentiment.polarity > 0:
        return 'positive'
    elif tweet_sentiment.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


# function to create the random forest sentiment model
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

    # save the random forest model to the disk
    # filename = 'random_forest_sentiment_model.sav'
    # pickle.dump(rf, open(filename, 'wb'))

    print('Random: ', accuracy_score(test_y, predict))

    return predict


# strip the tweets of all non ascii characters, like emojis ect
def strip_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w +:\ / \ / \S +)", " ", tweet).split())


# function to translate the text from any language to english
def translate_string(text_passed):
    # key to yandex api, spare keys down below if the limit is hit
    translate = YandexTranslate('trnsl.1.1.20170322T133238Z.691da702cdb5dbe6.e00850c5473b4c71346d4fcb0f5fff807b8c4f43')#emmet.hanratty
    # translate = YandexTranslate('trnsl.1.1.20170322T162921Z.24bcd1a440ac1ff1.4ae8817644cf4f0773b5134f2440b780af7044b7')#emmethanratty
    # translate = YandexTranslate('trnsl.1.1.20170322T165243Z.cc4baa633b54ee81.0e9596dca723df99fb296e20846bc47b66bb678f')#hanrattyemmet
    # translate = YandexTranslate('trnsl.1.1.20170322T133603Z.e0a9d5b997f7a2be.c687257bea28bf9d7cf7a7fb9eca4185e7a807a5')#hanratty.emmet
    # translate = YandexTranslate('trnsl.1.1.20170322T171859Z.0d7715e4714685b8.06cf2e36ed1bf7f690a849c257ef708f0bc5b9d4')#hanratty.emmet2017
    # translate = YandexTranslate('trnsl.1.1.20170322T173856Z.03bf0f193d8e4190.59e0b3e30b5055a4cb042664261e0d28dcdb0723')#emmet.hanratty2017
    # translate = YandexTranslate('trnsl.1.1.20170322T185516Z.decb48e6428fa93b.3a1dd757f63e0c4b010dc8d3badb2d3838c94fe5') # e.hanratty
    # translate = YandexTranslate('trnsl.1.1.20170322T192725Z.17b6e0c431553d40.92b0884a635191e901f2cb79253239e6688d6f41')#hanratty.e
    # translate = YandexTranslate('trnsl.1.1.20170322T194833Z.866d740e343c356a.3021ce8a6ba2e802ca0543cc8c42310b91ea6533')# em.hanratty
    # translate = YandexTranslate('trnsl.1.1.20170322T200249Z.aeb9bc1f1bdb5093.c1d999f94d59f1216bd2a4f8325e846033113c6f')  # e.hanratty2017
    # translate = YandexTranslate('trnsl.1.1.20170322T200516Z.8a278bab73cd7892.e5cdbb610699b45f69490ba1c04b5bd8d83a0f2d')  # hanratty.emmet2018
    # translate = YandexTranslate('trnsl.1.1.20170322T222615Z.e2e1e4b1b09df158.e4aef95a40a49b42f48f50857a21d15ec46ba5f1') # emmet.hanratty2018
    # translate = YandexTranslate('trnsl.1.1.20170322T224837Z.e0c8b2987ddf8ad7.d29c185d939ad5e9aae7354ccd029accdcf7596c')#ya.hanratty
    # translate = YandexTranslate('trnsl.1.1.20170322T231048Z.3de42387c73276fc.21beab3c4a0c5fc05f6f6871e51aeeffe53a2efd')#hanratty.em

    # translate the text to any language
    translated = translate.translate(text_passed, 'en')

    # batch update splits the translated batch text and updates the database
    batch_update(translated['text'][0])

    return translated['text'][0]


# insert the sentiment data into the database
def insert_sentiment(passed_id, sentiment):
    sent = sentiment_app(id=passed_id, positive=sentiment[0], neutral=sentiment[1], negative=[2], bot=sentiment[3])
    sent.save()


# updates the non english tweets in the database by id
def translated_tweet_update(tweet_id, text):
    t = tweets_app.objects.get(id=tweet_id)

    t.lang = 'en'
    t.text = text
    t.save()


# splits the batch translated text strings and adds the necessary info to the database
def batch_update(text):
    tweets_and_ids = text.split(';;;:;;;')

    # for to loop through the tweets
    for tweets in tweets_and_ids:

        # splits the tweets and ids
        split_tweet = tweets.split(':::;:::')

        # check to make sure the tweet is not empty, happens when the whole tweet was made of non ascii characters
        # not supported by the database and removed by the strip ascii function
        if split_tweet[0] != '' and split_tweet[1] != '':
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


# function to create the time data and create the time model
def time_analyses(all_tweet_entries):
    tweets = all_tweet_entries.values_list('user_id', 'created_at', 'bot')

    timing_array = []
    tweet_timing = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count = 0
    previous_day = ['', '', '']

    sorted_tweets = sorted(tweets, key=lambda tw: tw[0])
    tweet_id = sorted_tweets[0][0]

    # loop to go through the tweets
    for tweet in sorted_tweets:

        # check to see if the user has changed, and if so reset the data and add the new time data to an array
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

        # check to make sure no more then 1000 tweets are used
        if count <= 1000:
            full_date = tweet[1]

            split_date = full_date.split(' ')

            # split the date into usable fields
            month = split_date[1]
            day_num = split_date[2]
            time = split_date[3]
            year = split_date[5]

            # create a date time object of the time
            datetime_object = datetime.datetime.strptime(time, "%H:%M:%S")

            # check when the tweet was made and update the corresponding field in th array
            if datetime.time(0, 0, 0) <= datetime_object.time() <= datetime.time(2, 59, 0):
                tweet_timing[0] += 1
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

            # check to see if the tweet was tweeted on the same day as the previous one
            if month == previous_day[0] and day_num == previous_day[1] and year == previous_day[2]:
                tweet_timing[8] += 1

            previous_day[0] = month
            previous_day[1] = day_num
            previous_day[2] = year
            count += 1

    predict = timing_model_creation(timing_array)
    return predict


# function to create the timing model
def timing_model_creation(timing_array):

    bots = []
    real = []

    scaled = []
    full_scaled = []

    # loop to split the bot and real users
    for user in timing_array:
        if user[9] == 1:
            bots.append(user)
        else:
            real.append(user)

    # loop to upscale the bot users so it is not biased
    for i in range(0, 2):
        scaled.extend(real)

    full_scaled.extend(bots)
    full_scaled.extend(scaled)

    # add the data to a dataframe
    timing = np.core.records.fromrecords(full_scaled, names=['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21',
                                                              '21-24', 'same day', 'bot'])
    df = pd.DataFrame(timing, columns=['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21', '21-24', 'same day',
                                       'bot'])

    # split the data into a training and test set
    df['to_train'] = np.random.uniform(0, 1, len(df)) <= .75
    print(df.head())
    train, test = df[df['to_train'] == True], df[df['to_train'] == False]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    timing_names = df.columns[:8]

    y = train['bot']
    test_y = test['bot']

    # initialize, fit and predict the models
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=10000, max_features=.2, min_samples_leaf=100, oob_score=True,
                                random_state=50)
    rf.fit(train[timing_names], y)
    predict = rf.predict(test[timing_names])

    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1,
                               n_neighbors=5, p=2, weights='uniform')
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

    # save the random forest model to the disk
    filename = 'random_forest_timing_model.sav'
    pickle.dump(rf, open(filename, 'wb'))

    print('rf: ', accuracy_score(test_y, predict))
    print('knn: ', accuracy_score(test_y, predict2))
    print('svm: ', accuracy_score(test_y, predict3))
    print('clf: ', accuracy_score(test_y, predict4))
    print('gnb: ', accuracy_score(test_y, predict5))

    return predict, predict2, predict3, predict4, predict5


