from django.shortcuts import render, redirect
from textblob import TextBlob
from webapp.models import *
import pickle
import numpy as np
import tweepy
import pandas as pd
import datetime

# consumer key and secret to access the twitter api
consumer_token = "zwGPR1XN0wtllYEdbZyhKwJzX"
consumer_secret = "Ob5zlQFPl0eUUHoqj9ZUrYxWM9dFgrxE3Oz17ljqtvV4TuQwMR"


# function to go onto the home page
def index(request):
    return render(request, 'webapp/index.html')


# function to authenticate the request to access the twitter api
def auth(request):

    # checks to make sure that the method of the request has been a post method and gets the twitter handle of user
    if request.method == "POST":
        handle = request.POST["TwitterHandle"]
        request.session["TwitterHandle"] = handle

    # creating the oauth handler and specifying the callback request
    oauth = tweepy.OAuthHandler(consumer_token, consumer_secret, 'http://localhost/webapp/callback')

    # try except block to validate the redirect api for the oauth handler
    try:
        redirect_url = oauth.get_authorization_url()

        request.session['request_token'] = oauth.request_token

        return redirect(redirect_url)

    # except in the case of auth no validated
    except tweepy.TweepError:
        context = {
            'problem': 'Verifier for twitter error'
        }
        print('Error! Failed to get request token.')
        return render(request, 'webapp/error.html', context)


# function for authenticating the request to get a users twitter followers with access to the api
def auth_followers(request):

    # checks to make sure that the method of the request has been a post method and gets the twitter handle of user
    if request.method == "POST":
        handle = request.POST["TwitterHandle_f"]
        request.session["TwitterHandle_f"] = handle

    # creating the oauth handler and specifying the callback request
    oauth = tweepy.OAuthHandler(consumer_token, consumer_secret, 'http://localhost/webapp/followers_callback')#http://192.168.1.12/webapp/followers_callback

    # try except block to validate the redirect api for the oauth handler
    try:
        redirect_url = oauth.get_authorization_url()

        request.session['request_token'] = oauth.request_token

        return redirect(redirect_url)

    # except in the case of auth no validated
    except tweepy.TweepError:
        context = {
            'problem': 'Verifier for twitter error'
        }
        print('Error! Failed to get request token.')
        return render(request, 'webapp/error.html', context)


# function to accept the callback from the oauth authentication for checking a user
def callback(request):

    # check to make sure the handle has been successfully stored in the session
    if 'TwitterHandle' in request.session:
        # getting the oauth and handle
        handle = request.session['TwitterHandle']
        verifier = request.GET.get('oauth_verifier')

        auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
        token = request.session.get('request_token')
        # request.session.delete('request_token')
        auth.request_token = token

        # try except block to validate the oauth
        try:
            auth.get_access_token(verifier)
            api = tweepy.API(auth)

            user = api.get_user(handle)

            user_id, lang = insert_user_data(user)

            tweets = []

            tweets_200 = api.user_timeline(screen_name=handle, count=200)

            # check to make sure that some tweets have been successfully returned from the api
            if tweets_200:

                print(len(tweets_200))

                tweets.extend(tweets_200)

                realign = tweets[-1].id - 1

                # for loop to get 1000 of the most recent tweets from the user
                for i in range(0, 4):
                    tweets_200 = api.user_timeline(screen_name=handle, count=200, max_id=realign)

                    tweets.extend(tweets_200)
                    realign = tweets[-1].id - 1

                tweets_200 = api.user_timeline(screen_name=handle, count=1, max_id=realign)

                tweets.extend(tweets_200)
                print(len(tweets))

                insert_tweet_data(tweets, lang, user_id)

            # calling the prediction class to check the user
            prediction = rf_user_prediction(user_id, handle)

            # this is the context to return variables to the html UI
            context = {
                'prediction': prediction,
                'handle': handle
            }

            # renders the web page template for user prediction
            return render(request, 'webapp/prediction.html', context)
        except tweepy.TweepError:
            context = {
                'problem': 'Verifier for twitter error'
            }
            return render(request, 'webapp/error.html', context)

    # check if there is no handle in the session
    else:
        context = {
            'problem': 'Twitter Handle not passed in'
        }
        return render(request, 'webapp/error.html', context)


# function to insert the user data into the web app database so that the prediction can access the data
def insert_user_data(user_data):
    user = users_app(id=user_data.id, name=strip_non_ascii(user_data.name), screen_name=user_data.screen_name,
                     statuses_count=user_data.statuses_count, followers_count=user_data.followers_count,
                     friends_count=user_data.friends_count, favourites_count=user_data.favourites_count,
                     listed_count=user_data.listed_count, created_at=user_data.created_at, url=user_data.url,
                     lang=user_data.lang, time_zone=user_data.time_zone, location=user_data.location,
                     default_profile=user_data.default_profile, default_profile_image=user_data.default_profile_image,
                     geo_enabled=user_data.geo_enabled, profile_image_url=user_data.profile_image_url,
                     profile_use_background_image=user_data.profile_use_background_image,
                     profile_background_image_url_https=user_data.profile_background_image_url_https,
                     profile_text_color=user_data.profile_text_color, profile_image_url_https=user_data.profile_image_url_https,
                     profile_sidebar_border_color=user_data.profile_sidebar_border_color, profile_background_tile=user_data.profile_background_tile,
                     profile_sidebar_fill_color=user_data.profile_sidebar_fill_color, profile_background_image_url=user_data.profile_background_image_url,
                     profile_background_color=user_data.profile_background_color, profile_link_color=user_data.profile_link_color,
                     utc_offset=user_data.utc_offset, protected=user_data.protected, verified=user_data.verified,
                     description=user_data.description)

    user.save()
    return user.id, user.lang


# function to insert users followers data into the web app database so the prediction can access the data
def insert_followers_data(main_user_id, user_data):
    user = followers_app(following_id=main_user_id, id=user_data.id, name=strip_non_ascii(user_data.name),
                         screen_name=user_data.screen_name, statuses_count=user_data.statuses_count,
                         followers_count=user_data.followers_count, friends_count=user_data.friends_count,
                         favourites_count=user_data.favourites_count)

    user.save()


# function to insert the users tweet data into th web app database so the prediction can access the data
def insert_tweet_data(tweets, lang, user_id):
    for tweet in tweets:
        utf = strip_non_ascii(tweet.text)

        tweet_data = tweets_app(created_at=tweet.created_at, id=tweet.id, text=utf, source=tweet.source,
                                user_id=user_id, truncated=tweet.truncated, in_reply_to_status_id=tweet.in_reply_to_status_id,
                                in_reply_to_user_id=tweet.in_reply_to_user_id, in_reply_to_screen_name=tweet.in_reply_to_screen_name,
                                retweet_count=tweet.retweet_count,
                                favorite_count=tweet.favorite_count,
                                num_hashtags=len(tweet.entities['hashtags']), num_urls=len(tweet.entities['urls']),
                                num_mentions=len(tweet.entities['user_mentions']), lang=lang)
        tweet_data.save()


# function that takes in the user id and handle as passed arguments
# the function then loads the prediction models and performs the necessary prediction on the data
def rf_user_prediction(user_id, handle):
    rf_user_filename = 'random_forest_user_model.sav'
    rf_sentiment_filename = 'random_forest_sentiment_model.sav'
    rf_timing_filename = 'random_forest_timing_model.sav'
    rf_user_model = pickle.load(open(rf_user_filename, 'rb'))
    rf_sentiment_model = pickle.load(open(rf_sentiment_filename, 'rb'))
    rf_timing_model = pickle.load(open(rf_timing_filename, 'rb'))

    # returns all the data from the web app database about the user
    user = users_app.objects.all().filter(id__contains=user_id)
    tweets = tweets_app.objects.all().filter(user_id__contains=user_id)

    # call to sentiment and time analyses to perform the predictions
    sentiment = sentiment_analyses(tweets)
    timing = timing_analyses(tweets)

    # preparing the user data to be used in the prediction
    userdata_x_django = user.values_list('statuses_count', 'followers_count', 'friends_count', 'favourites_count')

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])

    df = pd.DataFrame(userdata_x, columns=['Statuses Count', 'Followers_Count', 'Friends Count', 'Favourite Count'])

    # performing the predictions on the data
    predict_sentiment = rf_sentiment_model.predict_proba(sentiment)
    predict_user = rf_user_model.predict_proba(df)
    predict_timing = rf_timing_model.predict_proba(timing)
    tweet_predict_percentage = tweet_analyses(tweets)

    print('Tweet Percentage: ', tweet_predict_percentage)
    print('User Predict: ', predict_user)
    print('Sentiment Predict: ', predict_sentiment)
    print('Timing Predict: ', predict_timing)

    user_percentage = predict_user[0][1] * 100
    timing_percentage = predict_timing[0][1] * 100
    sentiment_percentage = predict_sentiment[0][1] * 100

    print(user_percentage, timing_percentage, sentiment_percentage, tweet_predict_percentage)

    # this is a weighted overall prediction based on the accuracy of the models
    overall_prediction = ((user_percentage * 60) + (tweet_predict_percentage * 20) + (timing_percentage * 10) +
                          (sentiment_percentage * 10)) / 100

    print('Overall Percentage: ', overall_prediction)

    # returning the prediction to show on the UI
    return int(round(overall_prediction))


# function to strip non ascii characters from the tweets, needed for mysql database UTC-8 to work
def strip_non_ascii(passed_string):
    # Returns the string without non ASCII characters
    stripped = (c for c in passed_string if 0 < ord(c) < 127)
    return ''.join(stripped)


# function to check the sentiment of the tweet and create the data to use in the prediction model
def sentiment_analyses(all_tweet_entries):
    tweets = all_tweet_entries.values_list('user_id', 'text', 'lang', 'id')

    sentiment = [0, 0, 0]
    batch_update = ''

    # looping through tweets
    for tweet in tweets:
        # strips non ascii like emoji's or special characters
        passed_tweet = strip_non_ascii(tweet[1])

        # if the language is not english it converts the tweet to english
        if tweet[2] != 'en':
            batch_update += str(tweet[3]) + ':::;:::' + passed_tweet + ';;;:;;;'

        # call to get the tweets sentiment, either positive, negative or neutral
        tweet_sentiment = get_sentiment(tweet[1])

        # creating data depending on sentiment
        if tweet_sentiment == 'positive':
            sentiment[0] += 1
        elif tweet_sentiment == 'neutral':
            sentiment[1] += 1
        elif tweet_sentiment == 'negative':
            sentiment[2] += 1

    print(sentiment)
    return sentiment


# function to check the polarity of the text, and returning positive, neutral or negative
def get_sentiment(text):
    tweet_sentiment = TextBlob(text)

    if tweet_sentiment.sentiment.polarity > 0:
        return 'positive'
    elif tweet_sentiment.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


# function to check perform a prediction on each tweet from the user
# based on the retweet count, number of hashtags, numbers of url's and the the number of mentions
def tweet_analyses(all_tweet_entries):

    # loading the tweet prediction model
    rf_tweet_filename = 'random_forest_tweet_model.sav'
    rf_tweets_model = pickle.load(open(rf_tweet_filename, 'rb'))

    tweets = all_tweet_entries.values_list('retweet_count', 'num_hashtags', 'num_urls', 'num_mentions')

    predict = rf_tweets_model.predict(tweets)

    count = 0

    # loop to count the number of bot tweets predicted
    for bot in predict:
        if bot == 1:
            count += 1

    # returning the percentage of tweets that are predicted as bots
    percentage = count/len(predict)
    return percentage


# function to generate the time analyses data for use in the timing prediction
def timing_analyses(all_tweet_entries):
    tweets = all_tweet_entries.values_list('user_id', 'created_at', 'bot')

    tweet_timing = [0, 0, 0, 0, 0, 0, 0, 0]

    # for loop to split the time of tweet into data usable for the timing prediction
    for tweet in tweets:
        full_date = tweet[1]

        split_date = full_date.split(' ')

        date = split_date[0]
        time = split_date[1]

        datetime_object = datetime.datetime.strptime(time, "%H:%M:%S")

        # checks the time the tweet was tweeted at
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

    # returns the tweet timing data
    print(tweet_timing)
    return tweet_timing


# function to render the html template for prediction of users followers
def followers(request):
    return render(request, 'webapp/followers.html')


# function to authenticate the request to access the twitter api
def followers_callback(request):

    # checks to make sure that the method of the request has been a post method and gets the twitter handle of user
    if 'TwitterHandle_f' in request.session:
        handle = request.session['TwitterHandle_f']

        verifier = request.GET.get('oauth_verifier')

        auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
        token = request.session.get('request_token')
        # request.session.delete('request_token')
        auth.request_token = token

        # try except block to verify the oauth for the twitter api
        try:
            auth.get_access_token(verifier)

            api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

            main_user = api.get_user(handle)
            main_id = main_user.id

            ids = []
            prediction = []

            for follow_id in tweepy.Cursor(api.followers_ids, screen_name=handle).items(100):
                ids.append(follow_id)
                print(follow_id)

            print(len(ids))

            for user_id in ids:
                user = api.get_user(user_id)

                insert_followers_data(main_id, user)

                print(strip_non_ascii(user.name))

                prediction.extend(rf_follower_prediction(user_id))

            context = {
                'handle': handle
            }

            return render(request, 'webapp/follow_prediction.html', context)

        except tweepy.TweepError:
            context = {
                'problem': 'Verifier for twitter error'
            }
            return render(request, 'webapp/error.html', context)

    else:
        context = {
            'problem': 'Twitter handle not passed in properly'
        }
        return render(request, 'webapp/error.html', context)


def rf_follower_prediction(user_id):
    rf_user_filename = 'random_forest_user_model.sav'
    rf_user_model = pickle.load(open(rf_user_filename, 'rb'))

    user = followers_app.objects.all().filter(id__contains=user_id)

    userdata_x_django = user.values_list('statuses_count', 'followers_count', 'friends_count', 'favourites_count')

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])

    df = pd.DataFrame(userdata_x, columns=['Statuses Count', 'Followers_Count', 'Friends Count', 'Favourite Count'])

    predict_user = rf_user_model.predict(df)

    print('User Predict: ', predict_user)

    return predict_user

