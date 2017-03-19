from django.shortcuts import render, redirect
from webapp.models import *
from django.http import HttpResponse
import pickle
import numpy as np
import tweepy
import pandas as pd

consumer_token = "HMif7XOaMbrK8iBnZlYDwtnPa"
consumer_secret = "jZR8th1C8Hj2YoLDVNnbMalpDUsEsOEzcDjSIhW70UF1FQ4mhf"


def index(request):
    return render(request, 'webapp/index.html')


def home(request):
    return HttpResponse("Home Page")


def auth(request):

    if request.method == "POST":
        handle = request.POST["TwitterHandle"]
        request.session["TwitterHandle"] = handle

    auth = tweepy.OAuthHandler(consumer_token, consumer_secret)

    try:
        redirect_url = auth.get_authorization_url()
    except tweepy.TweepError:
        print
        'Error! Failed to get request token.'

    request.session['request_token'] = auth .request_token

    return redirect(redirect_url)


def callback(request):

    if 'TwitterHandle' in request.session:
        handle = request.session['TwitterHandle']

        #print(rf_user_model)

        verifier = request.GET.get('oauth_verifier')

        auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
        token = request.session.get('request_token')
        # request.session.delete('request_token')
        auth.request_token = token

        try:
            auth.get_access_token(verifier)
            api = tweepy.API(auth)

            user = api.get_user(handle)

            user_id = insert_user_data(user)

            balh = rf_user_prediction(user_id)

            # tweets = []
            #
            # tweets_200 = api.user_timeline(screen_name=handle, count=200)
            #
            # print(len(tweets_200))
            #
            # tweets.extend(tweets_200)
            #
            # realign = tweets[-1].id - 1
            #
            # for i in range(0, 4):
            #     tweets_200 = api.user_timeline(screen_name=handle, count=200, max_id=realign)
            #
            #     tweets.extend(tweets_200)
            #     realign = tweets[-1].id - 1
            #
            # tweets_200 = api.user_timeline(screen_name=handle, count=1, max_id=realign)
            #
            # tweets.extend(tweets_200)
            # print(len(tweets))

            #return render(request, "webapp/prediction.html")
            return HttpResponse(balh)
        except tweepy.TweepError:
            return HttpResponse("didn't work")
    else:
        return "didn't work cause"


def insert_user_data(user_data):
    user = users_app(id=user_data.id, name=user_data.name, screen_name=user_data.screen_name,
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
    return user.id


def rf_user_prediction(user_id):
    rf_user_filename = 'random_forest_user_model.sav'
    rf_user_model = pickle.load(open(rf_user_filename, 'rb'))

    user = users_app.objects.all().filter(id__contains=user_id)

    userdata_x_django = user.values_list('statuses_count', 'followers_count', 'friends_count', 'favourites_count')

    userdata_x = np.core.records.fromrecords(userdata_x_django, names=['Statuses Count', 'Followers_Count',
                                                                       'Friends Count', 'Favourite Count'])

    df = pd.DataFrame(userdata_x, columns=['Statuses Count', 'Followers_Count', 'Friends Count', 'Favourite Count'])

    predict = rf_user_model.predict(df)

    print(df)
    print(predict)

    return rf_user_model.predict_proba(df)
