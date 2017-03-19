from django.shortcuts import render, redirect

# Create your views here.
from django.http import HttpResponse

import tweepy

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

        verifier = request.GET.get('oauth_verifier')

        auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
        token = request.session.get('request_token')
        # request.session.delete('request_token')
        auth.request_token = token

        try:
            auth.get_access_token(verifier)
            api = tweepy.API(auth)

            user = api.get_user(handle)

            tweets = []

            tweets_200 = api.user_timeline(screen_name=handle, count=200)

            print(len(tweets_200))

            tweets.extend(tweets_200)

            realign = tweets[-1].id - 1

            for i in range(0, 4):
                tweets_200 = api.user_timeline(screen_name=handle, count=200, max_id=realign)

                tweets.extend(tweets_200)
                realign = tweets[-1].id - 1

            tweets_200 = api.user_timeline(screen_name=handle, count=1, max_id=realign)

            tweets.extend(tweets_200)
            print(len(tweets))

            #return render(request, "webapp/prediction.html")
            return HttpResponse(user, tweets)
        except tweepy.TweepError:
            return HttpResponse("didn't work")
    else:
        return "didn't work cause"