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
    auth = tweepy.OAuthHandler(consumer_token, consumer_secret)

    try:
        redirect_url = auth.get_authorization_url()
    except tweepy.TweepError:
        print
        'Error! Failed to get request token.'

    request.session['request_token'] = auth .request_token

    return redirect(redirect_url)


def callback(request):
    verifier = request.GET.get('oauth_verifier')

    auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
    token = request.session.get('request_token')
    # request.session.delete('request_token')
    auth.request_token = token

    try:
        auth.get_access_token(verifier)
        api = tweepy.API(auth)

        public_tweets = api.home_timeline()

        string = ""

        for tweets in public_tweets:
            string += tweets.text
            string += "           "

        return HttpResponse(string)
    except tweepy.TweepError:
        return HttpResponse("didn't work")

