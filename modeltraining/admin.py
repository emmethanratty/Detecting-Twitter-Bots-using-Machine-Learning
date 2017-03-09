from django.contrib import admin
from .models import users_app, tweets_app, links_app, friends_app, followers_app

admin.site.register(users_app)
admin.site.register(tweets_app)
admin.site.register(links_app)
admin.site.register(friends_app)
admin.site.register(followers_app)



