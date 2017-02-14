from django.db import models

# Create your models here.

class users_app(models.Model):
    id = models.IntegerField(default=0, primary_key=True)
    name = models.CharField(max_length=200)
    screen_name = models.CharField(max_length=200)
    statuses_count = models.IntegerField(default=0)
    followers_count = models.IntegerField(default=0)
    friends_count = models.IntegerField(default=0)
    favourites_count = models.IntegerField(default=0)
    listed_count = models.IntegerField(default=0)
    created_at = models.CharField(max_length=200)
    url = models.CharField(max_length=200)
    lang = models.CharField(max_length=200)
    time_zone = models.CharField(max_length=200)
    location = models.CharField(max_length=200)
    default_profile = models.CharField(max_length=200)
    default_profile_image = models.CharField(max_length=200)
    geo_enabled = models.CharField(max_length=200)
    profile_image_url = models.CharField(max_length=200)
    profile_banner_url = models.CharField(max_length=200)
    profile_use_background_image = models.CharField(max_length=200)
    profile_use_background_image_https = models.CharField(max_length=200)
    profile_text_color = models.CharField(max_length=200)
    profile_image_url_https = models.CharField(max_length=200)
    profile_sidebar_border_color = models.CharField(max_length=200)
    profile_background_tile = models.CharField(max_length=200)
    profile_sidebar_fill_color = models.CharField(max_length=200)
    profile_background_image_url = models.CharField(max_length=200)
    profile_background_color = models.CharField(max_length=200)
    profile_link_color = models.CharField(max_length=200)
    utc_offset = models.CharField(max_length=200)
    protected = models.CharField(max_length=200)
    verified = models.CharField(max_length=200)
    description = models.CharField(max_length=200)
    updated = models.DateTimeField(auto_now_add=True)
    dataset = models.CharField(max_length=5)


class tweets_app(models.Model):
    created_at = models.CharField(max_length=200)
    id = models.IntegerField(default=0, primary_key=True)
    text = models.CharField(max_length=200)
    source = models.CharField(max_length=200)
    user_id = models.IntegerField(default=0)
    truncated = models.CharField(max_length=200)
    in_reply_to_status_id = models.IntegerField(default=0)
    in_reply_to_user_id = models.IntegerField(default=0)
    in_reply_to_screen_name = models.CharField(max_length=200)
    retweeted_status_id = models.IntegerField(default=0)
    #geo = models.
    place = models.CharField(max_length=200)
    retweet_count = models.IntegerField(default=0)
    reply_count = models.IntegerField(default=0)
    favorite_count = models.IntegerField(default=0)
    num_hashtags = models.IntegerField(default=0)
    num_urls = models.IntegerField(default=0)
    num_mentions = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)


class links_app(models.Model):
    source_id = models.IntegerField(default=0)
    target_id = models.IntegerField(default=0)

