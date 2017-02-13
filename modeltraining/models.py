from django.db import models

# Create your models here.

class users_app(models.Model):
    id = models.IntegerField
    name = models.CharField
    screen_name = models.CharField
    statuses_count = models.IntegerField
    followers_count = models.IntegerField
    friends_count = models.IntegerField
    favourites_count = models.IntegerField
    listed_count = models.IntegerField
    created_at = models.CharField
    url = models.CharField
    lang = models.CharField
    time_zone = models.CharField
    location = models.CharField
    default_profile = models.CharField
    default_profile_image = models.CharField
    geo_enabled = models.CharField
    profile_image_url = models.CharField
    profile_banner_url = models.CharField
    profile_use_background_image = models.CharField
    profile_use_background_image_https = models.CharField
    profile_text_color = models.CharField
    profile_image_url_https = models.CharField
    profile_sidebar_border_color = models.CharField
    profile_background_tile = models.CharField
    profile_sidebar_fill_color = models.CharField
    profile_background_image_url = models.CharField
    profile_background_color = models.CharField
    profile_link_color = models.CharField
    utc_offset = models.CharField
    protected = models.CharField
    verified = models.CharField
    description = models.CharField
    updated = models.DateTimeField(auto_now_add=True)
    dataset = models.CharField(max_length=5)

class tweets_app(models.Model):
    created_at = models.CharField
    id = models.IntegerField
    text = models.CharField
    source = models.CharField
    user_id = models.IntegerField
    truncated = models.CharField
    in_reply_to_status_id = models.IntegerField
    in_reply_to_user_id = models.IntegerField
    in_reply_to_screen_name = models.CharField
    retweeted_status_id = models.IntegerField
    #geo = models.
    place = models.CharField
    retweet_count = models.IntegerField
    reply_count = models.IntegerField
    favorite_count = models.IntegerField
    num_hashtags = models.IntegerField
    num_urls = models.IntegerField
    num_mentions = models.IntegerField
    timestamp = models.DateTimeField(auto_now_add=True)

class links_app(models.Model):
    source_id = models.IntegerField
    target_if = models.IntegerField

