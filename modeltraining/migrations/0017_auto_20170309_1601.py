# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-03-09 16:01
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('modeltraining', '0016_auto_20170309_1549'),
    ]

    operations = [
        migrations.AlterField(
            model_name='followers_app',
            name='bot',
            field=models.NullBooleanField(default=None),
        ),
        migrations.AlterField(
            model_name='friends_app',
            name='bot',
            field=models.NullBooleanField(default=None),
        ),
        migrations.AlterField(
            model_name='tweets_app',
            name='bot',
            field=models.NullBooleanField(default=None),
        ),
    ]
