# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-03-09 15:29
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('modeltraining', '0013_auto_20170309_1522'),
    ]

    operations = [
        migrations.CreateModel(
            name='followers_app',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source_id', models.IntegerField(default=0)),
                ('target_id', models.IntegerField(default=0)),
                ('bot', models.BooleanField(default=None)),
            ],
        ),
        migrations.CreateModel(
            name='friends_app',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source_id', models.IntegerField(default=0)),
                ('target_id', models.IntegerField(default=0)),
                ('bot', models.BooleanField(default=None)),
            ],
        ),
        migrations.AddField(
            model_name='links_app',
            name='bot',
            field=models.BooleanField(default=None),
        ),
        migrations.AddField(
            model_name='tweets_app',
            name='bot',
            field=models.BooleanField(default=None),
        ),
    ]