from django.conf.urls import url

from . import views

app_name = 'webapp'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^home/', views.home, name='home'),
    url(r'^auth/', views.auth, name='auth'),
    url(r'^callback/', views.callback, name='callback'),
    # url(r'^oauth/authorize/$', views.authorize, name='oauth_authorize'),
    # url(r'^oauth/callback/$', views.callback, name='oauth_callback'),
]
