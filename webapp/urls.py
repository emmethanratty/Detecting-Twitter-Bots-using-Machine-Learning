from django.conf.urls import url

from . import views

app_name = 'webapp'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    #url(r'^index/', views.index, name='index'),
    url(r'^home/', views.home, name='home'),
]
