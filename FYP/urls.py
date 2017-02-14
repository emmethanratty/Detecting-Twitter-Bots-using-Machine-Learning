from django.conf.urls import include,url
from django.contrib import admin

urlpatterns = [
    url(r'^$', include('webapp.urls')),
    url(r'^admin/', admin.site.urls),
    url(r'^webapp/', include('webapp.urls')),
    url(r'^modeltraining/', include('modeltraining.urls')),
]
