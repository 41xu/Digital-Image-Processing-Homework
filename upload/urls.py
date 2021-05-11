from django.urls import path, re_path, include
from . import views
from django.conf.urls import url

urlpatterns = [
    path('', views.upload, name='upload'),
    path(r'^staic/.*)', views.upload)
]

