# -*- coding: utf-8 -*-
from django.conf.urls import url

from . import views
urlpatterns = [
    url(r'^$', views.search, name='search'),
    #url(r'^(?P<usid>.*)/$', views.case_text, name='case_text'),
]
