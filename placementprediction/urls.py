"""
URL configuration for placementprediction project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin # type: ignore
from django.urls import path # type: ignore
from django.conf import settings # type: ignore
from django.conf.urls.static import static # type: ignore
from placementapp.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name="home"),
    path('about/', about, name="about"),
    path('contact/', contact, name="contact"),
    path('adminlogin/', adminlogin, name="adminlogin"),
    path('signin/', signin, name="signin"),
    path('signup/', signup, name="signup"),
    path('logout-user/', logout_user, name="logout_user"),
    path('change-password/', change_password, name="change_password"),
    path('update-profile/', update_profile, name="update_profile"),

    path('prediction/', prediction, name="prediction"),
    path('my-history/', my_history, name="my_history"),
    path('prediction-detail/<int:pid>/', prediction_detail, name="prediction_detail"),
    path('all-user/', all_user, name="all_user"),
    path('delete-history/<int:pid>/', delete_history, name='delete_history'),
    path('delete_user/<int:pid>/', delete_user, name='delete_user'),
    path('data-visualization/', data_visulisation, name='data-visualization'),
    path('messages/', admin_messages_view, name='admin_messages'),
    path('message/<int:id>/', message_detail_view, name='message_detail'),


    
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
