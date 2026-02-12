from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/stock-chart/<str:symbol>/', views.get_stock_chart, name='stock_chart'),
    path('api/stock-news/<str:symbol>/', views.get_stock_news, name='stock_news'),
]