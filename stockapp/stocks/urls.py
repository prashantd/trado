from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('positions/', views.positions, name='positions'),
    path('api/stock-chart/<str:symbol>/', views.get_stock_chart, name='stock_chart'),
    path('api/stock-news/<str:symbol>/', views.get_stock_news, name='stock_news'),
    path('api/stock-list/', views.get_stock_list, name='stock_list'),
    path('kite/login/', views.kite_login, name='kite_login'),
    path('kite/callback/', views.kite_callback, name='kite_callback'),
]