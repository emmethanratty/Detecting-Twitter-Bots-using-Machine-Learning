from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse


def index(request):
    return HttpResponse("Web App")

def home(request):
    return HttpResponse("Home Page")