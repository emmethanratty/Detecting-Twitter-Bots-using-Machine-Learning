from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader


def index(request):
    return render(request, 'webapp/index.html')


def home(request):
    return HttpResponse("Home Page")
