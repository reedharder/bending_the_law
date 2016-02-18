from django.shortcuts import render

#simple home page
def index(request):
     return render(request, 'home/index.html' )
# Create your views here.
