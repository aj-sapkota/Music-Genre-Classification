from operator import index
from tkinter import CENTER
from django.shortcuts import render, redirect
from musicgenre.barchart import get_dataframe

from musicgenre.forms import uploadingForm
from django.contrib import messages
import numpy as np

from musicgenre.models import uploading
from .predict import load_data
import json
import matplotlib.pyplot as plt
from django.core.files.storage import FileSystemStorage


from .getMfcc import getmfcc
from .modelProcess import predictingOutput

JSON_PATH = "final.json"
DATA_PATH = "./final.json"


def uploadfile(request):
    form = uploadingForm()
    try:
        if request.method == 'POST':
            form = uploadingForm(request.POST, request.FILES)

            try:
                myfile = request.FILES['upload_music']
            except:
                messages.error(
                    request, 'Sorry, the field is empty. Please upload music file with extension .wav ')
                # print("error")
                return render(request, 'musicgenre/home.html')

            myfile = request.FILES['upload_music']
            fs = FileSystemStorage()
            print(fs)
            print(myfile)
            filename = fs.save(myfile.name, myfile)
            print(filename)
            # print(filename.url)
            uploaded_file_url = fs.url(filename)
            print(uploaded_file_url)
            if not myfile.name.endswith('.wav'):
                messages.error(
                    request, 'Sorry, Please upload music file with extension .wav ')
                print("error")
                return render(request, 'musicgenre/home.html')
            ok = form.save(commit=False)
            ok.save(myfile)
            #uploaded_file_url = ok.url(myfile)
            getmfcc(myfile, JSON_PATH)
            final_mfcc = load_data(DATA_PATH)
            # print(final_mfcc)
            # print(final_mfcc.shape)  # changed array shape

            genreLabelling, tempAverage = predictingOutput(final_mfcc)

            df = get_dataframe(tempAverage)

            messages.text = "Your predicted genre is : "
            messages.text = messages.text + genreLabelling
            messages.success(request, messages.text)
            #messages.success(request, 'Uploaded music file')

            df = df.to_html(justify='justify-all', index=False)
            df = df.replace('class="dataframe"',
                            'class="table table-dark table-hover"')
            #df = df.to_html().replace('<td>', '<td style="text-align: left">').replace('<th>', '<th style="text-align: left">')
            context = {
                'df': df,
                'form': form,
                'uploaded_file_url': uploaded_file_url,
                'filename': filename,
            }
            return render(request, 'musicgenre/home.html', context)
    except:
        messages.error(request, 'Oops!,  Something went wrong')
        return render(request, 'musicgenre/home.html')

    form = uploadingForm()
    return render(request, 'musicgenre/home.html', {'form': form})
