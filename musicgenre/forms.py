from django import forms
from musicgenre.models import uploading

class uploadingForm(forms.ModelForm):
    class Meta:
        model = uploading
        fields = ('upload_music', )

        