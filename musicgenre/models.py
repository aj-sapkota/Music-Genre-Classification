
from __future__ import unicode_literals

from django.db import models
# Create your models here.
class uploading(models.Model):
    description = models.CharField(max_length=255, blank=True)
    upload_music = models.FileField(upload_to= 'music/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    objects = models.Manager()

