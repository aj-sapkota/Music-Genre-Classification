# Generated by Django 4.0.2 on 2022-03-06 12:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('musicgenre', '0003_uploading_delete_upload'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploading',
            name='upload_music',
            field=models.FileField(upload_to='music/'),
        ),
    ]
