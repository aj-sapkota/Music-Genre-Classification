# Generated by Django 4.0.2 on 2022-03-06 11:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('musicgenre', '0002_rename_document_upload_upload_music'),
    ]

    operations = [
        migrations.CreateModel(
            name='uploading',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(blank=True, max_length=255)),
                ('upload_music', models.FileField(upload_to='audio/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.DeleteModel(
            name='upload',
        ),
    ]