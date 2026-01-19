from re import T
from django.db import models # type: ignore
from django.contrib.auth.models import User # type: ignore

# Create your models here.
class Register(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    address = models.CharField(max_length=200, default='')
    mobile = models.CharField(max_length=15, unique=True)
    image = models.ImageField(upload_to='profile_images/', blank=True, null=True)

    def __str__(self):
        return self.user.username

class History(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    max_prob = models.CharField(max_length=230, null=True, blank=True)
    prediction = models.CharField(max_length=230, null=True, blank=True)
    input_data = models.TextField(null=True, blank=True)
    output_data = models.TextField(null=True, blank=True)
    skill_improvement_tips = models.TextField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.user.username    
    
class Company(models.Model):
    company_name = models.CharField(max_length=230, null=True, blank=True)
    location = models.CharField(max_length=230, null=True, blank=True)
    average_salary = models.CharField(max_length=230, null=True, blank=True)
    job_role = models.CharField(max_length=230, null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.company_name 
    

class ContactMessage(models.Model):
    name = models.CharField(max_length=150)
    email = models.EmailField()
    subject = models.CharField(max_length=200)
    message = models.TextField()
    status = models.CharField(max_length=10, default='unread')  # unread/read
    sent_on = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.subject}"
