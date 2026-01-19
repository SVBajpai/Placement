from django.contrib import admin # type: ignore
from .models import History, Register, Company

# Register your models here.
admin.site.register(Register)
admin.site.register(History)
admin.site.register(Company)