import json
from django.contrib import messages # type: ignore
from django.shortcuts import render , redirect , get_object_or_404 # type: ignore
from django.contrib.auth import authenticate, login, logout # type: ignore
from django.contrib.auth.decorators import login_required # type: ignore
from .models import *
from .predict import plot_degree_boxplot, plot_feature_graph, plot_heatmap, plot_probability, plot_ssc_mba, predict_placement, train_dataset 
from django.http import JsonResponse 
# Create your views here.
def home(request):
    return render(request, "home.html", locals())

def about(request):
    return render(request, "about.html", locals())

def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        # Save to database
        ContactMessage.objects.create(
            name=name,
            email=email,
            subject=subject,
            message=message,
            status='unread'
        )
        return render(request, 'contact.html', {'success': True})

    return render(request, 'contact.html')




def adminlogin(request):
    if request.method == "POST":
        re = request.POST
        user = authenticate(username=re['username'], password=re['password'])
        if user.is_staff:
            login(request, user)
            messages.success(request, "Logged in successful")
            return redirect('home')
        else:
            messages.error(request, "Invalid Credential")
            return redirect('adminlogin')
    return render(request, "adminlogin.html", locals())

def signin(request):
    if request.method == "POST":
        re = request.POST
        user = authenticate(username=re['username'], password=re['password'])
        if user:
            login(request, user)
            messages.success(request, "Logged in successful")
            return redirect('home')
        else:
            messages.error(request, "Invalid Credential")
            return redirect('signin')
    return render(request, "signin.html", locals())

def signup(request):
    if request.method == "POST":
        re = request.POST
        rf = request.FILES
        user = User.objects.create_user(username=re['username'], first_name=re['first_name'], last_name=re['last_name'], password=re['password'])
        register = Register.objects.create(user=user, address=re['address'], mobile=re['mobile'], image=rf['image'])
        response_data = {
            'success': True,
            'message': 'Registration successful!',
            'redirect_url': '/signin/'
        }
        return JsonResponse(response_data)
    return render(request, "signup.html", locals())

def logout_user(request):
    logout(request)
    messages.success(request, "Logout Successfully")
    return redirect('home')

def change_password(request):
    if request.method == "POST":
        re = request.POST
        user = authenticate(username=request.user.username, password=re['old-password'])
        if user:
            if re['new-password'] == re['confirm-password']:
                user.set_password(re['confirm-password'])
                user.save()
                messages.success(request, "Password changed successfully")
                return redirect('home')
            else:
                messages.success(request, "Password mismatch")
        else:
            messages.success(request, "Wrong password")
    return render(request, "change_password.html", locals())
    

@login_required
def prediction(request):
    is_output = False
    if request.method == 'POST':
        # Extract form data
        gender = request.POST.get('gender')
        tenthmarks = request.POST.get('tenthmarks')
        ssc_b = request.POST.get('ssc_b')
        twelfthmarks = request.POST.get('twelfthmarks')
        hsc_b = request.POST.get('hsc_b')
        hsc_s = request.POST.get('hsc_s')
        cgpa = request.POST.get('cgpa')
        degree_t = request.POST.get('degree_t')
        workex = request.POST.get('work_experience')
        etest_p = request.POST.get('etest_p')
        # branch = request.POST.get('branch')
        # mba_p = request.POST.get('mba_p')
        backlogs = request.POST.get('backlogs')
        certification = request.POST.get('certification')
        domain = request.POST.get('domain')
        skills = request.POST.get('skills')
        preferredcompany = request.POST.get('preferredcompany')

        # Validate required fields
        required_fields = {
            'Gender': gender,
            '10th Marks': tenthmarks,
            '10th Board': ssc_b,
            '12th Marks': twelfthmarks,
            '12th Board': hsc_b,
            '12th Stream': hsc_s,
            'CGPA': cgpa,
            'Degree Type': degree_t,
            'Work Experience': workex,
            # 'Specialisation': branch,
            'Backlogs': backlogs
        }
        for field_name, field_value in required_fields.items():
            if not field_value:
                messages.error(request, f"Please fill in the {field_name} field.")
                return render(request, 'prediction.html')

        try:
            # Convert numeric fields
            tenthmarks = float(tenthmarks)
            twelfthmarks = float(twelfthmarks)
            cgpa = float(cgpa)
            etest_p = float(etest_p)
            # mba_p = float(mba_p)
            backlogs = int(backlogs) if backlogs else 0

            # Prepare input data for prediction
            # input_data = {
            #     'gender': gender,
            #     'tenthmarks': tenthmarks,
            #     'ssc_b': ssc_b,
            #     'twelfthmarks': twelfthmarks,
            #     'hsc_b': hsc_b,
            #     'hsc_s': hsc_s,
            #     'cgpa': cgpa,
            #     'degree_t': degree_t,
            #     'internship': internship,
            #     'etest_p': etest_p,
            #     'branch': branch,
            #     'mba_p': mba_p,
            #     'backlogs': backlogs,
            #     'certification': certification,
            #     'domain': domain,
            #     'skills': skills,
            #     'preferredcompany': preferredcompany
            # }

            input_data = {
                'gender': gender,
                'ssc_p': tenthmarks,
                'ssc_b': ssc_b,
                'hsc_p': twelfthmarks,
                'hsc_b': hsc_b,
                'hsc_s': hsc_s,
                'degree_p': float(cgpa) * 100,
                'degree_t': degree_t,
                'workex': workex,
                'etest_p': etest_p,
                'backlogs': backlogs,
                # 'specialisation': branch,
                'certification': certification,
                'domain': domain,
                'skills': skills,
                'preferredcompany': preferredcompany,
                # 'mba_p': mba_p
            }

            try:
                train_dataset()
                status, max_prob = predict_placement(input_data)
                print(f"\nPrediction for example input:")
                prediction_output = status
                print(f"Placement Status: {status}")
                print(f"Probability of Placement: {max_prob:.2%}")
            except ValueError as e:
                print(f"Error: {e}")
            # Call prediction function
            # prediction, max_prob, skill_improvement_tips, visualizations = predict_placement(input_data)
            skill_improvement_tips = ""
            visualizations = ""
            # Prepare data for History model
            input_data_str = json.dumps(input_data)
            output_data = {
                'prediction': prediction_output,
                'max_prob': float(max_prob) * 100,
                'skill_improvement_tips': skill_improvement_tips,
                'visualizations': visualizations
            }
            output_data_str = json.dumps(output_data)
            skill_improvement_tips = "Practice daily, embrace mistakes, seek feedback, stay consistent."

            # Save to History model
            history = History(
                user=request.user,
                max_prob= float(max_prob) * 100,
                prediction=prediction_output,
                input_data=input_data_str,
                output_data=output_data_str,
                skill_improvement_tips =skill_improvement_tips
            )
            history.save()

            # Pass results to template
            company_record =  Company.objects.filter()[:5]
            context = {
                'prediction': prediction_output,
                'max_prob': float(max_prob) * 100,  # Convert to percentage
                'skill_improvement_tips': skill_improvement_tips,
                'visualizations': visualizations,
                'is_output': True,
                'company_record':company_record
            }
            return render(request, 'prediction.html', context)

        except ValueError as e:
            messages.error(request, f"Invalid input: {str(e)}")
            return render(request, 'prediction.html')
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return render(request, 'prediction.html')

    return render(request, 'prediction.html')

def update_profile(request):
    register = Register.objects.get(user=request.user)
    if request.method == "POST":
        re = request.POST
        rf = request.FILES
        try:
            image = rf['image']
            data = Register.objects.get(user=request.user)
            data.image = image
            data.save()
        except:
            pass
        user = User.objects.filter(id=request.user.id).update(username=re['username'], first_name=re['first_name'], last_name=re['last_name'])
        register = Register.objects.filter(user=request.user).update(address=re['address'], mobile=re['mobile'])
        register  = Register.objects.get(user=request.user)
        messages.success(request, "Updation Successful")
        return redirect('update_profile')
    return render(request, "update_profile.html", locals())

import json

def my_history(request):
    history = None
    try:
        data_user = Register.objects.get(user=request.user)
        history = History.objects.filter(user=request.user)
    except:
        try:
            nor_user = Register.objects.filter(address=data_user.address)
            normal_user_id = [i.user.id for i in nor_user]
            history = History.objects.filter(user__id__in=normal_user_id)
        except:
            pass
    if request.user.is_staff:
        history = History.objects.filter()
    return render(request, "my_history.html", locals())

def delete_history(request, pid):
    history = History.objects.get(id=pid)
    history.delete()
    messages.success(request, "Selected prediction data deleted successfully.")
    return redirect("my_history")

def delete_user(request, pid):
    user = User.objects.get(id=pid)
    user.delete()
    messages.success(request, "User deleted successfully.")
    return redirect("all_user")

def prediction_detail(request, pid):
    history = History.objects.get(id=pid)

    # Parse input_data and output_data safely
    try:
        input_data = json.loads(history.input_data.replace("'", '"')) if history.input_data else {}
    except json.JSONDecodeError:
        input_data = {}

    try:
        output_data = json.loads(history.output_data.replace("'", '"')) if history.output_data else {}
    except json.JSONDecodeError:
        output_data = {}

    # Send everything to the template
    context = {
        'username': f"{history.user.first_name} {history.user.last_name}",
        'prediction': history.prediction or 'N/A',
        'max_prob': history.max_prob or 'N/A',
        'skill_improvement_tips': history.skill_improvement_tips or 'N/A',
        'is_output': True,

        # Input fields
        'tenth_marks': input_data.get('ssc_p', 'N/A'),
        'tenth_board': input_data.get('ssc_b', 'N/A'),
        'twelfth_marks': input_data.get('hsc_p', 'N/A'),
        'twelfth_board': input_data.get('hsc_b', 'N/A'),
        'twelfth_stream': input_data.get('hsc_s', 'N/A'),
        'degree_cgpa': input_data.get('degree_p', 'N/A'),
        'degree_type': input_data.get('degree_t', 'N/A'),
        'work_experience': input_data.get('workex', 'N/A'),
        'entrance_score': input_data.get('etest_p', 'N/A'),
        # 'specialisation': input_data.get('specialisation', 'N/A'),
        # 'mba_marks': input_data.get('mba_p', 'N/A'),
        'backlogs': input_data.get('backlogs', 'N/A'),
        'certification': input_data.get('certification', 'N/A'),
        'domain': input_data.get('domain', 'N/A'),
        'skills': input_data.get('skills', 'N/A'),
        'preferred_company': input_data.get('preferredcompany', 'N/A'),
    }

    return render(request, "prediction_detail.html", context)


def all_user(request):
    data = Register.objects.filter()
    return render(request, "all_user.html", locals())

def data_visulisation(request):
    data_tye = request.GET.get('data-type')
    img = None
    if data_tye == 'feature':
        img, msg = plot_feature_graph()

    if data_tye == 'heatmap':
        img, msg= plot_heatmap()

    if data_tye == 'degree-boxplot':
        img, msg = plot_degree_boxplot()

    if data_tye == 'ssc-hsc':
        img, msg = plot_ssc_mba()

    if data_tye == 'probability':
        img, msg = plot_probability()

    if img:
        img = img.split('static')[1]
    return render(request, 'data-visualization.html', locals())


# View all messages (unread shown first)
def admin_messages_view(request):
    messagess = ContactMessage.objects.order_by('status', '-sent_on')  # unread first
    return render(request, 'admin_contact_messages.html', {'messagess': messagess})

# View message detail and mark as read
def message_detail_view(request, id):
    msg = get_object_or_404(ContactMessage, id=id)

    # Update status to read if unread
    if msg.status == 'unread':
        msg.status = 'read'
        msg.save()

    return render(request, 'admin_contact_message_detail.html', {'message': msg})




