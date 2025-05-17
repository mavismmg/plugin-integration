from app.plugins import PluginBase, Menu, MountPoint
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.utils.translation import gettext as _
from django.http import HttpResponseRedirect
from django.urls import reverse

import json, shutil
import os
import pickle
from PIL import Image
import numpy as np

def get_memory_stats():
    """
    Get node total memory and memory usage (Linux only)
    https://stackoverflow.com/questions/17718449/determine-free-ram-in-python
    """
    try:
        with open('/proc/meminfo', 'r') as mem:
            ret = {}
            tmp = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) == 'MemTotal:':
                    ret['total'] = int(sline[1])
                elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                    tmp += int(sline[1])
            ret['free'] = tmp
            ret['used'] = int(ret['total']) - int(ret['free'])

            ret['total'] *= 1024
            ret['free'] *= 1024
            ret['used'] *= 1024
        return ret
    except:
        return {}

def process_image_with_model(image_path, model_path):
    """
    Processa uma imagem usando um modelo treinado (exemplo com sklearn).
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).reshape(1, -1)  # Ajuste conforme seu modelo
    result = model.predict(img_array)
    return result

def list_available_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return [
        f for f in os.listdir(models_dir)
        if f.endswith('.pkl')
    ]

class Plugin(PluginBase):
    def main_menu(self):
        return [Menu(_("Diagnostic"), self.public_url(""), "fa fa-chart-pie fa-fw")]

    def app_mount_points(self):
        @login_required
        def diagnostic(request):
            # Disk space
            total_disk_space, used_disk_space, free_disk_space = shutil.disk_usage('./')

            template_args = {
                'title': 'Diagnostic',
                'total_disk_space': total_disk_space,
                'used_disk_space': used_disk_space,
                'free_disk_space': free_disk_space
            }

            # Memory (Linux only)
            memory_stats = get_memory_stats()
            if 'free' in memory_stats:
                template_args['free_memory'] = memory_stats['free']
                template_args['used_memory'] = memory_stats['used']
                template_args['total_memory'] = memory_stats['total']

            # Load models
            models = list_available_models()
            template_args['models'] = models

            selected_model = request.POST.get('model') if request.method == 'POST' else (models[0] if models else None)

            # Process image with a pretrained model
            if request.method == 'POST' and request.FILES.get('image'):
                image_file = request.FILES['image']
                image_path = f"/tmp/{image_file.name}"
                with open(image_path, 'wb+') as destination:
                    for chunk in image_file.chunks():
                        destination.write(chunk)
                model_path = os.path.join(os.path.dirname(__file__), 'pretrained_model.pkl')
                try:
                    result = process_image_with_model(image_path, model_path)
                    template_args['model_result'] = result
                except Exception as e:
                    template_args['model_result'] = f"Erro ao processar: {e}"

            return render(request, self.template_path("diagnostic.html"), template_args)
        
        @login_required
        def manage_models(request):
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            if request.method == 'POST':
                action = request.POST.get('action')
                if action == 'upload' and request.FILES.get('new_model'):
                    new_model = request.FILES['new_model']
                    model_path = os.path.join(models_dir, new_model.name)
                    with open(model_path, 'wb+') as destination:
                        for chunk in new_model.chunks():
                            destination.write(chunk)
                elif action == 'delete':
                    model_to_delete = request.POST.get('model_to_delete')
                    if model_to_delete:
                        try:
                            os.remove(os.path.join(models_dir, model_to_delete))
                        except Exception:
                            pass
                return HttpResponseRedirect(reverse('diagnostic_manage_models'))

            models = list_available_models()
            return render(request, self.template_path("diagnostic.html"), {
                'title': 'Gerenciar Modelos',
                'models': models
            })

        return [
            MountPoint('$', diagnostic, name='diagnostic'),
            MountPoint('manage_models', manage_models, name='diagnostic_manage_models')
        ]