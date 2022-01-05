import json
from json import JSONEncoder
from .utils import *
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import numpy as np
import cv2

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def process_image(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        text_data = detect_text(image)
        image_data = detect_marked(image)
        mappings = find_nearest(text_data, image_data)
        return_object = json.dumps(mappings, cls=NumpyArrayEncoder)
        return JsonResponse({"Mapping":return_object}, status=200)
    return HttpResponse(status=200)
