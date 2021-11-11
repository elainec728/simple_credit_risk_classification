from django.shortcuts import render

# Create your views here.
import logging
from django.views.decorators.csrf import csrf_exempt
from django.core.serializers.json import json
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse, Http404
from credit_risk.credit_risk import *


@require_http_methods(["POST"])
@csrf_exempt
def scoring_service(request):
    try:
        print(request)
        params = request.body.decode('utf-8')
        # start_id = params.find('{\'fields\':')
        # stop_id = params.rfind(']]}') + 3
        # params = params[start_id:stop_id] # notebook not suitable

        print('request. {}: {}'.format(request.method, params))
        payload_scoring = {"input_data": params, }
        print('payload #{}'.format(str(len(payload_scoring['input_data']))))

        with open(MODEL_PATH, 'rb') as f:
            res = pickle.load(f)
        print('before scoring')
        result = scoring(payload_scoring, *res)
        context = result
    except Exception as e:
        print(e)
        context = {
            "fields": [],
            "labels": [],
            "values": []
        }

    return HttpResponse(json.dumps(context, ensure_ascii=False), content_type='application/json; charset=utf-8')
