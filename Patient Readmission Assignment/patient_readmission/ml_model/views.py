from django.shortcuts import render
from django.http import HttpResponseBadRequest, HttpResponse
import json
import pandas as pd
from patient_readmission.settings import PATIENT_PREDICTION_MODEL


### move to api file
# model = pickle.load(open('model_asd','rb'))
def get_prediction(request):
    if request.method != "POST":
        return
    pred_dict = json.loads(request.body)
    print(pred_dict)
    print('processing data...')
    pred_X = pd.DataFrame.from_dict(pred_dict, orient='columns')
    print(pred_X)
    predictions = PATIENT_PREDICTION_MODEL.predict(pred_X).tolist()
    return HttpResponse(json.dumps({
        "predictions": predictions
    }), content_type="application/json")