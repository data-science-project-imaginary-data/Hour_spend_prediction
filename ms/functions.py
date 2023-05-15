import numpy as np
import pandas as pd
from ms import model

# comment types
all_types = ['อื่นๆ', 'ถนน', 'ทางเท้า', 'แสงสว่าง', 'ความปลอดภัย', 'ความสะอาด', 'น้ำท่วม', 'กีดขวาง',
             'ท่อระบายน้ำ', 'จราจร', 'สะพาน', 'สายไฟ', 'เสียงรบกวน', 'คลอง', 'ต้นไม้', 'ร้องเรียน', 'ป้าย',
             'สัตว์จรจัด', 'สอบถาม', 'PM2.5', 'เสนอแนะ', 'คนจรจัด', 'การเดินทาง', 'ห้องน้ำ', 'ป้ายจราจร']
toi = {t:idx for idx, t in enumerate(all_types)}

def predict(data):
    # model prediction
    types = data["types"][0]
    type_vec = [0] * 25
    if len(types) == 0:
        type_vec[0] = 1
    else:
        for t in types:
            print(t)
            type_vec[toi[t]] = 1
    X = np.array([type_vec])
    prediction = model.predict(X)[0]
    return prediction


def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    prediction = predict(X)
    return {
        'prediction': prediction
    }