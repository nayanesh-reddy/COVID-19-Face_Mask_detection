import requests
import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

#%%

def thingSpeakWrite( Field_num, Data, Write_API_KEY='2JHI1UDGY1JNEMZN'):
    string='';
    for f,d in zip(Field_num, Data):
        string+='&field'+str(f)+'='+str(d)
        
    requests.get('https://api.thingspeak.com/update?api_key=' + 
                Write_API_KEY + string)

def IFTTT_post(value1, value2, value3):
    url='https://maker.ifttt.com/trigger/Someone_at_the_door/with/key/kqVSFUShd51yy9xw_5tPDYxY04nYnn90q36nNWdVUu5'
    requests.post(url, data = {'value1': str(value1), 'value2':str(value2), 'value3':str(value3)})

#%%
model = load_model("mask_model.h5")

def draw_image_with_boxes(image, result_list):
    global model,frame1,S
    count_mask=count_no_mask=0
    
    color_dict={0:(0,255,0),1:(0,0,255)}
    #labels_dict={0:'MASK', 1:'NO MASK'}
    im=image
    
    for result in result_list:
        x, y, w, h = result['box']
        I=im[y:y+h,x:x+w,:]
        I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

        I = cv.resize(I, (224, 224))
        I = img_to_array(I)
        I = np.expand_dims(I, axis=0)
        I = preprocess_input(I)
    
        preds = model.predict(I)
        label=np.argmax(preds[0])
        
        cv.rectangle(frame1,(S*x,S*y),(S*x+S*w,S*y+S*h),color_dict[label],2)
        #cv.putText(frame1, labels_dict[label], (S*x, S*(y-10)),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        if not label:
            count_mask+=1
        else:
            count_no_mask+=1
              
    return im, count_mask, count_no_mask

#%%

detector = MTCNN()

cap=cv.VideoCapture('people1.mkv'); S=1; delay=10000

#cap=cv.VideoCapture(0); S = 1; delay=1

_, frame = cap.read()
rows,cols,_ = frame.shape
rows = int(rows/S); cols = int(cols/S)


while cap.isOpened():
    ret, frame1 = cap.read()
    
    
    if not ret:
        break 
    
    frame =cv.resize(frame1,(cols, rows))
    
    rgb=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    

    faces = detector.detect_faces(rgb)
    frame, count_mask, count_no_mask = draw_image_with_boxes(frame, faces)
    
    thingSpeakWrite([1,2],[count_mask, count_no_mask])
    
    #if (count_mask + count_no_mask)>0:
    #  IFTTT_post(count_mask+count_no_mask, count_mask, count_no_mask)
    
    cv.imshow('face',frame1)
    
    if cv.waitKey(delay) == ord(' '):
        break 
    
cap.release()
cv.destroyAllWindows()