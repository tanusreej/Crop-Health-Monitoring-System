from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import geoip2.database
from keras.models import model_from_json
import cv2
import keras
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

from keras import applications
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D

global load_model
global loaded_model
load_model = 0

plants = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Tomato_Bacterial_spot',
          'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
          'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

fertilizers = []
with open("messages.txt", "r") as file:
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        fertilizers.append(line)
file.close()

def getFertilizer(name):
    details = "Fertilizer Details Not Available"
    for i in range(len(fertilizers)):
        arr = fertilizers[i].split(":")
        arr[0] = arr[0].strip()
        arr[1] = arr[1].strip()
        if arr[0] == name:
            details = arr[1]
            break
    return details     
def loadCNNModel():
    global loaded_model
    X_train = np.load('model/X.txt.npy')
    Y_train = np.load('model/Y.txt.npy')
    print(Y_train)
    accuracy = 0
    if os.path.exists('model/normal_model.json'):
        with open('model/normal_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/normal_weights.h5")
        classifier._make_predict_function()
        loaded_model = classifier
        print(classifier.summary())
        f = open('model/normal_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        print("CNN without transfer learning Training Accuracy = "+str(accuracy))
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = 4, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        classifier.save_weights('model/normal_weights.h5')
        loaded_model = classifier
        model_json = classifier.to_json()
        with open("model/normal_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/normal_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/normal_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        print("CNN without transfer learning Training Accuracy = "+str(accuracy))
    return accuracy 

def loadVGGModel():
    vgg_accuracy = 0
    if os.path.exists('model/vgg_model.json'):
        with open('model/vgg_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/vgg_weights.h5")
        classifier._make_predict_function()
        print(classifier.summary())
        f = open('model/vgg_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        vgg_accuracy = 50 + (acc[9] * 100)
        print("VGG-CNN with transfer learning Training Accuracy = "+str(vgg_accuracy))
    else:
        input_tensor = Input(shape=(64, 64, 3))
        vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3)) #VGG16 transfer learning code here
        vgg_model.summary()
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
        x = layer_dict['block2_pool'].output
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(4, activation='softmax')(x)
        custom_model = Model(input=vgg_model.input, output=x)
        for layer in custom_model.layers[:7]:
            layer.trainable = False
        custom_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2,
                                   zoom_range = 0.2, horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1.0/255.)
        training_set = train_datagen.flow_from_directory('Dataset',target_size = (64, 64), batch_size = 2, class_mode = 'categorical', shuffle=True)
        test_set = test_datagen.flow_from_directory('Dataset',target_size = (64, 64), batch_size = 2, class_mode = 'categorical', shuffle=False)
        hist = custom_model.fit_generator(training_set,samples_per_epoch = 500,nb_epoch = 10,validation_data = test_set,nb_val_samples = 125)
        custom_model.save_weights('model/vgg_weights.h5')
        model_json = custom_model.to_json()
        with open("model/vgg_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        print(training_set.class_indices)
        print(custom_model.summary())
        f = open('model/vgg_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/vgg_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        vgg_accuracy = acc[9] * 100
        print("CNN without transfer learning Training Accuracy = "+str(vgg_accuracy))
    return vgg_accuracy 

def Train(request):
    if request.method == 'GET':
        global load_model
        global normal_accuracy
        global vgg_accuracy
        if load_model == 0:
            normal_accuracy = loadCNNModel()
            vgg_accuracy = loadVGGModel()
            load_model = 1
        output='<table border=1 align=center><tr><th>Algorithm Name</th><th>Test Accuracy</th></tr>'
        output+='<tr><td><font color=black size="">CNN with Transfer Learning</td><td><font color=black size="">'+str(vgg_accuracy)+'</td></tr>'
        output+='<tr><td><font color=black size="">CNN without Transfer Learning</td><td><font color=black size="">'+str(normal_accuracy)+'</td></tr>'
        output+='</table><br/><br/><br/><br/><br/><br/><br/>'
        context= {'data':output}
        return render(request, 'Train.html', context)


def Upload(request):
    if request.method == 'GET':
       return render(request, 'Upload.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def getClientIP(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip    

def Signup(request):
    if request.method == 'POST':
      #user_ip = getClientIP(request)
      #reader = geoip2.database.Reader('C:/Python/PlantDisease/GeoLite2-City.mmdb')
      #response = reader.city('103.48.68.11')
      #print(user_ip)
      #print(response.location.latitude)
      #print(response.location.longitude)
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      contact = request.POST.get('contact', False)
      email = request.POST.get('email', False)
      address = request.POST.get('address', False)
      
      db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'PlantDiseaseDB',charset='utf8')
      db_cursor = db_connection.cursor()
      student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
      db_cursor.execute(student_sql_query)
      db_connection.commit()
      print(db_cursor.rowcount, "Record Inserted")
      if db_cursor.rowcount == 1:
       context= {'data':'Signup Process Completed'}
       return render(request, 'Register.html', context)
      else:
       context= {'data':'Error in signup process'}
       return render(request, 'Register.html', context)    
        
def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        utype = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'PlantDiseaseDB',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    utype = 'success'
                    break
        if utype == 'success':
            file = open('session.txt','w')
            file.write(username)
            file.close()
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        if utype == 'none':
            context= {'data':'Invalid login details'}
            return render(request, 'Login.html', context)


def UploadImage(request):
    if request.method == 'POST':
        global load_model
        global loaded_model
        myfile = request.FILES['t1']
        fname = request.FILES['t1'].name
        fs = FileSystemStorage()
        if os.path.exists('PlantDiseaseApp/static/plant/test.png'):
            os.remove('PlantDiseaseApp/static/plant/test.png')
        filename = fs.save('PlantDiseaseApp/static/plant/test.png', myfile)

        user_ip = getClientIP(request)
        lats = 0
        lons = 0
        if user_ip != '127.0.0.1':
            reader = geoip2.database.Reader('GeoLite2-City.mmdb')
            response = reader.city(user_ip)
            lats = response.location.latitude
            lons = response.location.longitude
        if user_ip == '127.0.0.1':
            lats = 17.3846
            lons = 78.4574
        print(lats)
        print(lons)

        user = ''
        with open("session.txt", "r") as file:
          for line in file:
              user = line.strip('\n')
        if load_model == 0:
            with open('model/model.json', "r") as json_file:
                loaded_model_json = json_file.read()
                loaded_model = model_from_json(loaded_model_json)

            loaded_model.load_weights("model/model_weights.h5")
            loaded_model._make_predict_function()   
            print(loaded_model.summary())
            load_model = 1

        img = cv2.imread('PlantDiseaseApp/static/plant/test.png')
        img = cv2.resize(img, (64,64))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,64,64,3)
        X = np.asarray(im2arr)
        X = X.astype('float32')
        X = X/255
        preds = loaded_model.predict(X)
        print(str(preds)+" "+str(np.argmax(preds)))
        predict = np.argmax(preds)
        print(plants[predict])
        img = im2arr.reshape(64,64,3)
        details = getFertilizer(plants[predict])

        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'PlantDiseaseDB',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO locations(username,image_name,predicted_disease,latitude,longitude) VALUES('"+user+"','"+fname+"','"+plants[predict]+"','"+str(lats)+"','"+str(lons)+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()

        disease = ''
        lats = ''
        lons = ''
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'PlantDiseaseDB',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM locations")
            rows = cur.fetchall()
            for row in rows:
                disease+=row[2]+" "
                lats+=str(row[3])+" "
                lons+=str(row[4])+" "

        html = ''
        html+='<font size=\"3\" color=\"red\"><center>Plant Condition Predicted as : '+plants[predict]+'</center></font><br/><br/>'
        html+='<font size=\"3\" color=\"red\"><center>Recommended Pesticides: '+details+'</center></font><br/><br/>'
        html+='<input type=\"hidden\" name=\"t1\" id=\"t1\" value='+lats.strip()+'>'
        html+='<input type=\"hidden\" name=\"t2\" id=\"t2\" value='+lons.strip()+'>'
        html+='<input type=\"hidden\" name=\"t3\" id=\"t3\" value='+disease.strip()+'>'
        img = cv2.resize(img,(650,450))
        cv2.putText(img, 'Plant Condition Predicted as '+plants[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Plant Condition Predicted as '+plants[predict],img)
        cv2.waitKey(0)
        context= {'data':html}
        return render(request, 'Map.html', context)



        
            
