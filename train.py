from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


batch_size = 10
epochs = 15

# Model loading
main_dir = 'Dataset'
train_dir = os.path.join(main_dir,'Train')
test_dir = os.path.join(main_dir,'Test')
valid_dir = os.path.join(main_dir,'Validation')

train_mask_dir = os.path.join(train_dir,'Mask')
train_nomask_dir = os.path.join(train_dir, 'Non Mask')


# Random train set 
train_mask_names = os.listdir(train_mask_dir)
#print(train_mask_names[:20])

train_nomask_names = os.listdir(train_nomask_dir)
#print(train_nomask_names[:20])



mask_pic =[]
for i in train_mask_names[0:300]:
  mask_pic.append(os.path.join(train_mask_dir,i))

nomask_pic= []
for i in train_nomask_names[0:300]:
  nomask_pic.append(os.path.join(train_nomask_dir, i))
  

#print(mask_pic)
#print(nomask_pic)

merged_list = mask_pic + nomask_pic


#image size - görüntünün okunacak satır ve sütunlarını temsil eder.Sayı büyüdükçe resim küçülür ve okunacak x-y artar.
nrows = 30
ncols = 30

#plt.figure(figsize=(10, 10))

#Display the image on the screen
for i in range(0,len(merged_list)):
    data = merged_list[i].split('/',4)
    sp = plt.subplot(nrows,ncols,i+1)
    sp.axis('Off')
    image = mpimg.imread(merged_list[i])
    sp.set_title(data,fontsize=10)
    plt.imshow(image,cmap='gray')
  
    #plt.show()


#Image preprocessing 
train_datagen = ImageDataGenerator ( rescale = 1./255, zoom_range = 0.2, rotation_range = 40, horizontal_flip =True )
#rescale : 1/255 ile ölçekleme yerine 0-1 aralığında yeniden ölçeklendirmek için bu işlem yapılır.                                   

test_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (150,150),   #image resize
                                                    batch_size = batch_size,
                                                    class_mode ='binary' 
                                                    )

test_generator = test_datagen.flow_from_directory(test_dir,
                                                    target_size = (150,150),
                                                    batch_size = batch_size,
                                                    class_mode ='binary' 
                                                    )

valid_generator = validation_datagen.flow_from_directory(valid_dir,
                                                    target_size = (150,150),
                                                    batch_size = batch_size,
                                                    class_mode ='binary' 
                                                    )


#Neural Network Building
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))  # %50 bırakma oranı

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())    # Gelen matrislerin tek boyutlu matrise 	çevrilmesidir.

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()


#Model Compile
model.compile( optimizer = 'adam' ,
               loss = 'binary_crossentropy',
               metrics = ['accuracy'] )

#Model Training
history = model.fit( train_generator,
                     epochs = epochs,
                     shuffle = True,  #her bir epoch’tan önce verilerin yerlerinin değiştirilmesi için kullanılır.
                     verbose = 1,     #0 eğitim sırasında ekranda bir sonuç göstermez, 1 anlık olarak güncellenen sonçları gösterir, 2 her bir epoch sonunda tek bir satır olarak çıktı verir.
                     validation_data = valid_generator)
                    
#print(history.history.keys())


#Plot loss curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'],loc='upper left')
plt.title('MODEL LOSS')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#Plot Accuracy curve
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train','validation'] , loc='upper left')
plt.title('MODEL ACCURACY')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()



#prediction = model.predict()
