import telebot
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import pickle
from keras.applications.vgg16 import VGG16
import os
import threading

bot = telebot.TeleBot('6150246741:AAEFYKG7N9xXsydWDf9PhOflyzE4wPV0CvA')#Token_Id

with open('C:/Users/Lenovo/Downloads/TELEBOT Bot/TELEBOT/model2.pkl', 'rb') as f:#load model 
    svm = pickle.load(f)
model = VGG16(weights='imagenet', include_top=False)

def start(message):
    bot.send_message(message.chat.id, "Welcome to IV_Detect")

def help(message):
    bot.send_message(message.chat.id, "Just send the image or video which you want to check \nif it's not work then send image or video as file")

def image(message):#for image input
    photo = message.photo[-1].file_id
    
    photo_info = bot.get_file(photo)
    photo_path = bot.download_file(photo_info.file_path)
    with open('img.jpg', 'wb') as img_file:
        img_file.write(photo_path)
    
    photo_path = "img.jpg"
    if is_jpg(photo_path):#feature extraction for image
        img = load_img(photo_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        features = model.predict(img)

        features = features.flatten()
        features = features.reshape(1, -1)
        label = svm.predict(features)[0]

        if label == 0:
            bot.send_message(message.chat.id, "The new image is AI-generated.")
        else:
            bot.send_message(message.chat.id, "The new image is real.")

    os.remove('img.jpg')

def is_jpg(photo_path):#check image is jpg form
    _, extension = os.path.splitext(photo_path)
    return extension.lower() == '.jpg'


@bot.message_handler(commands=['start'])
def handle_start(message):
    start(message)

@bot.message_handler(commands=['help'])
def handle_help(message):
    help(message)

@bot.message_handler(content_types=['photo'])
def handle_image(message):
    threading.Thread(target=image, args=(message,)).start()


bot.polling()
