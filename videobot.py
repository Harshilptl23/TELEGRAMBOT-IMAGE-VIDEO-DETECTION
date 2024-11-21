import telebot
import numpy as np
from keras.applications.vgg16 import preprocess_input
import pickle
from keras.applications.vgg16 import VGG16
import os
import cv2
from moviepy.editor import *
import threading
import concurrent.futures

bot = telebot.TeleBot('6265355773:AAER7kzPTTMXhZt3NDB5b5lkGX_PGGUPux8')#Token_Id

with open('model2.pkl', 'rb') as f:#load model
    svm = pickle.load(f)
model = VGG16(weights='imagenet', include_top=False)
def start(message):
    bot.send_message(message.chat.id, "Welcome to IV_Detect")

def help(message):
    bot.send_message(message.chat.id, "Just send the video which you want to check \nif it's not work then send video as file \n For image visit our Bot (t.me/image_checker_bot)")

def process_frame(frame):#process all frame from video
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)

    features = model.predict(frame)

    features = features.flatten()
    features = features.reshape(1, -1)
    label = svm.predict(features)[0]

    return label

def file(message):#for file inpute
    file_info = bot.get_file(message.document.file_id)
    video_path = bot.download_file(file_info.file_path)
    with open('media.gif', 'wb') as video_file:
        video_file.write(video_path)

    clip = VideoFileClip("media.gif")#convert gif to video
    clip.write_videofile("output.mp4")

    video_capture = cv2.VideoCapture('output.mp4')
    bot.send_message(message.chat.id,"File going under process.")


    frames = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)

    video_capture.release()

    with concurrent.futures.ThreadPoolExecutor() as executor:#multiple threading
        results = executor.map(process_frame, frames)
        print(results)

    count = sum(results)
    ac = count * 100
    acoutput = ac / len(frames)
    acoutput1 = "{:.2f}".format(acoutput)
    if acoutput > 50:
        bot.send_message(message.chat.id, f"The new file {acoutput1}% real.")

    else:
        ans = 100 - acoutput
        ans1="{:.2f}".format(ans)
        bot.send_message(message.chat.id, f"The new file {ans1}% AI-generated.")


    os.remove('media.gif')
    os.remove('output.mp4')
    return acoutput1

def video(message):#for video inpute
    file_info = bot.get_file(message.video.file_id)
    video_path = bot.download_file(file_info.file_path)
    with open('media.mp4', 'wb') as video_file:
        video_file.write(video_path)

    video_capture = cv2.VideoCapture('media.mp4')
    bot.send_message(message.chat.id, "Video going under process.")


    frames = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)

    video_capture.release()

    with concurrent.futures.ThreadPoolExecutor() as executor:#multiple threading
        results = executor.map(process_frame, frames)

    count = sum(results)
    ac = count * 100
    acoutput = ac / len(frames)
    acoutput1 = "{:.2f}".format(acoutput)
    if acoutput > 50:
        bot.send_message(message.chat.id, f"The new video {acoutput1}% real.")

    else:
        ans = 100 - acoutput
        ans1="{:.2f}".format(ans)
        bot.send_message(message.chat.id, f"The new video {ans1}% AI-generated.")

    os.remove('media.mp4')
    return acoutput1
@bot.message_handler(commands=['start'])
def handle_start(message):
    start(message)

@bot.message_handler(commands=['help'])
def handle_help(message):
    help(message)
@bot.message_handler(content_types=['document'])
def handle_video(message):
    threading.Thread(target=file, args=(message,)).start()

@bot.message_handler(content_types=['video'])
def handle_video1(message):
    threading.Thread(target=video, args=(message,)).start()

bot.polling()
