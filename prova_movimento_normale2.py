#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Get an image. Display it and save it using PIL."""

import qi
import argparse
import sys
import time
import PIL.Image as Image
import os
import numpy as np
from keras.engine.saving import model_from_json
from keras.preprocessing import image
from naoqi import ALProxy


def main(session):
    """
           This example uses ALDialog methods.
           It's a short dialog session with two topics.
           """
    # Getting the service ALDialog
    ALDialog = session.service("ALDialog")
    ALDialog.setLanguage("Italian")

    # writing topics' qichat code as text strings (end-of-line characters are important!)
    topic_content_1 = ('topic: ~example_topic_content()\n'
                       'language: iti\n'
                       'concept:(food) [fruits chicken beef eggs]\n'
                       'u: (Mi senti) Tutt appost\n'
                       )

    # Loading the topics directly as text strings
    topic_name_1 = ALDialog.loadTopicContent(topic_content_1)

    # Activating the loaded topics
    ALDialog.activateTopic(topic_name_1)

    # Starting the dialog engine - we need to type an arbitrary string as the identifier
    # We subscribe only ONCE, regardless of the number of topics we have activated
    ALDialog.subscribe('my_dialog_example')
    """
    First get an image, then show it on the screen with PIL.
    """
    # Get the service ALVideoDevice.

    video_service = session.service("ALVideoDevice")
    resolution = 2    # VGA
    colorSpace = 11   # RGB

    json_file = open('model_2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_2.h5")
    tts = ALProxy("ALTextToSpeech", "192.168.1.117", 9559)

    json_file2 = open('model.json', 'r')
    loaded_model_json2 = json_file2.read()
    json_file2.close()
    loaded_model2 = model_from_json(loaded_model_json2)
    # load weights into new model
    loaded_model2.load_weights("model.h5")



    videoClient = video_service.subscribeCamera("python_client", 0, resolution, colorSpace, 5)

    t0 = time.time()
    motion_service = session.service("ALMotion")

    # Get a camera image.
    # image[6] contains the image data passed as an array of ASCII chars.

    naoImage = video_service.getImageRemote(videoClient)

    t1 = time.time()

    # Time the image transfer.
    print "acquisition delay ", t1 - t0

    #video_service.unsubscribe(videoClient)


    # Now we work with the image returned and save it as a PNG  using ImageDraw
    # package.

    # Get the image size and pixel array.
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    image_string = str(bytearray(array))



    # Create a PIL Image from our pixel array.
    im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)



    # Save the image.
    im.save("predict.png", "PNG")
    test_image = image.load_img("predict.png",target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)

    if (np.argmax(result) == 0):
        tts.say("E' una box")
        print("E' una box con " + str(result[0][0]))
    elif (np.argmax(result) == 1):
        tts.say("E' una tesi")
        print("E' una tesi con " + str(result[0][1]))
    elif (np.argmax(result) == 2):
        tts.say("E' una valigetta")
        print("E' una valigetta con " + str(result[0][2]))
    else:
        tts.say("E' uno zaino")
        print("E' uno zaino con " + str(result[0][3]))
    #im.show()
    #os.remove("C:\Users\leopo\PycharmProjects\prova\camImage.png")

    posture_service = session.service("ALRobotPosture")
    localizationProxy = session.service("ALLocalization")

    x = 1.7
    y = 0
    theta = 0
    motion_service.moveTo(x, y, theta)

    motion_service.setStiffnesses("Head", 1.0)

    # Simple command for the HeadYaw joint at 10% max speed
    names = "HeadPitch"
    #    angles = 0.0 * almath.TO_RAD
    fractionMaxSpeed = 0.5
    # motion_service.setAngles(names, angles, fractionMaxSpeed)

    motion_service.setStiffnesses("Head", 0.0)

    # rivede l'oggetto
    naoImage = video_service.getImageRemote(videoClient)

    t1 = time.time()

    # Time the image transfer.
    print "acquisition delay ", t1 - t0

    video_service.unsubscribe(videoClient)

    # Now we work with the image returned and save it as a PNG  using ImageDraw
    # package.

    # Get the image size and pixel array.
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    image_string = str(bytearray(array))

    # Create a PIL Image from our pixel array.
    im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)

    # Save the image.
    im.save("predict2.png", "PNG")
    test_image = image.load_img("predict2.png", target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model2.predict(test_image)

    #   if (np.argmax(result) == 0):
    #      tts.say("E' una box")
    #     print("E' una box con " + str(result[0][0]))
    # elif (np.argmax(result) == 1):
    #   tts.say("E' una tesi")
    #  print("E' una tesi con " + str(result[0][1]))
    # elif (np.argmax(result) == 2):
    #   tts.say("E' una valigetta")
    #  print("E' una valigetta con " + str(result[0][2]))
    # elif (np.argmax(result) == 3):
    #   tts.say("E' una testa")
    #  print("E' una testa con " + str(result[0][3]))
    # else:
    #   tts.say("E' uno zaino")
    #  print("E' uno zaino con " + str(result[0][4]))

    # fine
    if (np.argmax(result) == 0):
        tts.say("E' una box")
        print("E' una box con " + str(result[0][0]))
    elif (np.argmax(result) == 1):
        tts.say("E' una tesi")
        print("E' una tesi con " + str(result[0][1]))
    elif (np.argmax(result) == 2):
        tts.say("E' una valigetta")
        print("E' una valigetta con " + str(result[0][2]))
    else:
        tts.say("E' uno zaino")
        print("E' uno zaino con " + str(result[0][3]))

    x = -1.7
    y = 0
    theta = 0
    motion_service.moveTo(x, y, theta)


if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.117",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
            session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)