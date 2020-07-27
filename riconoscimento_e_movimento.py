
import qi
import time
import sys
import argparse

import PIL.Image as Image
import os
import numpy as np
from keras.engine.saving import model_from_json
from keras.preprocessing import image
from naoqi import ALProxy




class EventClass(object):
    """
    A simple class to react to events.
    """

    def __init__(self, app):
        """
        Initialisation of qi framework and event detection.
        """
        super(EventClass, self).__init__()
        app.start()
        session = app.session

        self.ALDialog = session.service("ALDialog")
        self.video_service = session.service("ALVideoDevice")
        self.motion_service = session.service("ALMotion")
        self.audio_service = ALProxy("ALAudioPlayer", "192.168.1.117", 9559)
        # Get the service ALMemory.
        self.memory = session.service("ALMemory")
        self.tts = session.service("ALTextToSpeech")
        self.face_detection = session.service("ALSegmentation3D")
        self.face_detection.subscribe("EventClass")
        self.moved = False

    def on_segment_updated(self, v):
        """
        Callback for event detected.
        """
        value = self.memory.getData("Segmentation3D/BlobsList")

       #the value of the distance from the object detected
        print(str(value))
        dst = value[1][0][2]

        self.face_detection.unsubscribe("EventClass")

        #using di distance to move Pepper
        print dst
        x = dst - 2
        y=0
        theta=0
        if not self.moved:
            txt = "L'oggetto e a %.2f metri di distanza" % dst
            self.tts.say(txt)

            self.tts.say("Meglio controllare da vicino")

            self.motion_service.moveTo(x, y, theta)
            self.moved = True

            #it is used to set the head in the right position for the detection
            self.motion_service.setStiffnesses("Head", 1.0)

            # Simple command for the HeadYaw joint at 10% max speed
            names = "HeadPitch"
            #angles = 0.0 * almath.TO_RAD
            fractionMaxSpeed = 0.5
            #self.motion_service.setAngles(names, angles, fractionMaxSpeed)

            self.motion_service.setStiffnesses("Head", 0.0)

            t0 = time.time()

            # it sees the object
            naoImage = self.video_service.getImageRemote(self.videoClient)

            t1 = time.time()

            # Time the image transfer.
            print "acquisition delay ", t1 - t0

            self.video_service.unsubscribe(self.videoClient)

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
            # recognition of the image using the model of the network
            result = self.loaded_model_vicino.predict(test_image)

            if (np.argmax(result) == 0):
                self.tts.say("E' una box")
                print("E' una box con " + str(result[0][0]))
            elif (np.argmax(result) == 1):
                self.tts.say("E' una tesi")
                print("E' una tesi con " + str(result[0][1]))
            elif (np.argmax(result) == 2):
                self.tts.say("E' una valigetta")
                print("E' una valigetta con " + str(result[0][2]))
            else:
                self.tts.say("E' uno zaino")
                print("E' uno zaino con " + str(result[0][3]))

            #Pepper returns in the previous position
            x = -x
            self.motion_service.moveTo(x, y, theta)
            self.tts.say("Grazie per l'attenzione")
            os._exit(0)



    def run(self):
        print "Starting Pepper"

        self.ALDialog.setLanguage("Italian")

        # writing topics' qichat code as text strings (end-of-line characters are important!)
        topic_content_1 = ('topic: ~example_topic_content()\n'
                           'language: iti\n'
                           'concept:(food) [fruits chicken beef eggs]\n'
                           'u: (Mi senti) Tutt appost\n'
                           )

        # Loading the topics directly as text strings
        topic_name_1 = self.ALDialog.loadTopicContent(topic_content_1)

        # Activating the loaded topics
        self.ALDialog.activateTopic(topic_name_1)

        # Starting the dialog engine - we need to type an arbitrary string as the identifier
        # We subscribe only ONCE, regardless of the number of topics we have activated
        self.ALDialog.subscribe('my_dialog_example')

        #First get an image, then show it on the screen with PIL.

        # Get the service ALVideoDevice.

        resolution = 2  # VGA
        colorSpace = 11  # RGB

        #meanwhile it loads the network, it plays music
        self.tts.say("Sto caricando le reti. Vai pure a prendere un caffe ")
        self.audio_service.post.playFile("/data/home/nao/recordings/cameras/Elevator_Music.wav")
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model_vicino = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model_vicino.load_weights("model.h5")

        self.audio_service.stopAll()
        self.tts.say("Reti caricate")

        #the camera service, the attribute 0 is referred to which camera is used (0 is for the top one, 1 for the bottom one)
        #5 is for the fps
        self.videoClient = self.video_service.subscribeCamera("python_client", 0, resolution, colorSpace, 5)

        t0 = time.time()

        # Get a camera image.
        # image[6] contains the image data passed as an array of ASCII chars.

        naoImage = self.video_service.getImageRemote(self.videoClient)

        t1 = time.time()

        # Time the image transfer.
        print "acquisition delay ", t1 - t0

        # video_service.unsubscribe(videoClient)


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
        test_image = image.load_img("predict.png", target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.loaded_model_vicino.predict(test_image)

        if (np.argmax(result) == 0):
            self.tts.say("E' una box")
            print("E' una box con " + str(result[0][0]))
        elif (np.argmax(result) == 1):
            self.tts.say("E' una tesi")
            print("E' una tesi con " + str(result[0][1]))
        elif (np.argmax(result) == 2):
            self.tts.say("E' una valigetta")
            print("E' una valigetta con " + str(result[0][2]))
        else:
            self.tts.say("E' uno zaino")
            print("E' uno zaino con " + str(result[0][3]))

        # Connect the event callback.
        self.subscriber = self.memory.subscriber("Segmentation3D/SegmentationUpdated")
        self.subscriber.signal.connect(self.on_segment_updated)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print "Interrupted by user, stopping Pepper"
            self.face_detection.unsubscribe("EventClass")
            #stop
            sys.exit(0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.117",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["EventClass", "--qi-url=" + connection_url])
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    event_class = EventClass(app)
    event_class.run()
