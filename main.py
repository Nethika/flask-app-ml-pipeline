from flask import Flask, render_template, Response
from camera import VideoCamera
from flask import jsonify
import json
import time
import cv2
import datetime

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    nexttime = time.time() 
    while True:
        im,frame = camera.get_frame()
        if nexttime < time.time():
            #print(nexttime)
            nexttime += 13
            #print("Next time:",nexttime)
            datetime_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            fimage = "model_start/" + datetime_name + ".jpg"
            #print(fimage)
            cv2.imwrite(fimage, im)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_anlysis')
def video_anlysis():
    # Read Json
    j_file = "results.json" 
    j_data = json.load(open(j_file))
    #print("json:",j_data)
    """
    return jsonify({"result_image_path": "result_images/20190627205539.png", "Calculated Age": 28, "Calculated Gender": "F", 
    "Calculated Emotion": "happy", "Person Identified": "Nethika Suraweera", 
    "Real Age": 36, "Real Gender": "F", 
    "About": "AI Scientist at Tinman Kinetics-exploring the wonders of modern artificial intelligence and deep neural network solutions to practical problems. Previously worked as a Data Scientist at ByteCubed - experimenting efficient use of data analysis tools to derive accurate outcomes from big data. Before joining the commercial world, Nethika held positions in academia at CU Boulder, at University of Tennessee - Knoxville and at University of Moratuwa - Sri Lanka. She provided computational collaboration to scientific research through molecular models, simulations and data analysis. Enjoys outdoors, hiking, camping, music and Taekwondo.",
     "image1": "profile/nethika/image1.png",
     "image2": "profile/nethika/image1.png",
     "image3": "profile/nethika/image1.png"})
    """
    return jsonify(j_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
