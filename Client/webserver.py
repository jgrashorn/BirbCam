# import the necessary packages
from flask import Response
from flask import Flask
from flask import render_template
import threading
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# Replace with your Raspberry Pi's IP and streaming port
udp_stream_url = 'udp://@192.168.178.83:8081?pkt_size=1316'

def capture_stream():
    global outputFrame
    cap = cv2.VideoCapture(udp_stream_url)
    if not cap.isOpened():
        print("Error: Could not open UDP stream.")
        return
    
    while True:
        # Capture the latest frame and update the shared variable
        ret, frame = cap.read()
        if ret:
            outputFrame = frame
        else:
            print("Warning: No frame received.")
            break

    cap.release()
	
if __name__ == '__main__':
	
	# start a thread that will capture the stream
	t = threading.Thread(target=capture_stream)
	t.daemon = True
	t.start()
	
	# start the flask app
	app.run(host='0.0.0.0', port=5000, debug=True,
		threaded=True, use_reloader=False)