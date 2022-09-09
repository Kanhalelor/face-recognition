from flask import Flask, render_template, Response
import cv2
import face_recognition
# from fer import FER
import datetime
import numpy as np
import csv

app = Flask(__name__)

current_date = datetime.datetime.now()

camera = cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.
robson_img = face_recognition.load_image_file("Pohamba\Robson Kanhalelo.JPG")
robson_face_encoding = face_recognition.face_encodings(robson_img)[0]

# Load a second sample picture and learn how to recognize it.
hage_img = face_recognition.load_image_file("Hage\Hage Geingob.jpg")
hage_face_encoding = face_recognition.face_encodings(hage_img)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    robson_face_encoding,
    hage_face_encoding
]
known_face_names = [
    "Robson Kanhalelo",
    "Hage Geingob"
]

students = known_face_names.copy()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# functions
@app.route('/Register')
def register():
    return render_template('register.html')
                        
def gen_frames():
    # create a csv file
    f = open('attendance.csv', 'a', newline='')
    writer = csv.writer(f)
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            rgb_small_frame = small_frame[:, :, ::-1]

            
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown Guest Detected!"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                
                face_names.append(name)

                # Write Data to a CSV file
                if name in known_face_names:
                    if name in students:
                        students.remove(name)
                        print(students)
                        current_time = datetime.now.strftime("%H,-%M-%S")
                        writer.writerow(name, current_time)
            # captured_image = cv2.imshow("caption", frame)
            # detector = FER()
            # emotion, score = detector.top_emotion(captured_image)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')#, gen_frames.emotion, gen_frames.score)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# start the application
if __name__=='__main__':
    app.run(debug=True)