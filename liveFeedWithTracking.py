import os
import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

def load_database(folder):
    known_encodings, known_names = [], []
    if not os.path.exists(folder):
        print(f"Error: Database folder '{folder}' missing.")
        return known_encodings, known_names
        
    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = face_recognition.load_image_file(os.path.join(folder, file))
            encodes = face_recognition.face_encodings(img)
            if encodes:
                known_encodings.append(encodes[0])
                known_names.append(os.path.splitext(file)[0].capitalize())
    return known_encodings, known_names

class LiveTrackerApp:
    def __init__(self, window, database_folder):
        self.window = window
        self.window.title("Real-Time Facial Recognition Tracker")
        
        # 1. DEFINE YOUR DESIRED RESOLUTION
        # Most Mac webcams natively support 1280x720 (720p HD) or 1920x1080 (1080p Full HD)
        self.width = 1280
        self.height = 720
        
        self.known_encodings, self.known_names = load_database(database_folder)
        print(f"Loaded {len(self.known_names)} identities. Opening tracking window...")

        # 2. Match the UI display canvas size to the HD resolution
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # 3. Request HD resolution directly from the Mac webcam hardware using plugins
        # we pass size parameters to force the camera configuration
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Optional: reduce buffering for lower latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame_skip_interval = 2  
        self.frame_count = 0
        
        self.current_locations = []
        self.current_names = []

        self.update_stream()

    def update_stream(self):

        try:
            ret, frame = self.cap.read()

            if not ret:
                return

            # Flip horizontally (optional mirror effect)
            frame = cv2.flip(frame, 1)

            # =========================
            # FACE PROCESSING
            # =========================
            if self.frame_count % self.frame_skip_interval == 0:

                # Resize for fast processing
                small_frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=0.25,
                    fy=0.25
                )

                # BGR -> RGB
                rgb_small = cv2.cvtColor(
                    small_frame,
                    cv2.COLOR_BGR2RGB
                )

                # FAST HOG DETECTION
                self.current_locations = face_recognition.face_locations(
                    rgb_small,
                    model="hog"
                )

                encodings = face_recognition.face_encodings(
                    rgb_small,
                    self.current_locations
                )

                self.current_names = []

                for encoding in encodings:

                    name = "Unknown"

                    if len(self.known_encodings) > 0:

                        distances = face_recognition.face_distance(
                            self.known_encodings,
                            encoding
                        )

                        best_match = np.argmin(distances)

                        if distances[best_match] < 0.55:
                            name = self.known_names[best_match]

                    self.current_names.append(name)

            # =========================
            # DRAW BOXES
            # =========================
            for (top, right, bottom, left), name in zip(
                self.current_locations,
                self.current_names
            ):

                # Scale back to original frame size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    color,
                    3
                )

                cv2.rectangle(
                    frame,
                    (left, bottom - 35),
                    (right, bottom),
                    color,
                    cv2.FILLED
                )

                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            # =========================
            # DISPLAY FRAME
            # =========================
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame_rgb)

            self.photo = ImageTk.PhotoImage(image=image)

            if not hasattr(self, "image_on_canvas"):

                self.image_on_canvas = self.canvas.create_image(
                    0,
                    0,
                    image=self.photo,
                    anchor=tk.NW
                )

            else:
                self.canvas.itemconfig(
                    self.image_on_canvas,
                    image=self.photo
                )

            self.frame_count += 1

        except Exception as e:
            print(f"Stream warning: {e}")

        self.window.after(1, self.update_stream)

    def close_app(self):
        self.cap.release()
        self.window.destroy()
        print("Camera closed safely.")

if __name__ == "__main__":
    DATABASE = "/Users/ibrahim/Coding/Facial Regonition Project/known_faces"
    
    root = tk.Tk()
    app = LiveTrackerApp(root, DATABASE)
    
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()