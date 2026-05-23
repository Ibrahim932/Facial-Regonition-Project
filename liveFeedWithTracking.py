import os
import face_recognition
import imageio
from PIL import Image, ImageDraw, ImageTk
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
        self.camera_reader = imageio.get_reader("<video0>", size=(self.width, self.height))
        
        self.frame_skip_interval = 2  
        self.frame_count = 0
        
        self.current_locations = []
        self.current_names = []

        self.update_stream()

    def update_stream(self):
        try:
            frame = self.camera_reader.get_next_data()
            
            # If the camera returned an image size different from our request, 
            # we automatically adjust the Pillow layout so it doesn't break
            frame_height, frame_width = frame.shape[:2]
            
            if self.frame_count % self.frame_skip_interval == 0:
                self.current_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, self.current_locations)
                
                self.current_names = []
                for encoding in face_encodings:
                    if not self.known_encodings:
                        self.current_names.append("Unknown")
                        continue
                        
                    distances = face_recognition.face_distance(self.known_encodings, encoding)
                    best_match = distances.argmin()
                    
                    if distances[best_match] <= 0.55:
                        self.current_names.append(self.known_names[best_match])
                    else:
                        self.current_names.append("Unknown")

            pil_image = Image.fromarray(frame)
            
            # If the hardware forced a different frame size, scale it to match the window
            if frame_width != self.width or frame_height != self.height:
                pil_image = pil_image.resize((self.width, self.height), Image.Resampling.LANCZOS)
                
                # Recalculate coordinates ratio if image was scaled
                x_scale = self.width / frame_width
                y_scale = self.height / frame_height
                scaled_locations = []
                for (top, right, bottom, left) in self.current_locations:
                    scaled_locations.append((
                        int(top * y_scale), 
                        int(right * x_scale), 
                        int(bottom * y_scale), 
                        int(left * x_scale)
                    ))
                locations_to_draw = scaled_locations
            else:
                locations_to_draw = self.current_locations

            draw = ImageDraw.Draw(pil_image)

            for (top, right, bottom, left), name in zip(locations_to_draw, self.current_names):
                box_color = (34, 197, 94) if name != "Unknown" else (239, 68, 68)
                
                # Make the box line thicker (width=4) so it looks clean on an HD screen
                draw.rectangle(((left, top), (right, bottom)), outline=box_color, width=4)
                
                # Scale up text background box relative to HD resolution
                text_height = 20
                draw.rectangle(((left, bottom), (right, bottom + text_height + 6)), fill=box_color)
                draw.text((left + 8, bottom + 4), name, fill=(255, 255, 255))

            del draw
            self.frame_count += 1

            self.photo = ImageTk.PhotoImage(image=pil_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        except Exception as e:
            print(f"Stream read warning: {e}")

        self.window.after(15, self.update_stream)

    def close_app(self):
        self.camera_reader.close()
        self.window.destroy()
        print("Camera tracking session terminated safely.")

if __name__ == "__main__":
    DATABASE = "/Users/ibrahim/Coding/Facial Regonition Project/known_faces"
    
    root = tk.Tk()
    app = LiveTrackerApp(root, DATABASE)
    
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()