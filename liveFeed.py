import os
import face_recognition
import imageio
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

def load_database(folder):
    known_encodings, known_names = [], []
    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = face_recognition.load_image_file(os.path.join(folder, file))
            encodes = face_recognition.face_encodings(img)
            if encodes:
                known_encodings.append(encodes[0])
                known_names.append(os.path.splitext(file)[0].capitalize())
    return known_encodings, known_names

def start_live_recognition(database_folder):
    known_encodings, known_names = load_database(database_folder)
    print(f"✅ Loaded {len(known_names)} identities. Starting live tracking feed...")
    print("Press Ctrl+C in the terminal to stop the stream.\n")

    # Connect to your Mac's built-in webcam system
    # '<video0>' targets the default integrated camera interface
    camera_reader = imageio.get_reader("<video0>")
    
    frame_count = 0
    
    try:
        for frame in camera_reader:
            # PERFORMANCE TRICK: Only calculate math on every 3rd frame
            if frame_count % 3 == 0:
                # Find all faces in the current live webcam snapshot
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                
                if len(face_locations) > 0:
                    print(f"🎬 [Frame {frame_count}] Tracking {len(face_locations)} face(s) live:")
                    
                    for coords, unknown_encoding in zip(face_locations, face_encodings):
                        distances = face_recognition.face_distance(known_encodings, unknown_encoding)
                        best_match = distances.argmin()
                        
                        if distances[best_match] <= 0.55:
                            print(f"   🎯 TARGET SPOTTED -> {known_names[best_match]} (Dist: {distances[best_match]:.3f})")
                        else:
                            print(f"   👤 Unknown Subject Detected (Dist: {distances[best_match]:.3f})")
            
            frame_count += 1

    except KeyboardInterrupt:
        print("\n🛑 Live stream terminated safely by user.")
    finally:
        camera_reader.close()

if __name__ == "__main__":
    DATABASE = "/Users/ibrahim/Coding/Facial Regonition Project/known_faces"
    start_live_recognition(DATABASE)