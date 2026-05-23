import dlib
import face_recognition
import warnings

# Silence the cosmetic setuptools warning we patched earlier
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

def verify_person(profile_pic_path, target_scene_path, target_name):
    print("✨ System Initialized.")
    print(f"Using dlib version: {dlib.__version__}")
    print("─" * 50)

    # 1. Load your profile picture and calculate its 128-dimensional facial embedding vector
    print("⏳ Processing known profile image...")
    known_image = face_recognition.load_image_file(profile_pic_path)
    known_encodings = face_recognition.face_encodings(known_image)

    if len(known_encodings) == 0:
        print("❌ Error: No face detected in the profile picture. Use a sharper image.")
        return

    # Grab the first face math representation found in your profile pic
    profile_face_vector = known_encodings[0]
    print(f"✅ Facial metrics calculated and registered for: {target_name}")
    print("─" * 50)

    # 2. Load the scene image where you want to scan for yourself
    print(f"⏳ Scanning target file: '{target_scene_path}'...")
    scene_image = face_recognition.load_image_file(target_scene_path)
    
    # Locate bounding box coordinates (Top, Right, Bottom, Left pixels) for all faces present
    face_locations = face_recognition.face_locations(scene_image)
    
    # Compute the mathematical vector maps for each face found in the scene
    scene_face_encodings = face_recognition.face_encodings(scene_image, face_locations)

    total_faces = len(face_locations)
    print(f"🔍 Scan complete. Found {total_faces} face structural region(s) in the image.")

    # 3. Compare the vectors to find a match
    match_spotted = False

    for index, (coords, scene_encoding) in enumerate(zip(face_locations, scene_face_encodings)):
        top, right, bottom, left = coords
        
        # compare_faces evaluates vector Euclidean distance under the hood. 
        # tolerance=0.6 is the default limit. Stricter checking = lower number (e.g., 0.5)
        matches = face_recognition.compare_faces([profile_face_vector], scene_encoding, tolerance=0.6)
        
        # Calculate raw facial distance metric (Lower value = closer match, 0.0 is an identical pixel match)
        face_distance = face_recognition.face_distance([profile_face_vector], scene_encoding)[0]

        print(f"\nEvaluating Face #{index + 1}:")
        print(f"   📐 Pixel Coordinates -> Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")
        print(f"   📊 Structural Distance Score: {face_distance:.4f}")

        if matches[0]:
            print(f"   🎯 TARGET CONFIRMED: This is {target_name}!")
            match_spotted = True
        else:
            print("   👤 Result: Unknown Person (Distance threshold not met).")

    print("─" * 50)
    if not match_spotted:
        print(f"❌ Verification complete: {target_name} was NOT detected in the target image.")
    else:
        print(f"🎉 Success: {target_name} was verified successfully.")

# --- EXECUTION ---
if __name__ == "__main__":
    verify_person(
            profile_pic_path="/Users/ibrahim/Coding/Facial Regonition Project/me.jpg", 
            target_scene_path="/Users/ibrahim/Coding/Facial Regonition Project/group.jpg", 
            target_name="Simi"
        )