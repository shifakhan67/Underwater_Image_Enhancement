import cv2
import os
from tkinter import Tk, filedialog

# Function to select a video file and process frames
def process_video():
    # Hide the root tkinter window
    root = Tk()
    root.withdraw()
    
    # Open file dialog for selecting video file
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=(("MP4 files", ".mp4"), ("All files", ".*"))
    )
    
    if not video_path:
        print("No file selected.")
        return

    cam = cv2.VideoCapture(video_path)
 
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print("Error: creating Directory of Data")
        return

    currentframe = 0

    while True:
        ret, frame = cam.read()
        if ret:
            # Save frame as a .jpg file
            name = f'../data/frame{currentframe}.jpg'
            print("Creating.. " + name)
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

    # Release the video capture and close any OpenCV windows
    cam.release()
    cv2.destroyAllWindows()
    print("Frame extraction completed.")

# Call the function to start the video processing
process_video()