# # import os

# # file_path = r'G:/Videos/Frames'
# # print(os.path.exists(file_path))

# # import cv2

# # img = cv2.imread(r'G:/Videos/Frames/Example/frame0.jpg')
# # if img is None:
# #     print("Failed to load image.")
# # else:
# #     print("Image loaded successfully.")



import cv2
import os

def images_to_video(folder_path, output_video_path, fps):
    
    images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  

   
    if not images:
        raise ValueError("No images found in the specified folder.")

    
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    
    for image in images:
        image_path = os.path.join(folder_path, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    
    video.release()
    print(f"Video created successfully: {output_video_path}")


folder_path = r"Videos/Videos/data"  
output_video_path = r"Videos/Videos/output_video/"
fps = 30  

images_to_video(folder_path, output_video_path, fps)
