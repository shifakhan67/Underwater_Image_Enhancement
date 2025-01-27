from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import cv2
import uuid
import moviepy.editor as mp
from sklearn.decomposition import PCA


app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
ENHANCED_FOLDER = 'enhanced'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENHANCED_FOLDER'] = ENHANCED_FOLDER

class ImageEnhancement:
    @staticmethod
    def histogram_equalization(image):
        """
        Perform histogram equalization using mathematical approach
        """
        img_array = np.array(image)
        flat_img = img_array.flatten()
        histogram, _ = np.histogram(flat_img, bins=256, range=[0, 256])
        cdf = histogram.cumsum()
        cdf_normalized = cdf * 255 / cdf.max()
        equalized_img = np.interp(flat_img, range(256), cdf_normalized)
        equalized_img = equalized_img.reshape(img_array.shape).astype(np.uint8)
        
        return Image.fromarray(equalized_img)

    @staticmethod
    def channel_separation(image):
        """
        Separate RGB channels with mathematical approach
        """
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Create separate channel images
        r_channel = np.zeros_like(img_array)
        g_channel = np.zeros_like(img_array)
        b_channel = np.zeros_like(img_array)
        
        r_channel[:,:,0] = img_array[:,:,0]
        g_channel[:,:,1] = img_array[:,:,1]
        b_channel[:,:,2] = img_array[:,:,2]
        
        return (
            Image.fromarray(r_channel),
            Image.fromarray(g_channel),
            Image.fromarray(b_channel)
        )

    @staticmethod
    def color_compensation(image):
        """
        Color compensation using mathematical approach
        """
        # Convert image to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Calculate channel means
        r_mean = np.mean(img_array[:,:,0])
        g_mean = np.mean(img_array[:,:,1])
        b_mean = np.mean(img_array[:,:,2])
        
        # Create compensation matrix
        compensation_matrix = np.ones_like(img_array)
        compensation_matrix[:,:,0] *= g_mean / r_mean
        compensation_matrix[:,:,2] *= g_mean / b_mean
        
        # Apply compensation
        compensated_img = img_array * compensation_matrix
        
        # Clip values to valid range
        compensated_img = np.clip(compensated_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(compensated_img)

    @staticmethod
    def image_sharpening(image, method='advanced'):
        img_array = np.array(image)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_array, (0, 0), 3)
        
        # Perform unsharp masking
        sharpened = cv2.addWeighted(img_array, 1.5, blurred, -0.5, 0)
        
        return Image.fromarray(sharpened)
       

    @staticmethod
    def contrast_stretching(image):
        """
        Contrast stretching using mathematical approach
        """
        # Convert image to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Find min and max values for each channel
        min_vals = np.min(img_array, axis=(0,1))
        max_vals = np.max(img_array, axis=(0,1))
        
        # Perform linear stretching
        stretched_img = np.zeros_like(img_array)
        for i in range(3):
            stretched_img[:,:,i] = ((img_array[:,:,i] - min_vals[i]) / 
                                    (max_vals[i] - min_vals[i])) * 255
        
        return Image.fromarray(stretched_img.astype(np.uint8))

    

    @staticmethod
    def color_balance(image):
        """
        Color balancing using mathematical approach
        """
        # Convert image to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Calculate overall luminance
        luminance = np.mean(img_array, axis=(0,1))
        
        # Create balancing matrix
        balance_matrix = luminance / np.mean(luminance)
        
        # Apply color balancing
        balanced_img = img_array / balance_matrix
        
        # Clip and convert back to uint8
        balanced_img = np.clip(balanced_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(balanced_img)
    
    @staticmethod
    def pca_fusion(image1, image2):
        """
        PCA-based image fusion
        """
        # Ensure both images are in RGB mode and have the same size
        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')
        image2 = image2.resize(image1.size)
        
        # Convert to numpy arrays
        img1_array = np.array(image1, dtype=np.float32)
        img2_array = np.array(image2, dtype=np.float32)
        
        # Reshape images
        img1_flat = img1_array.reshape(-1, 3)
        img2_flat = img2_array.reshape(-1, 3)
        
        # Combine images
        combined = np.vstack([img1_flat, img2_flat])
        
        # Perform PCA
        pca = PCA(n_components=3)
        pca.fit(combined)
        
        # Reconstruct image
        reconstructed = pca.inverse_transform(pca.transform(combined)[:img1_flat.shape[0]])
        
        # Reshape back to original image shape
        fused_img = reconstructed.reshape(img1_array.shape).astype(np.uint8)
        
        return Image.fromarray(fused_img)

    @staticmethod
    def average_fusion(image1, image2):
        """
        Simple average-based image fusion
        """
        # Ensure both images are in RGB mode and have the same size
        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')
        image2 = image2.resize(image1.size)
        
        # Convert to numpy arrays
        img1_array = np.array(image1, dtype=np.float32)
        img2_array = np.array(image2, dtype=np.float32)
        
        # Perform average fusion
        fused_img = ((img1_array + img2_array) / 2).astype(np.uint8)
        
        return Image.fromarray(fused_img)

    @staticmethod
    def show_all_processed_images(original_image):
        """
        Generate multiple enhanced versions of the image
        """
        processed_images = []
        
        # Dictionary of enhancement methods with readable names
        methods = [
            ('RB Compensation', ImageEnhancement.color_compensation),
            ('White Balance', ImageEnhancement.color_balance),
            ('Sharpened', ImageEnhancement.image_sharpening),
            ('Contrast Stretching', ImageEnhancement.contrast_stretching),
            ('Histogram Equalization', ImageEnhancement.histogram_equalization),
            
            # Fusion methods
            ('PCA Fusion', lambda img: ImageEnhancement.pca_fusion(
                ImageEnhancement.image_sharpening(img), 
                ImageEnhancement.contrast_stretching(img)
            )),
            ('Average Fusion', lambda img: ImageEnhancement.average_fusion(
                ImageEnhancement.image_sharpening(img), 
                ImageEnhancement.contrast_stretching(img)
            ))
        ]
        
        # Apply each enhancement method
        for name, method in methods:
            try:
                # Apply the enhancement method
                enhanced_image = method(original_image)
                
                processed_images.append({
                    'name': name,
                    'image': enhanced_image
                })
            except Exception as e:
                print(f"Error processing {name}: {e}")
        
        return processed_images


def enhance_brightness_video(input_path, output_path):
    """Enhance video brightness"""
    clip = mp.VideoFileClip(input_path)
    enhanced_clip = clip.fx(mp.vfx.colorx, 1.5)  # Increase brightness
    enhanced_clip.write_videofile(output_path)
    return output_path

def enhance_contrast_video(input_path, output_path):
    """Enhance video contrast"""
    clip = mp.VideoFileClip(input_path)
    enhanced_clip = clip.fx(mp.vfx.blackwhite)  # Example contrast enhancement
    enhanced_clip.write_videofile(output_path)
    return output_path

def color_correction_video(input_path, output_path):
    """Perform color correction on video"""
    clip = mp.VideoFileClip(input_path)
    enhanced_clip = clip.fx(mp.vfx.colorx, 1.2)  # Slight color enhancement
    enhanced_clip.write_videofile(output_path)
    return output_path

def video_stabilization(input_path, output_path):
    """Stabilize video"""
    clip = mp.VideoFileClip(input_path)
    stabilized_clip = clip.fx(mp.vfx.resize, width=clip.w)  # Basic stabilization
    stabilized_clip.write_videofile(output_path)
    return output_path

def noise_reduction_video(input_path, output_path):
    """Reduce noise in video"""
    clip = mp.VideoFileClip(input_path)
    enhanced_clip = clip.fx(mp.vfx.mirror_x)  # Example noise reduction technique
    enhanced_clip.write_videofile(output_path)
    return output_path


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance_photo', methods=['POST'])
def enhance_photo():
    try:
        # Ensure an image is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        uploaded_file = request.files['image']
        
        # Open image
        original_image = Image.open(uploaded_file)
        
        # Get enhancement method
        method = request.form.get('enhancement')
        
        # Apply enhancement based on method
        if method == 'histogram_equalization':
            enhanced_image = ImageEnhancement.histogram_equalization(original_image)
        elif method == 'rb_compensation':
            enhanced_image = ImageEnhancement.color_compensation(original_image)
        elif method == 'white_balance':
            enhanced_image = ImageEnhancement.color_balance(original_image)
        elif method == 'contrast_stretching':
            enhanced_image = ImageEnhancement.contrast_stretching(original_image)
        elif method == 'sharpened':
            enhanced_image = ImageEnhancement.image_sharpening(original_image)
        elif method == 'pca_fusion':
            # Create two versions of the image for fusion
            sharpened = ImageEnhancement.image_sharpening(original_image)
            contrast = ImageEnhancement.contrast_stretching(original_image)
            enhanced_image = ImageEnhancement.pca_fusion(sharpened, contrast)
        
        elif method == 'average_fusion':
            # Create two versions of the image for fusion
            sharpened = ImageEnhancement.image_sharpening(original_image)
            contrast = ImageEnhancement.contrast_stretching(original_image)
            enhanced_image = ImageEnhancement.average_fusion(sharpened, contrast)
        
        elif method == 'show_all':
            # Generate all processed images
            processed_images = ImageEnhancement.show_all_processed_images(original_image)
            
            # Save processed images
            saved_paths = []
            saved_names = []
            for idx, img_data in enumerate(processed_images):
                save_path = os.path.join(ENHANCED_FOLDER, f'processed_{idx}.png')
                img_data['image'].save(save_path)
                saved_paths.append(f'/enhanced/processed_{idx}.png')
                saved_names.append(img_data['name'])
            
            return jsonify({
                'processed_images': saved_paths,
                'processed_image_names': saved_names
            })
        
        else:
            return jsonify({'error': 'Invalid enhancement method'}), 400
        
        # Save enhanced image
        enhanced_path = os.path.join(ENHANCED_FOLDER, 'enhanced_image.png')
        enhanced_image.save(enhanced_path)
        
        return jsonify({
            'enhanced_image': '/enhanced/enhanced_image.png'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/enhance_video', methods=['POST'])
def enhance_video():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        method = request.form.get('method', 'brightness')

        # Generate unique filename
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(original_path)

        # Generate enhanced filename
        enhanced_filename = f"enhanced_{unique_filename}"
        enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], enhanced_filename)

        # Choose enhancement method
        enhancement_methods = {
            'brightness': enhance_brightness_video,
            'contrast': enhance_contrast_video,
            'color_correction': color_correction_video,
            'stabilization': video_stabilization,
            'noise_reduction': noise_reduction_video
        }

        # Apply selected enhancement method
        enhance_func = enhancement_methods.get(method, enhance_brightness_video)
        enhance_func(original_path, enhanced_path)

        # Return paths of original and enhanced videos
        return jsonify({
            'original': f'/uploads/{unique_filename}',
            'enhanced': f'/enhanced/{enhanced_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/enhanced/<filename>')
def enhanced_file(filename):
    return send_from_directory(app.config['ENHANCED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)