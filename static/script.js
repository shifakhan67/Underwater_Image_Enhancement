document.addEventListener('DOMContentLoaded', () => {
    // Elements Selection
    const photoButton = document.getElementById('photo-button');
    const videoButton = document.getElementById('video-button');
    const photoUpload = document.getElementById('photo-upload');
    const videoUpload = document.getElementById('video-upload');
    const photoOptions = document.getElementById('photo-options');
    const videoOptions = document.getElementById('video-options');
    const originalPhoto = document.getElementById('original-photo');
    const enhancedPhoto = document.getElementById('enhanced-photo');
    const originalVideo = document.getElementById('original-video');
    const enhancedVideo = document.getElementById('enhanced-video');
    const enhancePhotoButton = document.getElementById('enhance-photo-button');
    const enhanceVideoButton = document.getElementById('enhance-video-button');
    const photoEnhancementSelect = document.getElementById('photo-enhancement-select');
    const videoEnhancementSelect = document.getElementById('video-enhancement-select');
    const allProcessedImagesContainer = document.getElementById('all-processed-images');
    const processedImagesGrid = document.getElementById('processed-images-grid');

    // Processed Images Tracking
    let processedImages = [];
    let lastUploadedFile = null;

    // Utility Functions
    function resetUIState() {
        photoOptions.style.display = 'none';
        videoOptions.style.display = 'none';
        allProcessedImagesContainer.style.display = 'none';
        originalPhoto.src = '';
        enhancedPhoto.src = '';
        originalVideo.src = '';
        enhancedVideo.src = '';
        photoEnhancementSelect.selectedIndex = 0;
        videoEnhancementSelect.selectedIndex = 0;
    }

    function renderProcessedImages(images, names) {
        processedImagesGrid.innerHTML = '';
        images.forEach((imagePath, index) => {
            const imageItem = document.createElement('div');
            imageItem.classList.add('processed-image-item');

            const img = document.createElement('img');
            img.src = imagePath;
            
            const caption = document.createElement('h5');
            caption.textContent = names[index] || `Processed Image ${index + 1}`;

            imageItem.appendChild(img);
            imageItem.appendChild(caption);
            processedImagesGrid.appendChild(imageItem);
        });
    }

    // Media Type Selection
    photoButton.addEventListener('click', () => {
        resetUIState();
        photoUpload.click();
    });

    videoButton.addEventListener('click', () => {
        resetUIState();
        videoUpload.click();
    });

    // Photo Upload Event
    photoUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        lastUploadedFile = file;
        
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                originalPhoto.src = event.target.result;
                photoOptions.style.display = 'block';
                videoOptions.style.display = 'none';
                
                processedImages = [{
                    src: event.target.result,
                    method: 'Original Image'
                }];
            };
            reader.readAsDataURL(file);
        }
    });

    // Video Upload Event
    videoUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        lastUploadedFile = file;
        
        if (file) {
            const videoURL = URL.createObjectURL(file);
            originalVideo.src = videoURL;
            
            photoOptions.style.display = 'none';
            videoOptions.style.display = 'block';
        }
    });

    // Photo Enhancement Method Selection
    photoEnhancementSelect.addEventListener('change', () => {
        const selectedMethod = photoEnhancementSelect.value;

        if (selectedMethod === 'show_all') {
            document.getElementById('photo-preview-container').style.display = 'none';
            allProcessedImagesContainer.style.display = 'block';
        } else {
            document.getElementById('photo-preview-container').style.display = 'flex';
            allProcessedImagesContainer.style.display = 'none';
        }
    });

    // Enhance Photo Button
    enhancePhotoButton.addEventListener('click', () => {
        const file = lastUploadedFile;
        const method = photoEnhancementSelect.value;
        
        if (!method) {
            alert('Please select an enhancement method');
            return;
        }

        if (!file) {
            alert('Please upload a photo first');
            return;
        }

        const formData = new FormData();
        formData.append('image', file);
        formData.append('enhancement', method);

        enhancePhotoButton.disabled = true;
        enhancePhotoButton.textContent = 'Processing...';

        fetch('/enhance_photo', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            enhancePhotoButton.disabled = false;
            enhancePhotoButton.textContent = 'Enhance Photo';

            if (data.processed_images) {
                // Show all processed images
                renderProcessedImages(data.processed_images, data.processed_image_names);
                allProcessedImagesContainer.style.display = 'block';
                document.getElementById('photo-preview-container').style.display = 'none';
            } else if (data.enhanced_image) {
                // Show single enhanced image
                enhancedPhoto.src = data.enhanced_image;
                allProcessedImagesContainer.style.display = 'none';
                document.getElementById('photo-preview-container').style.display = 'flex';
            } else {
                alert('Error enhancing image');
            }
        })
        .catch(error => {
            enhancePhotoButton.disabled = false;
            enhancePhotoButton.textContent = 'Enhance Photo';

            console.error('Error:', error);
            alert('An error occurred while processing the image. Please try again.');
        });
    });

    // Enhance Video Button
    enhanceVideoButton.addEventListener('click', () => {
        const file = lastUploadedFile;
        const method = videoEnhancementSelect.value;
        
        if (!method) {
            alert('Please select an enhancement method');
            return;
        }

        if (!file) {
            alert('Please upload a video first');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('method', method);

        enhanceVideoButton.disabled = true;
        enhanceVideoButton.textContent = 'Processing...';

        fetch('/enhance_video', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            enhanceVideoButton.disabled = false;
            enhanceVideoButton.textContent = 'Enhance Video';

            if (data.enhanced) {
                enhancedVideo.src = data.enhanced;
                enhancedVideo.load();
                enhancedVideo.style.display = 'block';
            } else {
                alert('Error enhancing video');
            }
        })
        .catch(error => {
            enhanceVideoButton.disabled = false;
            enhanceVideoButton.textContent = 'Enhance Video';

            console.error('Error:', error);
            alert('An error occurred while processing the video. Please try again.');
        });
    });
});