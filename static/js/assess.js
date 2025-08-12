// JavaScript for the Assess page functionality

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const imageUpload = document.getElementById('imageUpload');
    const captureBtn = document.getElementById('captureBtn');
    const imagePreview = document.getElementById('imagePreview');
    const injuryDescription = document.getElementById('injuryDescription');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultSection = document.getElementById('resultSection');
    const injuryTypeSpan = document.querySelector('#injuryType span');
    const precautionsList = document.getElementById('precautionsList');
    const medicationsList = document.getElementById('medicationsList');
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    
    // Variables
    let capturedImage = null;
    let videoStream = null;
    let reportId = null;
    
    // Event Listeners
    imageUpload.addEventListener('change', handleImageUpload);
    captureBtn.addEventListener('click', toggleWebcam);
    analyzeBtn.addEventListener('click', analyzeInjury);
    downloadReportBtn.addEventListener('click', downloadReport);
    
    // Functions
    function handleImageUpload(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(event) {
                displayImage(event.target.result);
                capturedImage = file;
            };
            reader.readAsDataURL(file);
        }
    }
    
    function displayImage(src) {
        // Clear any existing content
        imagePreview.innerHTML = '';
        
        // Create and add image
        const img = document.createElement('img');
        img.src = src;
        imagePreview.appendChild(img);
        
        // If webcam is active, stop it
        stopWebcam();
    }
    
    function toggleWebcam() {
        if (videoStream) {
            stopWebcam();
            captureBtn.textContent = 'Capture from Webcam';
        } else {
            startWebcam();
            captureBtn.textContent = 'Take Photo';
        }
    }
    
    function startWebcam() {
        // Clear preview
        imagePreview.innerHTML = '';
        
        // Create video element
        const video = document.createElement('video');
        video.setAttribute('autoplay', '');
        video.setAttribute('playsinline', '');
        imagePreview.appendChild(video);
        
        // Access webcam
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function(stream) {
                video.srcObject = stream;
                videoStream = stream;
                
                // Add capture button inside preview
                const capturePhotoBtn = document.createElement('button');
                capturePhotoBtn.textContent = 'Capture';
                capturePhotoBtn.className = 'capture-photo-btn';
                capturePhotoBtn.addEventListener('click', capturePhoto);
                imagePreview.appendChild(capturePhotoBtn);
            })
            .catch(function(error) {
                console.error('Error accessing webcam:', error);
                imagePreview.innerHTML = '<p>Error accessing webcam. Please check permissions.</p>';
            });
    }
    
    function stopWebcam() {
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
            
            // Update button text
            captureBtn.textContent = 'Capture from Webcam';
        }
    }
    
    function capturePhoto() {
        if (videoStream) {
            // Create canvas to capture frame
            const video = document.querySelector('video');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert to blob for upload
            canvas.toBlob(function(blob) {
                capturedImage = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
                displayImage(canvas.toDataURL('image/jpeg'));
            }, 'image/jpeg');
            
            // Stop webcam
            stopWebcam();
        }
    }
    
    function analyzeInjury() {
        if (!capturedImage) {
            alert('Please upload or capture an image first.');
            return;
        }
        
        const description = injuryDescription.value.trim();
        if (!description) {
            alert('Please describe how the injury happened.');
            return;
        }
        
        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        
        // Create form data
        const formData = new FormData();
        formData.append('image', capturedImage);
        formData.append('description', description);
        
        // Send to server
        fetch('/api/predict', {  
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
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during analysis. Please try again.');
        })
        .finally(() => {
            // Reset button
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Injury';
        });
    }
    
    function displayResults(data) {
        // Display injury type
        if (data.cropped_image) {
            document.getElementById('croppedImage').src = `data:image/jpeg;base64,${data.cropped_image}`;
        }
        injuryTypeSpan.textContent = data.injury_type.charAt(0).toUpperCase() + data.injury_type.slice(1);
        
        // Set color based on injury type
        const colors = {
            'abrasion': '#FF9800', // Orange
            'bruise': '#9C27B0',   // Purple
            'burn': '#F44336',     // Red
            'cut': '#E91E63',      // Pink
            'ulcer': '#795548'     // Brown
        };
        injuryTypeSpan.style.color = colors[data.injury_type] || '#3B82F6';
        
        // Display precautions
        precautionsList.innerHTML = '';
        data.precautions.forEach(precaution => {
            const li = document.createElement('li');
            li.textContent = precaution;
            precautionsList.appendChild(li);
        });
        
        // Display medications
        medicationsList.innerHTML = '';
        data.medications.forEach(medication => {
            const li = document.createElement('li');
            li.textContent = medication;
            medicationsList.appendChild(li);
        });
        
        // Store report ID for download
        reportId = data.report_id;
        
        // Show results section
        resultSection.style.display = 'block';
        
        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function downloadReport() {
        if (reportId) {
            window.location.href = `/api/download-report/${reportId}`;  // Updated path
        } else {
            alert('No report available. Please analyze an injury first.');
        }
    }

});
