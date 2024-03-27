// Global array to store selected frame paths
var selectedFramePaths = [];

document.getElementById('reloadButton').addEventListener('click', function() {
    window.scrollTo({
        top: 0,
        left: 0,
        behavior: 'smooth' // Enables smooth scrolling
    });

    // Increase the delay to ensure the smooth scroll completes
    setTimeout(function() {
        location.reload();
    }, 1000); // Try 1 second, adjust as needed
});




function updateVideoUrl() {
    var videoUrl = document.getElementById('video_url').value;
    document.getElementById('videoPreview').src = videoUrl;
}

function updateProgressBar() {
    fetch('/progress')
        .then(response => response.json())
        .then(data => {
            var progress = data.progress;
            var progressBar = document.getElementById('progressBar');
            if (progressBar) {
                progressBar.style.height = progress + '%';
            }
        });
}

function previewVideo() {
    var preview = document.getElementById('videoPreview');
    var fileInput = document.getElementById('video_file');
    var framesContainer = document.getElementById('framesContainer');
    var extractFramesButton = document.getElementById('extractFramesButton');
    var file = fileInput.files[0];
    var reader = new FileReader();
    
    var allowedExtensions = /\.(mp4|webm)$/i;

    if (file && allowedExtensions.exec(file.name)) {
        reader.onloadend = function () {
            preview.src = reader.result;
            preview.muted = true;
            framesContainer.style.opacity = 1; // Make it appear normal
            framesContainer.style.pointerEvents = 'auto'; // Make it interactive
            // extractFramesButton.style.display = 'block'; // Show the button
            extractFramesButton.style.opacity = 1;
            extractFramesButton.style.pointerEvents = 'auto';
        };
        reader.readAsDataURL(file);
    } else {
        if (file) {
            alert('Invalid file type. Please select an MP4 or MKV video file.');
        }
        fileInput.value = '';
        preview.src = "";
        framesContainer.style.opacity = 0.5; // Keep it greyed out
        framesContainer.style.pointerEvents = 'none'; // Keep it non-interactive
        extractFramesButton.style.opacity = 0.5; // Hide the button
    }
}

document.getElementById('manualFrameSelection').addEventListener('change', function() {
    var framesContainer = document.getElementById('framesContainer');
    var originalRoute = document.getElementById('originalRoute');
    var secondRoute = document.getElementById('secondRoute');
    var mainButton = document.getElementById('mainButton');
    var extractFramesButton = document.getElementById('extractFramesButton');


    if (this.checked) {
        framesContainer.style.display = 'flex';
        originalRoute.style.display = 'none';
        secondRoute.style.display = 'block';
        mainButton.style.display = 'none';
        extractFramesButton.style.display = 'block';
        extractFramesButton.style.opacity = 0.5
        extractFramesButton.style.pointerEvents = 'none';
    } else {
        framesContainer.style.display = 'none';
        originalRoute.style.display = 'block';
        secondRoute.style.display = 'none';
    }
});






function toggleFpsMethodInputs() {
    var selectedMethod = document.querySelector('input[name="fpsMethod"]:checked').value;
    var targetFpsContainer = document.getElementById('targetFpsContainer');
    var uniformFrameRateContainer = document.getElementById('uniformFrameRateContainer');

    if (selectedMethod === 'set_fps') {
        targetFpsContainer.style.display = 'block';
        uniformFrameRateContainer.style.display = 'none';
    } else if (selectedMethod === 'uniform') {
        targetFpsContainer.style.display = 'none';
        uniformFrameRateContainer.style.display = 'block';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var extractFramesButton = document.getElementById('extractFramesButton');

    extractFramesButton.addEventListener('click', function() {
        extractFramesButton.value = '(Re-)Extract Frames';
    });
});


document.getElementById('enableTranscript').addEventListener('change', function() {
    var whisperDiv = document.getElementById('whisperParent');

    if (this.checked) {
        whisperDiv.style.pointerEvents = 'auto';  // Enable interaction
        whisperDiv.style.opacity = '1';           // Make it visually active
    } else {
        whisperDiv.style.pointerEvents = 'none';  // Disable interaction
        whisperDiv.style.opacity = '0.5';         // Make it visually inactive
    }
});
document.getElementById('enableRME').addEventListener('change', function() {
    var whisperDiv = document.getElementById('audioIntensityParent');

    if (this.checked) {
        whisperDiv.style.pointerEvents = 'auto';  // Enable interaction
        whisperDiv.style.opacity = '1';           // Make it visually active
    } else {
        whisperDiv.style.pointerEvents = 'none';  // Disable interaction
        whisperDiv.style.opacity = '0.5';         // Make it visually inactive
    }
});

document.addEventListener('DOMContentLoaded', function () {
    // Function to enable/disable tracking model based on detection model selection
    function toggleTrackingModel(detectionModelId, trackingModelId) {
        var detectionModel = document.getElementById(detectionModelId);
        var trackingModel = document.getElementById(trackingModelId);

        function checkAndToggle() {
            trackingModel.disabled = detectionModel.value !== 'YD1';
        }

        // Event listener for change on detection model dropdown
        detectionModel.addEventListener('change', checkAndToggle);

        // Initial check in case the page is reloaded with a different option pre-selected
        checkAndToggle();
    }

    // Apply the function to both sets of dropdowns
    toggleTrackingModel('detectionModel', 'trackingModel');
    // Assuming the other set of IDs are 'detectionModel3' and 'trackingModel3'
    toggleTrackingModel('detectionModel2', 'trackingModel2');
});

function toggleImageSelection(img, index, framePaths) {
    
    console.log("toggleImageSelection called"); // For debugging: check if function is called

    img.classList.toggle('selected-frame');
    if (img.classList.contains('selected-frame')) {
        selectedFramePaths.push(framePaths[index]);
        enlargeImage(img.src);
    } else {
        var pathIndex = selectedFramePaths.indexOf(framePaths[index]);
        if (pathIndex > -1) {
            selectedFramePaths.splice(pathIndex, 1);
        }
    }
     console.log("Selected Frame Paths:", selectedFramePaths); // Log the current state of selectedFramePaths
}

function enlargeImage(src) {
    document.getElementById('largeImage').src = src;
    document.getElementById('imageModal').style.display = 'flex';
}




document.getElementById('uploadForm').onsubmit = function(event) {
    event.preventDefault();
    var progressBarContainer = document.getElementById('progressBarContainer');
    progressBarContainer.style.display = 'block'; // Make the container visible
    progressBarContainer.classList.add('fadeIn'); // Add the animation class to start the animation

    var formData = new FormData(this);
    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response_frames => response_frames.json())
    .then(data => {
        localStorage.setItem('processedVideoPath', data.processed_video_path);
        // Update the UI with extracted frame images
        if (data.frame_paths && Array.isArray(data.frame_paths)) {
            let framesContainer = document.getElementById('extractFrames');
            framesContainer.innerHTML = '';

            data.frame_paths.forEach(function(framePath, index) {
                let img = document.createElement('img');
                img.src = framePath;
                img.alt = 'Extracted frame';
                img.classList.add('frame-image');
                
                img.addEventListener('click', function() {
                    toggleImageSelection(img, index, data.frame_paths);
                });
                framesContainer.appendChild(img);
                
            });
            } else {
            
            document.getElementById('results').style.display = 'block';
            // Update the video player source with the path of the processed/downloaded video
            if (data.processed_video_path) {
                document.getElementById('videoPreview').src = '/uploads/' + data.processed_video_path.split('/').pop();
            }
            // document.getElementById('processedVideoPathBox').innerHTML = '<strong>Processed Video Path:</strong> ' + data.processed_video_path;
            document.getElementById('finalTranscriptBox').innerHTML = '<strong>Transcript:</strong> ' + data.final_transcript;
            
            document.getElementById('rmsAnimation').src = data.animation_file_path;
            
            // Assuming 'data' contains the results from your Python function
            var framesMessageDet = `<strong>The Detection model ran on ${data.total_frames} number of frames.</strong><br><br>`;
            var framesMessageSeg = `<strong>The Segmentation model ran on ${data.total_frames} number of frames.</strong><br><br>`;

            console.log(framesMessageDet);
            document.getElementById('objectDetectionResult').innerHTML = 
                framesMessageDet +
                '<strong>Number of detected Objects in each class:</strong> ' + '<br>' +
                Object.entries(data.object_counts)
                .map(([key, value]) => `${key}: ${value}`)
                .join('<br>');

            document.getElementById('pitchSegmentationResult').innerHTML = 
                framesMessageSeg +
                '<strong>Number of Segmented Objects in each class:</strong> ' + '<br>' +
                Object.entries(data.segment_counts)
                .map(([key, value]) => `${key}: ${value}`)
                .join('<br>');



            
            document.getElementById('objectTrackingResult').src = data.tracking_path;

            var chatResponseDiv = document.getElementById('chatResponse');
            // chatResponseDiv.innerHTML = '<h3>Chat Response</h3>';
            
            // Check if the chat response contains an error message
            if (data.chat_response && !data.chat_response.startsWith("Tweet:")) {
                // Display the error message
                chatResponseDiv.innerHTML += `<p>Error: ${data.chat_response}</p>`;
            } else {
                // Regular expression to match the tweet format
                var tweetRegex = /Tweet:[\s\S]+?(?=\n\nTweet:|\n\n$)/g;
                var tweets = data.chat_response.match(tweetRegex);
            
                if (tweets) {
                    tweets.forEach(function (tweet) {
                        var tweetBox = document.createElement('div');
                        tweetBox.className = 'tweet-box';
                        tweetBox.textContent = tweet.trim();
                        
                        tweetBox.addEventListener('mouseenter', function () {
                            // Create a custom tooltip element
                            var tooltip = document.createElement('div');
                            tooltip.className = 'tooltip';
                            tooltip.textContent = 'Copy to clipboard';
                            
                            // Append the tooltip to the tweet box
                            tweetBox.appendChild(tooltip);
                
                            // Create a temporary text area to copy the text
                            var tempTextArea = document.createElement('textarea');
                            tempTextArea.style.position = 'fixed';
                            tempTextArea.style.opacity = 0;
                            tempTextArea.value = tweetBox.textContent;
                            document.body.appendChild(tempTextArea);
                
                            // Select the text and copy it to the clipboard
                            tempTextArea.select();
                            document.execCommand('copy');
                
                            // Remove the temporary text area
                            document.body.removeChild(tempTextArea);
                
                        });
                
                        chatResponseDiv.appendChild(tweetBox);
                    });
                } else {
                    chatResponseDiv.innerHTML += '<p>No tweets found in the response.</p>';
                }
            }
            console.log(data.frame_paths);
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
    });

    // Start updating the progress bar
    var progressInterval = setInterval(updateProgressBar, 1000);

    // Stop updating the progress bar when the process is deemed complete
    setTimeout(() => {
        clearInterval(progressInterval);
        if (document.getElementById('progressBar')) {
            document.getElementById('progressBar').style.height = '0%';
        }
    }, 100000); // Adjust this duration as needed
};


document.getElementById('uploadFormSelective').onsubmit = function(event) {
    event.preventDefault();

    // Make the POST request to the '/handle-selected-frames' route
    fetch('/handle-selected-frames', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('results').style.display = 'block';
        
        // Update the video player source with the path of the processed/downloaded video
        if (data.processed_video_path) {
            document.getElementById('videoPreview').src = '/uploads/' + data.processed_video_path.split('/').pop();
        }
        document.getElementById('processedVideoPathBox').innerHTML = '<strong>Processed Video Path:</strong> ' + data.processed_video_path;
        
        // Other response processing...
        document.getElementById('finalTranscriptBox').innerHTML = '<strong>Transcript:</strong> ' + data.final_transcript;
        document.getElementById('rmsAnimation').src = data.animation_file_path;

        var chatResponseDiv = document.getElementById('chatResponse');
            // chatResponseDiv.innerHTML = '<h3>Chat Response</h3>';
            
            // Check if the chat response contains an error message
            if (data.chat_response && !data.chat_response.startsWith("Tweet:")) {
                // Display the error message
                chatResponseDiv.innerHTML += `<p>Error: ${data.chat_response}</p>`;
            } else {
                // Regular expression to match the tweet format
                var tweetRegex = /Tweet:[\s\S]+?(?=\n\nTweet:|\n\n$)/g;
                var tweets = data.chat_response.match(tweetRegex);
            
                if (tweets) {
                    tweets.forEach(function (tweet) {
                        var tweetBox = document.createElement('div');
                        tweetBox.className = 'tweet-box';
                        tweetBox.textContent = tweet.trim();
                        
                        tweetBox.addEventListener('mouseenter', function () {
                            // Create a custom tooltip element
                            var tooltip = document.createElement('div');
                            tooltip.className = 'tooltip';
                            tooltip.textContent = 'Copy to clipboard';
                            
                            // Append the tooltip to the tweet box
                            tweetBox.appendChild(tooltip);
                
                            // Create a temporary text area to copy the text
                            var tempTextArea = document.createElement('textarea');
                            tempTextArea.style.position = 'fixed';
                            tempTextArea.style.opacity = 0;
                            tempTextArea.value = tweetBox.textContent;
                            document.body.appendChild(tempTextArea);
                
                            // Select the text and copy it to the clipboard
                            tempTextArea.select();
                            document.execCommand('copy');
                
                            // Remove the temporary text area
                            document.body.removeChild(tempTextArea);
                
                        });
                
                        chatResponseDiv.appendChild(tweetBox);
                    });
                } else {
                    chatResponseDiv.innerHTML += '<p>No tweets found in the response.</p>';
                }
            }
        console.log(data.frame_paths);
    })
    .catch(error => {
        console.error('Fetch error:', error);
    });

    // Start updating the progress bar
    var progressInterval = setInterval(updateProgressBar, 100000);

    // Stop updating the progress bar when the process is deemed complete
    setTimeout(() => {
        clearInterval(progressInterval);
        if (document.getElementById('progressBar')) {
            document.getElementById('progressBar').style.height = '0%';
        }
    }, 100000); // Adjust this duration as needed
};



document.querySelectorAll('input[name="fpsMethod"]').forEach(function(radio) {
    radio.addEventListener('change', toggleFpsMethodInputs);
});

toggleFpsMethodInputs();


document.getElementById('metadataFile').addEventListener('change', function(event){
    var file = event.target.files[0];
    var formData = new FormData();
    formData.append('metadataFile', file);

    fetch('/process-json', {
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
        if(data.error) {
            console.error('Error:', data.error);
        } else {
            // Define a function to update field value and style
            function updateField(id, value) {
                var element = document.getElementById(id);
                if(element) {
                    element.value = value || '';
                    element.style.backgroundColor = 'rgba(0, 255, 0, 0.2)'; // Transparent green
                }
            }

            // Update each field using the new function
            updateField('metadataDate', data.date);
            updateField('metadataTime', data.time);
            updateField('metadataTeam', data.team ? data.team.value : '');
            updateField('metadataScorer', data.scorer ? data.scorer.value : '');
            updateField('metadataShotType', data['shot type'] ? data['shot type'].value : '');
            // ... Update other fields similarly
        }
    })
    .catch(error => console.error('Error:', error));
});

document.getElementById('metadataFile2').addEventListener('change', function(event){
    var file = event.target.files[0];
    var formData = new FormData();
    formData.append('metadataFile2', file);

    fetch('/process-json', {
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
        if(data.error) {
            console.error('Error:', data.error);
        } else {
            // Define a function to update field value and style
            function updateField(id, value) {
                var element = document.getElementById(id);
                if(element) {
                    element.value = value || '';
                    element.style.backgroundColor = 'rgba(0, 255, 0, 0.2)'; // Transparent green
                }
            }

            // Update each field using the new function
            updateField('metadataDate2', data.date);
            updateField('metadataTime2', data.time);
            updateField('metadataTeam2', data.team ? data.team.value : '');
            updateField('metadataScorer2', data.scorer ? data.scorer.value : '');
            updateField('metadataShotType2', data['shot type'] ? data['shot type'].value : '');
            // ... Update other fields similarly
        }
    })
    .catch(error => console.error('Error:', error));
});




// Example function to send selected frame paths back to the server
function sendSelectedFrames() {
    // Collect user inputs
    var userInputs = {
        //Whisper Model
        whisper_model: document.getElementById('whisperModel2').value,
        //RME Audio
        frame_length: document.getElementById('frameLength2').value,
        hop_length: document.getElementById('hopLength2').value,
        //object detection
        selected_YDM: document.getElementById('detectionModel2').value,
        //segmentation
        selected_YSM: document.getElementById('segmentationModel2').value,
        //Tracking
        selected_YTO: document.getElementById('trackingModel2').value,
        //Metadaata
        metadata_date: document.getElementById('metadataDate2').value,
        metadata_time: document.getElementById('metadataTime2').value,
        metadata_team: document.getElementById('metadataTeam2').value,
        metadata_scorer: document.getElementById('metadataScorer2').value,
        metadata_shot_type: document.getElementById('metadataShotType2').value,

        //openai
        openai_api_key: document.getElementById('openAIApiKey2').value,
        temperature: parseFloat(document.getElementById('temperature2').value),
        max_tokens: document.getElementById('maxTokens2').value,
        top_p: document.getElementById('topP2').value,
        presence_penalty: document.getElementById('presencePenalty2').value,
        frequency_penalty: document.getElementById('frequencyPenalty2').value,
        seed_input: document.getElementById('seed2').value,
        
    };
    var enableTranscriptChecked = document.getElementById('enableTranscript2').checked;

    var enableRME = document.getElementById('enableRME2').checked;

    var processedVideoPath = localStorage.getItem('processedVideoPath');
    // Create an object containing both the frame paths and user inputs
    var dataToSend = {
        selectedFrames: selectedFramePaths,
        userInput: userInputs,
        enableTranscript: enableTranscriptChecked,
        enableRMEAudio: enableRME,
        processedVideo: processedVideoPath
    };

    fetch('/handle-selected-frames', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataToSend)
    })
    .then(response => response.json())
    .then(data => {
        // Update the UI with the response data
        document.getElementById('results').style.display = 'block';
        
        // Update the video player source with the path of the processed/downloaded video
        if (data.processed_video_path) {
            document.getElementById('videoPreview').src = '/uploads/' + data.processed_video_path.split('/').pop();
        }
        // document.getElementById('processedVideoPathBox').innerHTML = '<strong>Processed Video Path:</strong> ' + data.processed_video_path;
        
        // Other response processing...
        document.getElementById('finalTranscriptBox').innerHTML = '<strong>Transcript:</strong> ' + data.final_transcript;
        document.getElementById('rmsAnimation').src = data.animation_file_path;

        var framesMessageDet = `<strong>The Detection model ran on ${data.total_frames} number of frames.</strong><br><br>`;
        var framesMessageSeg = `<strong>The Segmentation model ran on ${data.total_frames} number of frames.</strong><br><br>`;

        document.getElementById('objectDetectionResult').innerHTML = 
            framesMessageDet + 
            '<strong>Number of detected Objects in each class:</strong> ' + '<br>' +
            Object.entries(data.object_counts)
            .map(([key, value]) => `${key}: ${value}`)
            .join('<br>');
        
        document.getElementById('pitchSegmentationResult').innerHTML = 
            framesMessageSeg + 
            '<strong>Number of Segmented Objects in each class:</strong> ' + '<br>' +
            Object.entries(data.segment_counts)
            .map(([key, value]) => `${key}: ${value}`)
            .join('<br>');

        
        document.getElementById('objectTrackingResult').src = data.tracking_path;

        var chatResponseDiv = document.getElementById('chatResponse');
        //chatResponseDiv.innerHTML = '<h3>Chat Response</h3>';
        var tweetRegex = /(.+?#\w+)(?=\s+#|$)/g;
        var tweets = data.chat_response.match(tweetRegex);

        if (tweets) {
            tweets.forEach(function (tweet) {
                var tweetBox = document.createElement('div');
                tweetBox.className = 'tweet-box';
                tweetBox.textContent = tweet.trim();
                
                tweetBox.addEventListener('mouseenter', function () {
                    // Create a custom tooltip element
                    var tooltip = document.createElement('div');
                    tooltip.className = 'tooltip';
                    tooltip.textContent = 'Copy to clipboard';
                    
                    // Append the tooltip to the tweet box
                    tweetBox.appendChild(tooltip);
        
                    // Create a temporary text area to copy the text
                    var tempTextArea = document.createElement('textarea');
                    tempTextArea.style.position = 'fixed';
                    tempTextArea.style.opacity = 0;
                    tempTextArea.value = tweetBox.textContent;
                    document.body.appendChild(tempTextArea);
        
                    // Select the text and copy it to the clipboard
                    tempTextArea.select();
                    document.execCommand('copy');
        
                    // Remove the temporary text area
                    document.body.removeChild(tempTextArea);
        
                });
        
                chatResponseDiv.appendChild(tweetBox);
            });
        } else {
            chatResponseDiv.innerHTML += '<p>No tweets found in the response.</p>';
        }
        console.log('Selected frames and user data submitted:', data);
    })
    .catch(error => {
        console.error('Error submitting data:', error);
    });
}


