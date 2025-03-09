// app.js

const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const webcamVideo = document.getElementById('webcam');
const gestureOutput = document.getElementById('gestureOutput');

let mediaStream;

// Start webcam
startButton.addEventListener('click', async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamVideo.srcObject = mediaStream;
            gestureOutput.textContent = 'Camera is active. Show your sign language gesture.';
        } catch (err) {
            console.error('Error accessing webcam: ', err);
            gestureOutput.textContent = 'Unable to access webcam. Please check your browser permissions.';
        }
    }
});

// Stop webcam
stopButton.addEventListener('click', () => {
    if (mediaStream) {
        const tracks = mediaStream.getTracks();
        tracks.forEach(track => track.stop());
        gestureOutput.textContent = 'Camera stopped. Click Start to try again.';
        webcamVideo.srcObject = null;
    }
});
