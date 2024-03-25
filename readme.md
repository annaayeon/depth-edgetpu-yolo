
## Running Edge TPU YOLO with RealSense Depth Camera D435i

This repository extends the capabilities of the Edge TPU YOLO object detection model by integrating support for the RealSense Depth Camera D435i. This integration allows for the combination of high-accuracy object detection with depth sensing, enabling applications that require spatial awareness and precise object localization within the environment.

### Key Features:
- **Real-Time Object Detection**: Leverages the power of the Edge TPU to perform fast and accurate object detection in real-time video streams.
- **Depth Sensing**: Utilizes the RealSense Depth Camera D435i to obtain depth information for each detected object, providing their distance from the camera.
- **Spatial Awareness**: Enhances applications with the ability to understand the scene spatially, which is crucial for navigation, obstacle avoidance, and interactive projects.

### Setup and Requirements:
To use this repository, you will need:
- A Google Coral Edge TPU device.
- A RealSense Depth Camera D435i.
- The latest version of the Edge TPU runtime and API libraries installed on your system.
- The `pyrealsense2` library to interface with the RealSense camera.

### Quick Start Guide:
1. **Connect Your Devices**: Ensure your Google Coral device and RealSense Depth Camera D435i are connected to your computer.
2. **Install Dependencies**: Install the necessary libraries and dependencies as mentioned in the Setup section.
3. **Run the Application**: Navigate to the repository's root directory and execute the provided script to start object detection with depth sensing:
   ```shell
   python3 detect.py
   ```
4. **View the Results**: The application will display the video feed with detected objects and their distances labeled in the output window.

For detailed instructions on installation, configuration, and customization, please refer to the [Installation](#installation) and [Configuration](#configuration) sections below.
