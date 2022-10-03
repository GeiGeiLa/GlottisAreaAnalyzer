Your fist acquisition
=====================

Follow these steps:

    #. Connect everything according to :ref:`hardware`.
    #. Ensure all software is installed according to :ref:`software`.
    #. Turn on computer, illumination, and camera.
    #. Open the OpenHSV software.
    #. Click on "Initialize camera". This opens a connection to the camera using the default settings. 
    #. Click on "Start Camera Feed". Now, the camera should stream images as live preview and the audio signals should show the reference signal and an acoustic trace.
    #. Click on "Stop Camera Feed" or use the optional foot switch to stop the recording.
    #. Use one of the sliders below to browse through the acquired images.
    #. Select a region of interest using the two range sliders. The bright blue area indicates the selected files.
    #. You may analyze your data by clicking on "Start Analysis - Glottis Segmentation" after selecting the glottis using the rectangle in the preview window. See also :ref:`Data Analysis`. 
    #. Click on "Save data" to save the audio, video and meta data to the local file system. The frames are downloaded from the camera and stored in conventional formats.

Saving data
-----------

Data is stored in several ways:

+--------------+--------------------------------------------------+
| Data source  | File type                                        |
+==============+==================================================+
| Video data   | mp4 (compressed)                                 |
+              +--------------------------------------------------+
|              | mp4 (lossless, recommended for further analysis) |
+--------------+--------------------------------------------------+
| Audio data   | wav (uncompressed)                               |
+--------------+--------------------------------------------------+
| Meta data    | json (human-readable, uncompressed)              |
+--------------+--------------------------------------------------+
| Analysis     | hdf5 (if available)                              |
+--------------+--------------------------------------------------+
| Parameters   | csv (if available)                               |
+--------------+--------------------------------------------------+