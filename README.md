# Tracking Task


For the problem of multiple object tracking I used YOLOV4 as object detector and fed this detections to the Deep SORT tracker.

I provide a colab environmet shows the solution steps, But if you want to run it locally set up the following dependecis, download YOLOV4 pre-trained weights and run the save_model script to convert darknet model to the tensor flow graphs then object_tracker to do the tracking.



### Requirments

```bash
# TensorFlow GPU
pip install -r requirements-gpu.txt
```


## YOLOv4 Pre-trained Weights
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
Note:Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.


## Running the Tracker with YOLOv4

```bash
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on video
python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/tracker.avi --model yolov4

```

## Discussion
the multiple object tracking(MOT) problem has two main steps:
-Detection: where all objects are detected in frames
-Association: once we have detections for the frame, a matching is performed for similar detections with respesct to previous frame.

for the detection used YOLO, which is detector applying single neural network predict the bounding boxes, multi-label classification.
YOLO was trained using COCO dataset which has 80 classes, in this task we only concern about the cars,person,trucks,buses.

object tracking which is process of locating moving objects over time in sequence of frames, involves: tacking initial set of object detections, create unique ID for each of the detections, tracking the object over time, maintaining the ID assignment.
for tracking used Deep SORT which has high rate for real time tracking methods as it use the motion measurments and the appearance features through Kalman filter work frame.

So, YOLO and Deep sort work through 3 steps:
-Object detection and recognition
-motion predection and feature generation
-Tracking

## The Bonus requirments
-for the objects moving in the wrong direction I implemented a simple mathematical logic to check wether the object has an ID (known to the tracker) or first time to track it, then check the moving direction of the bounding box by comparing the Y-axis values to check the direction of the motion, it works fine for large scale objects but for persons it needs more tuning, it can be improved by making the compariosn through each 10 frames for examples not frame by frame. in the output video I used red boxes and "!" to show that this object moves in wrong direction, the correct direction is the Y pixel increasing co-ordinate.
Note that this scene is not for a well organized road it was difficult to choose certain direction, as the objects were moving in the 2 direction nearly with the same intensity so the output may seems messy somehow!

-for the accident detection, I followed a sequence depend on two steps first  use the tracking to know the difference between the object motion in 2 frames and compare it to a tunable value (somehow now the motion speed according to pixels), then divide the number of objects that move slowely to the total number of objects and also compare this value to a tunable value and according to it decide if the traffic is heavey (may be there is an accident).
ofcourse it's not the most accurate model for this critcal mission, but the accident detection systems as I know need to have a data collection phase then train the model according to it as it differ with the statues of the road (if it has intersections it will cause crash, if it is high way so it will have high speed accidents and so on)
this model has many limits as the person is object and also the car is object but we can't compare their motion to the same value ! so it show many defects but it can be tuned for certain scene so it work for it in accepted way with considiration of limitation.

## Challenges, limitations, enhancments:
Currently I prepare a report to discuss them in organized way but let me share some ideas before forget them.

-YOLO is trained for 80 objects as I mentioned before, But unfortunately  auto rickshaw (toktok) isn't one of them so the detection for it was some times truck another times car and so on, it's just an example about what I want to say: during the system deployment we always need to collect more data acoording to our scene to enhance our model performance (make our custome dataset)

-The scene itself is somehow complicated to implement many ideas to it, for example lane detection which can make direction detection easier! so as mentioned before it need to has it's own data set according to it's unique conditions.

-Hardware problems for this type of tracking videos for long time will ofcourse appear for long time videos or live tracking,we need to optimize our solution use a weights file related to our objects only and use techniques to reduce the frames rate without having effect on the detection and tracking. 

### References  

The original repo that helped me (AI Guy): https://github.com/theAIGuysCode/yolov4-deepsort
The implementation of deep sort: https://github.com/nwojke/deep_sort
youtube tutorials:
 -AI guy the author of the original repo that I followed: https://www.youtube.com/watch?v=_zrNUzDS8Zc&list=PLKHYJbyeQ1a3tMm-Wm6YLRzfW1UmwdUIN&index=10&ab_channel=TheAIGuy
  -eMaster Class Academy explain the code line by line: https://www.youtube.com/watch?v=zi-62z-3c4U&t=4384s&ab_channel=eMasterClassAcademy
  -Cyrill stanchis, the man who made Kalman filter easy to understand! : https://www.youtube.com/watch?v=E-6paM_Iwfc&t=3272s&ab_channel=CyrillStachniss
  -good article to explain deep sort: https://nanonets.com/blog/object-tracking-deepsort/
