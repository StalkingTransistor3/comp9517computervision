# COMP9517 Group Project
For this assignment, we chose to implement several methods, compare them, and submit the best one as our final algorithm. 
These implementations included the YOLOv8 algorithm, ResNet50, Faster R-CNN, and the CNN algorithm.
The algorithm with the best results was the YOLOv8 algorithm, hence we chose to submit that as our model.

## Instructions for running the algorithm
When running the YOLOv8 algorithm, you can choose to train a new model, or use an existing set of weights.
NOTE: the program is designed to run on windows.
To train a new model, when in the directory of the program, run the command (replace the path with the correct system path to the penguins
vs turtles dataset)

`py .\project_yolo.py path/to/penguins_vs_turtles/dataset`

To only format the dataset for yolo without training and validating a new model, run the command

`py .\project_yolo.py path/to/penguins_vs_turtles/dataset -f`

To test a set of weights that have already been trained, run the command
NOTE: this requires an already formatted database. to do this, run the above command.

`py .\project_yolo.py path/to/weights.pt`

A set of already trained weights is included with this submission. To validate with those weights,
after formatting the database, run the command

`py .\project_yolo.py .\pretrained_example.pt`