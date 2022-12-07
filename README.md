# ERD Clustering
 The tool allows database/data science instructors and TAs to cluster ERD (Entity Relationship Diagram) submissions

## To Run: 
1. you may need to run these commands in this order to utilize object detection module (Also, you will need a gpu that works with Cuda 11.3 (we utilize tensorflow-gpu):
   1. cd models/research
   2. protoc object_detection/protos/*.proto --python_out=.
   3. cp object_detection/packages/tf2/setup.py .
   4. python -m pip install .
2. We have also created a requirements.txt which has all of the libraries we utilized for this project.
3. In order to run this project, navigate to directory ERD-Clustering and run the following:
   1. python ERD-Clustering --parameters_txt <path_to_directory_containing_parameters.txt>
      1. Example: python clustering.py --parameters_txt '2_cluster/parameters.txt'

Please do not hesitate to reach out and schedule a meeting. I would be happy to help install this and run this once more! Thank you so much once more for a great project
