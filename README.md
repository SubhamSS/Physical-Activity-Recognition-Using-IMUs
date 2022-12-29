# Physical-Activity-Recognition-Using-IMUs
Raw dataset used:
https://archive.ics.uci.edu/ml/machine-learning-databases/00231/

To run the code and get results, add all 6 following .py files to an environment:
	1. preprocess_1.py 
	2. main.py
	3. main_ts_l2.py
	4. time_series_feat.py
	5. ts_prep.py
	6. ts_eval.py

For Preprocessing,

In preprocess_1.py,
In lines 39 and 40, change the path to load the dataset
In lines 96, change path to write the preprocessed dataset csv file to desired path. This can be commented out as well.

For intensity based classification:

Change the classifier by giving the function get_model the right argument in line 191 of main.py
Change the pathname to the desired path in line 282 of main.py
Optional: Change the pathname to the desired path in line 180 of main.py if the data frame from preprocess_1.py is saved

Activity based classification:

In main_ts_l2.py,
In line 16, Change the path to read the intensity classified dataset obtained using main.py 
In line 20, select the classifier; SVM or Random Forest
In line 25, change the overlap percentages 
In line 29, change the folder path to save the confusion matrices and time series preprocessed datasets.
Run

For the heart rate regression model:
In the main.py file
Ucomment indicated parts in the following functions:
									main()
									within_subject()
									between_subject()
Change the last argument of evaluation() to 1 on lines 96 and 139 in main.py
