# MRI-EBD
This is a model for eye blink detection from videos acquired in MRI

# Dependencies
asttokens==2.2.1
backcall==0.2.0
decorator==5.1.1
dlib==19.24.1
executing==1.2.0
imutils==0.5.4
ipython==8.12.2
jedi==0.18.2
matplotlib-inline==0.1.6
numpy==1.24.3
opencv-python==4.7.0.72
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
prompt-toolkit==3.0.38
ptyprocess==0.7.0
pure-eval==0.2.2
Pygments==2.15.1
scipy==1.10.1
six==1.16.0
stack-data==0.6.2
traitlets==5.9.0
typing_extensions==4.6.2
wcwidth==0.2.6

# Use
before running eye_det.py
- Enter video file to read using variable filename (line 18)
- Enter desired output filename in csv format (line 35)
- run script eye_det.py
- select eye contour using the interactive user interface the press enter
- Wait until the end of the landmark detection

# csv file 
the csv file is composed of 14 columns
column 1: frame number
column 2 through 13: xy coordinates of 6 landmarks
column 14: Computed EAR (Eye Aspect Ratio)
