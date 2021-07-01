# Data-Analysis-Presentation-Lab-HW2
Course 094295 HW2 submission

Maxim 318828761 | Avi 312485816

Included are all the needed files. Three important notes:
* predict.py downloads 3 files while running: 
 1) 2 pickles - 1 for a classifying model and 1 for a "boxing" model from our google drive (those are pretty heavy - ~800mb together)
 2) ResNet50 model for its architecture (we of course used pretrained=False)

* We assumed that the test path is of the shape "/path/to/folder" (as presented in the hw instructions) and not "/path/to/folder/",

in case it's of the second shape - adjustment in predict.py is needed (at line 116 - just removing the +'/')

* You can find the original Colab notebook from which the report is made here: (the report will contain the link too, as well as to this git repo) 
https://colab.research.google.com/drive/1x0I9yKmsxHXW2uJO6Ba8tXCN-cYHs8LY?usp=sharing

(That is - if you wish to uncover hidden cells, run some things, or just enjoy a slightly better looking report)

Thanks and enjoy the summer :),

Maxim and Avi
