Intro:
  1. This project is to build a Multi-Layer Perceptron (MLP) feed-forward network style of artificial neural network classifier, with Back-Propagation plus Gradient Descent induction learning. The project will include a Classified Set of feature vectors with 8 possible classes. The Training Set will be about 60 percentage of these vectors. A single MLP will be constructed with one hidden layer including 13 nodes (one for bias) and one multi-class output layer including 8 nodes. It also has 10 inputs plus one for bias. Python is used to implement the project – there is no visualization component to this project.
  2. The MLP architecture:
	- Two matrices [11x12] and [13x8] for saving weights, each weigh is set to a random number. The bias input for hidden layer is set to -50, and the bias input for output layer is set to 1.5
	- Logistic fucntion is used for calculatinig the gradient
	- Learning rate is set to 10% (0.1) for back-propagation, and is descreased over time by exponential decrease rate with alpha = 0.1
	- Detect local minima by calculating MSE every epoch with epsilon = 0.001
	- Escape local minima by take a random jump

Contents: Files Included:
  README.txt -- This file.
  mlp.py -- The main python file for initiazling, training, and outputing
  classified_set.txt -- Input file copied from the .pdf file professor provided
  validation_set.txt -- Input file copied from the .pdf file professor provided
  mlp_architecture.png -- Image file to show the MLP architecture  

Installation:
  0. This zip includes all neccessary files
  1. Unzip the zip file into a folder.
  2. Install python 3.8 or higher in the system. E.g, in linux:
	$ sudo apt-get install python 3.8
  3. Install pip for python 3. E.g, in linux:
	$ sudo apt install python3-pip
  4. Install numpy. E.g, in linux:
	$ pip install numpy
  5. Cd to the folder unzipped above
  6. Run the mlp.py file:
	$ python mlp.py

Feature:
  - It runs 150,000 epochs to adjust the weights, and it takes about 45 seconds to finish in the computer using AMD Ryzen 5 1600 CPU
  - Decrease this value may help the program run faster, but the result may be worse because MLP is not trained enough.
  
Bugs:
  - It has "RuntimeWarning: overflow encountered in exp" because we used float 64 for saving weights which can't handle a number as small as exp(-1300). However, it is rounded to zero and doesn't cause any problem. To prevent this unneccessary warning, we use one line code to ignore it (line #4), and this line can be deleted.
 
References:
- Lecture files week 10, 11, 12,... which Professor Siska provided. They are used as the basic concepts of this project.
- How to build a simple neural network in 9 lines of Python code:
https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
- How to build a multi-layered neural network in Python:
https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a
- Backpropagation
https://www.cse.unsw.edu.au/~cs9417ml/MLP2/BackPropagation.html
- Numpy documentation
https://thispointer.com/numpy-array-tutorials/
- Python : How to add / append key value pairs in dictionary
https://thispointer.com/python-how-to-add-append-key-value-pairs-in-dictionary-using-dict-update/
- Mean Squared Error in Numpy
https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy
- Getting the index of the returned max or min item using max()/min() on a list
https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
- Drawing neural networks
https://softwarerecs.stackexchange.com/questions/47841/drawing-neural-networks