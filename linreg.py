#
#Gowtham Kommineni
#
# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
#
# TODO: Write this.
#
# Takes the yx file as input, where on each line y is the first element
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np
from numpy.linalg import inv

from pyspark import SparkContext


if __name__ == "__main__":
  if len(sys.argv) !=2:
    print(sys.stderr, "Usage: linreg <datafile>")
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength  
  def calcA(line):
      # Adding 1 to the first column of X
      line[0]=1     			
      X=np.array([line], dtype=float)
      return np.dot(X.transpose(),X)

  def calcB(line):
      # Adding 1 to the first column of X
      y=float(line[0])
      line[0]=1
      X=np.array([line], dtype=float)
      return X.transpose()*y
  ##########	
  #beta=A*B
  #A=INV(XT*X)
  #B=XT*y
  #########
  
  #Calculating A by adding all the matrices from the map func	
  A=inv(yxlines.map(calcA).reduce(lambda x,y: x+y))  #Calculating B by adding all the matrices from the map func
  B=yxlines.map(calcB).reduce(lambda x,y: x+y)
  beta=np.dot(A,B)
  # print the linear regression coefficients in desired output format
  print("beta: ")
  for coeff in beta:
      print(coeff)
  sc.stop()
