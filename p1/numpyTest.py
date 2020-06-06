from numpy import *

a = random.rand(4,4)
#print(a)
randMat = mat(a)
print(randMat)
invRandmat = randMat.I
print("\n")
print(invRandmat)
print("\n")
myEye = randMat*invRandmat
print(myEye)
print("\n")
print(myEye - eye(4))