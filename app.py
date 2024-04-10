import numpy as np
import gradio as gr
import cv2 as cv
import matplotlib.pyplot as plt
import os
import pickle
from sklearn import tree
import pandas as pd
from time import time
from functools import reduce
with open("model.pkl",'rb') as f:
    clf=pickle.load(f)

def read(im_or):
  t1=time()
  gray = cv.cvtColor(im_or, cv.COLOR_BGR2GRAY)
  thresh_98,im_1 = cv.threshold(gray, 100, 255, cv.THRESH_OTSU)
  im_2=cv.copyMakeBorder(src=im_1, top=15, bottom=15, left=15, right=15, borderType=cv.BORDER_CONSTANT,value=0)
  contours_98, hierarchy_98  = cv.findContours(im_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#   contours_98 =list(contours_98)
  t=0
  ans=[]
  while(t!=-1):
    ans.append(checkdigit(contours_98[t]))
    t=hierarchy_98[0][t][0]
  if(len(ans)==0):return ""
  ans.sort(key= lambda x:x[0])
  return reduce(lambda x,y:x+y[1],ans,"")
def checkdigit(cnt):
    if cv.contourArea(cnt)<30:return (2000,"") 
    mom=cv.moments(cnt)
    Preds=clf.predict([pd.Series(mom)])
    return (int(mom['m10']/mom['m00']),str(Preds[0]))

demo = gr.Interface(read, gr.Image(), "text")
if __name__ == "__main__":
    demo.launch()
