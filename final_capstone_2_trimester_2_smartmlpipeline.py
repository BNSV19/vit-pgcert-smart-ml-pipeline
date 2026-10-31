# -*- coding: utf-8 -*-
"""Final_Capstone-2_Trimester-2_SmartMLPipeline.ipynb
#Data Discovery
"""

def checkCategorical(data):
  import numpy as np
  if(len(data.unique())<=0.05*len(data.dropna())):
    return True
  else:
    return False

def checkFloatCategory(data):
  import numpy as np
  import pandas as pd
  if(len((data.dropna()%1).unique())==1):
    return True
  else:
    return False

def checkPersonName(data):
  return True
  pass

def checkId(data):
  return True
  pass

def featureTypeDiscovery(data):
  import pandas as pd
  import numpy as np

  featId=[]
  featType=[]
  featSubType=[]

  for i in data.columns:
    if (data[i].dtype == 'int64'):
      if (checkCategorical(data[i])==True):
        featId.append(1)
        featType.append('Categorical')
        featSubType.append('Number')
      else:
        if(checkId(data[i]) == True):
          featId.append(2)
          featType.append('Numerical')
          featSubType.append('ID')
        else:
          featId.append(3)
          featType.append('Numerical')
          featSubType.append('Integer')

    elif (data[i].dtype == 'float64'):
      if(checkFloatCategory(data[i]) == False):
        featId.append(4)
        featType.append('Numerical')
        featSubType.append('Real')
      else:
        featId.append(10)
        featType.append('Categorical')
        featSubType.append('Number')

    elif (data[i].dtype == 'object'):
      if (checkCategorical(data[i])==True):
        featId.append(5)
        featType.append('Categorical')
        featSubType.append('Thing')
      else:
        if(checkId(data[i]) == True):
          featId.append(6)
          featType.append('Word')
          featSubType.append('ID')
        else:
          if(checkPersonName(data[i])):
            featId.append(7)
            featType.append('Word')
            featSubType.append('Name')
          else:
            featId.append(8)
            featType.append('Word')
            featSubType.append('Real')

    elif (data[i].dtype == 'bool'):
      featId.append(9)
      featType.append('Categorical')
      featSubType.append('Boolean')

  finalFeatCategorization = pd.DataFrame([data.columns,featId,featType,featSubType],index=['Feature Name','Feature Category ID','Feature Type','Feature Subtype'])

  return finalFeatCategorization

"""# Data Cleaning"""

def dfIdClear(data,dataCategory):
  dataTemp = data.copy()
  dataTypeTemp = dataCategory.copy()
  for i in dataCategory.columns:
    if(dataCategory[i][3] == 'ID'):
      dataTemp = dataTemp.drop([dataCategory[i][0]],axis=1)
      dataTypeTemp = dataTypeTemp.drop([i],axis=1)
  return dataTemp,dataTypeTemp,dataCategory

def dfCatThingClear(data,dataCategory):
  from sklearn.preprocessing import LabelEncoder
  dataTemp = data.copy()
  dataTypeTemp = dataCategory.copy()
  for i in dataCategory.columns:
    if(dataCategory[i][2] == 'Categorical' and dataCategory[i][3] != 'Number'):
      lbl = LabelEncoder()
      dataTemp[dataCategory[i][0]] = lbl.fit_transform(dataTemp[dataCategory[i][0]])
      dataTypeTemp[i][3] = "Number"
  return dataTemp,dataTypeTemp,dataCategory

"""# Feature Selection"""

def selectKBestFS(data,yFeat,numFeat):
  from sklearn.feature_selection import SelectKBest, chi2
  dfTemp = data.copy()
  X = dfTemp.drop([yFeat],axis=1)
  Y = dfTemp[yFeat]
  numF = (numFeat % X.shape[1]) if (numFeat % X.shape[1])!=0 else 1
  skBest = SelectKBest(score_func = chi2,k=numF).fit(X,Y)
  selectedColumns = X[X.columns[skBest.get_support()]]
  temp =  pd.concat([selectedColumns,Y],axis=1)
  return temp

"""# ML Models"""

def logisticRegressionModel(data,yFeat,splitRatio,params=['l2',False,1.0,True,1,None,'lbfgs',100,'auto',None]):
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score
  try:
    X = data.drop([yFeat],axis=1)
    Y = data[yFeat]
    xTrain,xTest,yTrain,yTest = train_test_split(X,Y,test_size = splitRatio)

    model = LogisticRegression(penalty=params[0],
                                dual=params[1],
                                C=params[2],
                                fit_intercept=params[3],
                                intercept_scaling=params[4],
                                class_weight=params[5],
                                solver=params[6],
                                max_iter=params[7],
                                multi_class=params[8],
                                l1_ratio=params[9])
    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)

    accScore = accuracy_score(yTest,yPred)

    return model,accScore
  except Exception as e:
    return None,"NahHuh"

def randomForestModel(data,yFeat,splitRatio=0.2,params=[100,"gini",None,2,1,0.0,"sqrt",None,0.0]):
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score

  try:
    X = data.drop([yFeat],axis=1)
    Y = data[yFeat]
    xTrain,xTest,yTrain,yTest = train_test_split(X,Y,test_size = splitRatio)

    model = RandomForestClassifier(n_estimators = params[0],
                                  criterion = params[1],
                                  max_depth = params[2],
                                  min_samples_split = params[3],
                                  min_samples_leaf = params[4],
                                  min_weight_fraction_leaf = params[5],
                                  max_features = params[6],
                                  class_weight = params[7],
                                  ccp_alpha = params[8])

    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)

    accScore = accuracy_score(yTest,yPred)

    return model,accScore

  except Exception as e:
    return None,"NahHuh"

"""# Decision Tree Maker"""

class TreeNode:

  def __init__(self,data,dataStruc,acc,children=[]):
    self.data = data
    self.dataStruc = dataStruc
    self.acc = acc
    self.children = children
    self.parent = None

  def __str__(self,level=0):
    ret = "|" * level + str(self.data) + "[" + str(self.acc) + "]" + "\n"
    for child in self.children:
      ret += child.__str__(level + 1)
    return ret

  def printTree(self,accMax = 0.75,oriHor=False,overrideBigPrint = True):
    if(len(self.levelOrderNavEnd())>128 and overrideBigPrint == True):
      return "Sorry, Too Big"
    from PrettyPrint import PrettyPrintTree
    from colorama import Back,Fore
    if oriHor == True:
      ppt = PrettyPrintTree(lambda x:x.children,
                            lambda x:[x.acc],lambda x:[x.data],
                            orientation=PrettyPrintTree.Horizontal,
                            color=Fore.WHITE + Back.LIGHTBLACK_EX,
                            label_color = Fore.WHITE + Back.BLACK)
    else:
      ppt = PrettyPrintTree(lambda x:x.children,
                            lambda x:x.acc if x.acc<accMax else (Back.BLUE +Fore.WHITE+str(x.acc)),lambda x:[x.data],
                            color=Fore.WHITE + Back.LIGHTBLACK_EX,
                            label_color = Fore.WHITE + Back.BLACK)
    return ppt(self)

  def printTreeAsImg(self):
    from PrettyPrint import PrettyPrintTree
    from colorama import init,Back,Fore
    import matplotlib.pyplot as plt
    plt.figure(figsize=(0,0),dpi=100)
    pltTree = PrettyPrintTree(lambda x:x.children,
                            lambda x:x.acc,
                            lambda x:[x.data],
                            color='',
                            label_color = '',
                            return_instead_of_print=True)
    plt.text(0,1,pltTree(self))
    plt.axis('off')
    plt.savefig('/content/dtree.png', bbox_inches='tight')
    plt.show()


  def addChild(self,TreeNode):
    TreeNode.parent = self
    self.children.append(TreeNode)


  def showParent(self):
    return self.parent.data

  def levelOrderNavEnd(self):
    tempQ = []
    tempEndNodes = []

    tempQ.append(self)

    while(len(tempQ)!= 0):
      n = len(tempQ)

      while(n>0):
        tempP = tempQ[0]
        tempQ.pop(0)

        if(tempP.children == []):
          tempEndNodes.append(tempP)

        for i in range(len(tempP.children)):
          tempQ.append(tempP.children[i]);
        n-=1
    return tempEndNodes

  def backPropagation(self):
    tempArr = []
    for i in self.levelOrderNavEnd():
      currNode = i
      temperArr=[]
      while True:
        temperArr.append(currNode)
        if(currNode.parent != None):
          currNode = currNode.parent
        else:
          break
      tempArr.append(temperArr)

    return tempArr

  def bestAccTree(self):
    tempArr = self.backPropagation()
    max = tempArr[0][0].acc
    maxPos = 0

    for i in range (len(tempArr)):
      if tempArr[i][0].acc > max:
        max = tempArr[i][0].acc
        maxPos = i

    tempTreeData=[]
    for i in tempArr[maxPos]:
      tempTreeData.append(i.data)
    return tempTreeData,max

  def bestAccTreeBranch(self,nBranches=10):
    tempArr = self.backPropagation()
    maxBranches = []

    N = min(len(tempArr),nBranches)

    for i in range(0,N):
      maxLoc = 0
      posLoc = 0
      for j in range(len(tempArr)):
        if tempArr[j][0].acc > maxLoc:
          maxLoc = tempArr[j][0].acc
          posLoc = j
      maxBranches.append(tempArr[posLoc])
      tempArr.remove(tempArr[posLoc])

    maxBranchData=[]
    for i in maxBranches:
      tempBranch = []
      for j in i:
        tempBranch.append(j.data)
      maxBranchData.append(tempBranch)

    maxBranchAcc=[]
    for i in maxBranches:
      maxBranchAcc.append(i[0].acc)

    return maxBranchAcc,maxBranchData,maxBranches

  def printBestAccTreeBranch(self,nBranches=10):
    from PrettyPrint import PrettyPrintTree
    from colorama import Back,Fore
    temp =  self.bestAccTreeBranch(nBranches)
    ppt1 = PrettyPrintTree(lambda x:list(set(x.children).intersection(set([j for set in temp[2] for j in set]))),
                          lambda x:x.acc,
                          lambda x:x.data,
                          color=Fore.WHITE + Back.LIGHTBLACK_EX,
                          label_color = Fore.WHITE + Back.BLACK)
    return ppt1(self)

  def printBestAccTreeBranchColored(self,nBranches):
    from PrettyPrint import PrettyPrintTree
    from colorama import Back,Fore
    global iPBATBC,jPBATBC,tempLenPrev
    temp =  self.bestAccTreeBranch(nBranches)
    colorPal = [Back.BLUE,Back.CYAN,Back.LIGHTBLUE_EX,Back.LIGHTCYAN_EX,Back.LIGHTGREEN_EX,Back.LIGHTMAGENTA_EX,Back.LIGHTRED_EX,Back.LIGHTWHITE_EX,Back.LIGHTYELLOW_EX]
    iPBATBC=0
    jPBATBC=1
    tempLenPrev = 1

    def accColor(x):
      global iPBATBC,jPBATBC,tempLenPrev
      strAcc = ""
      tempLen = len(list(set(x.children).intersection(set([j for set in temp[2] for j in set]))))
      if tempLenPrev > jPBATBC:
        iPBATBC = iPBATBC - 1
        jPBATBC = jPBATBC + tempLenPrev
      if tempLenPrev == jPBATBC:
        iPBATBC = iPBATBC
      else:
        iPBATBC = iPBATBC + 1
        jPBATBC = jPBATBC - tempLenPrev

      tempLenPrev = tempLen
      #print(tempLen,i,j)
      if colorPal[iPBATBC%len(colorPal)] in [Back.LIGHTCYAN_EX,Back.LIGHTWHITE_EX,Back.LIGHTYELLOW_EX]:
        strAcc = (colorPal[iPBATBC%len(colorPal)] + Fore.BLACK + str(x.acc))
      else:
        strAcc = (colorPal[iPBATBC%len(colorPal)] + str(x.acc))
      return strAcc

    ppt1 = PrettyPrintTree(lambda x:list(set(x.children).intersection(set([j for set in temp[2] for j in set]))),
                              lambda x:accColor(x),
                              lambda x:x.data,
                              color=Fore.WHITE + Back.LIGHTBLACK_EX,
                              label_color = Fore.WHITE + Back.BLACK)
    return ppt1(self)

"""# ML Sub Tree Creator"""

def logisticRegressionSubTree(data,yFeat):
  import numpy as np
  import warnings
  from tqdm.notebook import tqdm
  warnings.filterwarnings("ignore")

  baseMLTree = TreeNode('LR-Model',None,0,[])

  ftd = featureTypeDiscovery(data)
  dfC1 = dfIdClear(data,ftd)
  dfC2 = dfCatThingClear(dfC1[0],dfC1[1])

  penalty = ['l1','l2','elasticnet',None] if False else [None]
  dual = [True,False] if False else [False]
  C = [i/100 for i in range(100,1000,100)] if False else [1.0]
  fit_intercept = [True, False]
  intercept_scaling = [i/100 for i in range(100,1000,100)] if False else [1]
  class_weight = ['balanced',None]
  solver = ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']
  max_iter = [i for i in range(100,1000,250)] if False else [100]
  multi_class = ['auto','ovr','multinomial'] if False else ['auto']
  l1_ratio = [i/100 for i in range(1,100,25)] if False else [None]

  final_params_str = ["penalty","dual","C","fit_intercept","intercept_scaling","class_weight","solver","max_iter","multi_class","l1_ratio"]
  final_params = [penalty,dual,C,fit_intercept,intercept_scaling,class_weight,solver,max_iter,multi_class,l1_ratio]
  default_params = np.array(['l2',False,1.0,True,1,None,'lbfgs',100,'auto',None])

  n = len(final_params)
  temp = baseMLTree
  for iTemp in tqdm(range (len(final_params)),desc="Parameter Category",leave=False):
    i = final_params[iTemp]
    iStr = final_params_str[iTemp]
    n = n - 1
    for k in tqdm(temp.levelOrderNavEnd(),desc="Leaf Node",leave=False,nrows=1,position=20):
      if k.acc != -6611:
        for j in tqdm(i,desc="Parameter Value",leave=False,nrows=1,position=40):
          tempParamsArr = None
          model = None
          acc = None
          if(k.dataStruc is None):
            tempParamsArr = np.array([j])
            tempParamsArr = np.append(tempParamsArr,default_params[-n:])
            #print(tempParamsArr)

            #k.addChild(TreeNode(str(iStr)+" "+str(j),tempParamsArr,0,[]))
          else:
            tempParamsArr = np.array(k.dataStruc[:len(final_params)-n-1])
            tempParamsArr = np.append(tempParamsArr,[j])
            tempParamsArr = np.append(tempParamsArr,default_params[-n:])
            #print(tempParamsArr)
            #model,acc = logisticRegressionModel(df,'Exited',0.2,tempParamsArr)
          # k.addChild(TreeNode(str(iStr)+" "+str(j),tempParamsArr,0,[]))
          model,acc = logisticRegressionModel(dfC2[0],'Exited',0.2,tempParamsArr)
          if(acc == "NahHuh"):
            k.addChild(TreeNode(str(iStr)+" "+str(j),tempParamsArr,-6611,[]))
          else:
            k.addChild(TreeNode(str(iStr)+" "+str(j),tempParamsArr,acc,[]))


  return baseMLTree

def randomForestClassifierSubTree(data,yFeat):
  import numpy as np
  import warnings
  from tqdm.notebook import tqdm
  warnings.filterwarnings("ignore")

  baseMLTree = TreeNode('RF-Model',None,0,[])

  ftd = featureTypeDiscovery(data)
  dfC1 = dfIdClear(data,ftd)
  dfC2 = dfCatThingClear(dfC1[0],dfC1[1])

  n_estimators = [i for i in range(100,1000,100)] if False else [100]
  criterion = ["gini","entropy","log_loss"] if False else ["gini"]
  max_depth = [i for i in range(2,100,2)] if False else [None]
  max_depth.append(None)
  min_samples_split = [i for i in range(2,10,1)] if False else [2]
  min_samples_leaf = [i for i in range(1,10,1)] if False else [1]
  min_weight_fraction_leaf = [i/100 for i in range(0,100,1)] if False else [0.0]
  max_features = ["sqrt","log2",None] if False else ["sqrt"]
  max_features.append([i/100 for i in range(1,100,10)] if False else [None])
  class_weight = ["balanced","balanced_subsample",None]
  ccp_alpha = [i/100 for i in range(0,100,10)] if False else [0.0]

  final_params_str = ["n_estimators","criterion","max_depth","min_samples_split","min_samples_leaf","min_weight_fraction_leaf","max_features","class_weight","ccp_alpha"]
  final_params = [n_estimators,criterion,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_features,class_weight,ccp_alpha]
  default_params = np.array([100,"gini",None,2,1,0.0,"sqrt",None,0.0])

  n = len(final_params)
  temp = baseMLTree
  for iTemp in tqdm(range (len(final_params)),desc="Parameter Category",leave=False):
    i = final_params[iTemp]
    iStr = final_params_str[iTemp]
    n = n - 1
    for k in tqdm(temp.levelOrderNavEnd(),desc="Leaf Node",leave=False,nrows=1,position=20):
      if k.acc != -6611:
        for j in tqdm(i,desc="Parameter Value",leave=False,nrows=1,position=40):
          tempParamsArr = None
          model = None
          acc = None
          if(k.dataStruc is None):
            tempParamsArr = np.array([j])
            tempParamsArr = np.append(tempParamsArr,default_params[-n:])
            #print(tempParamsArr)
            #model,acc = randomForestModel(dfC2[0],'Exited',0.2,tempParamsArr)
            #k.addChild(TreeNode(str(iStr)+" "+str(j),tempParamsArr,0,[]))
          else:
            tempParamsArr = np.array(k.dataStruc[:len(final_params)-n-1])
            tempParamsArr = np.append(tempParamsArr,[j])
            tempParamsArr = np.append(tempParamsArr,default_params[-n:])
            #print(tempParamsArr)
            #model,acc = randomForestModel(dfC2[0],'Exited',0.2,tempParamsArr)
            #k.addChild(TreeNode(str(iStr)+" "+str(j),tempParamsArr,0,[]))
          model,acc = randomForestModel(dfC2[0],'Exited',0.2,tempParamsArr)
          if(acc == "NahHuh"):
            k.addChild(TreeNode(str(iStr)+" "+str(j),tempParamsArr,-6611,[]))
          else:
            k.addChild(TreeNode(str(iStr)+" "+str(j),tempParamsArr,acc,[]))


  return baseMLTree

"""#Final Tree Creator"""

def BeanML(data,yFeat):
  from tqdm.notebook import tqdm
  try:
    base = TreeNode('Base',None,0,[])

    ftd = featureTypeDiscovery(data)
    dfC1 = dfIdClear(data,ftd)
    dfC2 = dfCatThingClear(dfC1[0],dfC1[1])

    for i in range(1,dfC2[0].shape[1]):
      skBest = selectKBestFS(dfC2[0],yFeat,i)
      base.addChild(TreeNode('SelectKBest-'+str(i),skBest,0,[]))

    snapshot = base.levelOrderNavEnd()

    for i in tqdm(snapshot,desc="Feature Selection"):
      i.addChild(logisticRegressionSubTree(i.dataStruc,yFeat))
      i.addChild(randomForestClassifierSubTree(i.dataStruc,yFeat))

    return base
  except Exception as e:
    print(e)

"""# Interpretation"""

!pip install PrettyPrintTree

import pandas as pd
df1 = pd.read_csv('/content/drive/MyDrive/Trimester-2/Machine Learning/Copy of Churn_Modelling.csv')

dTree1 = BeanML(df1,'Exited')

print(dTree1)

dTree1.printTree(overrideBigPrint = False)

dTree1.printBestAccTreeBranch(6)

dTree1.printBestAccTreeBranchColored(6)

dTree1.bestAccTree()

from PrettyPrint import PrettyPrintTree
from colorama import Back,Fore

ftd = featureTypeDiscovery(df1)
dfC1 = dfIdClear(df1,ftd)
dfC2 = dfCatThingClear(dfC1[0],dfC1[1])
lr = logisticRegressionModel(data=dfC2[0],yFeat='Exited',splitRatio=0.2)
rf = randomForestModel(data=dfC2[0],yFeat='Exited',splitRatio=0.2)
base = TreeNode("Base",None,0,[])
base.addChild(TreeNode("Logestic Regression",lr[0],lr[1],[]))
base.addChild(TreeNode("Random Forest Classifier",rf[0],rf[1],[]))
PrettyPrintTree(lambda x: x.children,lambda x:x.acc,lambda x:x.data,color=Fore.WHITE + Back.LIGHTBLACK_EX,
                              label_color = Fore.WHITE + Back.BLACK)(base)

def printBestAccTreeBranchColored(self,nBranches):
  from PrettyPrint import PrettyPrintTree
  from colorama import Back,Fore

  temp =  self.bestAccTreeBranch(nBranches)
  colorPal = [Back.BLUE,Back.CYAN,Back.LIGHTBLUE_EX,Back.LIGHTCYAN_EX,Back.LIGHTGREEN_EX,Back.LIGHTMAGENTA_EX,Back.LIGHTRED_EX,Back.LIGHTWHITE_EX,Back.LIGHTYELLOW_EX]
  global iPBATBC,jPBATBC,tempLenPrev
  iPBATBC=0
  jPBATBC=1
  tempLenPrev = 1
  def accColor(x):
    global iPBATBC,jPBATBC,tempLenPrev
    strAcc = ""
    tempLen = len(list(set(x.children).intersection(set([j for set in temp[2] for j in set]))))
    if tempLenPrev > jPBATBC:

      iPBATBC = iPBATBC - 1
      jPBATBC = jPBATBC + tempLenPrev
    if tempLenPrev == jPBATBC:
      iPBATBC = iPBATBC
    else:
      iPBATBC = iPBATBC + 1
      jPBATBC = jPBATBC - tempLenPrev

    tempLenPrev = tempLen
    #print(tempLen,i,j)
    if colorPal[iPBATBC%len(colorPal)] in [Back.LIGHTCYAN_EX,Back.LIGHTWHITE_EX,Back.LIGHTYELLOW_EX]:
      strAcc = (colorPal[iPBATBC%len(colorPal)] + Fore.BLACK + str(x.acc))
    else:
      strAcc = (colorPal[iPBATBC%len(colorPal)] + Fore.BLACK + str(x.acc))
    return strAcc

  ppt1 = PrettyPrintTree(lambda x:list(set(x.children).intersection(set([j for set in temp[2] for j in set]))),
                            lambda x:accColor(x),
                            lambda x:x.data,
                            color=Fore.WHITE + Back.LIGHTBLACK_EX,
                            label_color = Fore.WHITE + Back.BLACK)
  return ppt1(self)

printBestAccTreeBranchColored(self=dTree1,nBranches=6)

