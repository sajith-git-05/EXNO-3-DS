## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/3e2c3f13-f360-4e5a-8d61-5169eafcb98d)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/e2f1474f-714b-4ef2-afaf-9b5fdf23142b)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/8685f7fe-859b-41e5-91b8-afba55e70a0e)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/becbc5d4-43c7-4acc-a4a6-af477227c14f)
```
from sklearn.preprocessing import OneHotEncoder
df
```
![image](https://github.com/user-attachments/assets/1790e1d1-fa0e-4f83-9d6f-0c94bd026e59)

```
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```

![image](https://github.com/user-attachments/assets/509ad985-3c54-4557-b14f-e996f45cace2)
```
df2=pd.concat([df,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/c13488c8-4cce-46e6-97d8-4ac2443fc3f1)
```
from category_encoders import BinaryEncoder
df=pd.read_csv(r"C:\Users\admin\Downloads\data (1).csv")
df
```
![image](https://github.com/user-attachments/assets/e9a795e9-ed1d-4fa8-a035-e6cfa448cefb)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

![image](https://github.com/user-attachments/assets/3a9c7c9d-880d-46f9-aea7-b828222df16d)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```

![image](https://github.com/user-attachments/assets/79b55d26-9b96-49d6-9cec-99d6042b4d86)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv(r"C:\Users\admin\Downloads\Data_to_Transform.csv")
df.skew()
```

![image](https://github.com/user-attachments/assets/5e2f6ae6-8f4b-419d-bd1e-ec68b094344a)
```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/cf2c15d2-71d9-4796-9ace-c289c29ac2d4)
```
np.reciprocal(df["Highly Positive Skew"])

```
![image](https://github.com/user-attachments/assets/e6df7e7d-d7b0-4f92-8b46-8304ea2683b7)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/39763fde-32b5-4142-989a-484e1a77e65d)

```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/1f3bdd91-ba0a-4afc-9e65-26b0e8beb4bd)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/23d6d599-870a-4e45-bbe1-d9df16135fc7)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/ac88de47-6981-49bf-bf42-358400e3ce16)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/user-attachments/assets/d3ce8b2c-565c-4147-a9a4-c2b2f1bb98bf)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/65788724-304e-4196-9b20-0d4124d0f0ed)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/027b670a-4a64-4150-88e1-c2e4a2a4eee1)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/fef2e34b-1e4f-4b75-acc1-f799ab618040)


# RESULT:
Thus the code for Data Transformation is executed successfully.


       
