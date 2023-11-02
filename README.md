# multiclass-classifier

We have a dataset that collected by Mr.Parham ghadermazi 
(Email: parham.ghadermazi@gmail.com)
<br >
Nima Rahmani from Kurdistan University wrote the machine learning codes
(Email: nimarahmani2012@gmail.com)

this dataset have 9 columns, one of them is “Country” which is useless, seven columns which are our features and one last column “RES” which is the label of our dataset
we would use machine learning models and deep neural network (DNN) and train them on feature to predict the labels.

At first we drop the “Country” column and extract the “RES” column into a variable which is our labels.

Now we have to normalize our dataset, let’s see our dataset describe before normalize.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Outside Dry-Bulb Temperature</th>
      <th>Outside Dew-Point Temperature</th>
      <th>Direct Normal Solar</th>
      <th>Diffuse Horizontal Solar</th>
      <th>Wind Speed</th>
      <th>Wind Direction</th>
      <th>Atmospheric Pressure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>109.000000</td>
      <td>109.000000</td>
      <td>109.000000</td>
      <td>109.000000</td>
      <td>109.000000</td>
      <td>109.000000</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>21.070910</td>
      <td>13.318540</td>
      <td>4.242275</td>
      <td>2.451578</td>
      <td>3.158868</td>
      <td>163.070337</td>
      <td>97031.062294</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.893768</td>
      <td>6.227864</td>
      <td>3.290580</td>
      <td>0.797008</td>
      <td>1.431860</td>
      <td>71.114429</td>
      <td>5474.019179</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.333333</td>
      <td>-2.720833</td>
      <td>0.000000</td>
      <td>0.623000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>77133.340000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16.329170</td>
      <td>9.733334</td>
      <td>0.527000</td>
      <td>1.870000</td>
      <td>2.308333</td>
      <td>103.333300</td>
      <td>96700.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.691670</td>
      <td>13.208330</td>
      <td>4.520000</td>
      <td>2.301000</td>
      <td>2.829167</td>
      <td>152.916700</td>
      <td>99438.750000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>25.558330</td>
      <td>16.970830</td>
      <td>7.041000</td>
      <td>2.865000</td>
      <td>3.912500</td>
      <td>217.916700</td>
      <td>100516.700000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>37.700000</td>
      <td>28.695830</td>
      <td>10.634000</td>
      <td>5.101000</td>
      <td>8.641666</td>
      <td>310.416700</td>
      <td>102358.300000</td>
    </tr>
  </tbody>
</table>

Now we normalize our data using StandardScaler from sklearn.preprocessing, let's see our dataset describe after normalizing

<table>
  <thead>
    <tr>
      <th></th>
      <th>Outside Dry-Bulb Temperature</th>
      <th>Outside Dew-Point Temperature</th>
      <th>Direct Normal Solar</th>
      <th>Diffuse Horizontal Solar</th>
      <th>Wind Speed</th>
      <th>Wind Direction</th>
      <th>Atmospheric Pressure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.090000e+02</td>
      <td>1.090000e+02</td>
      <td>1.090000e+02</td>
      <td>1.090000e+02</td>
      <td>1.090000e+02</td>
      <td>1.090000e+02</td>
      <td>1.090000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.179704e-16</td>
      <td>1.146827e-16</td>
      <td>9.472545e-17</td>
      <td>7.435439e-17</td>
      <td>-8.021107e-18</td>
      <td>3.982543e-16</td>
      <td>-2.924266e-15</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.004619e+00</td>
      <td>1.004619e+00</td>
      <td>1.004619e+00</td>
      <td>1.004619e+00</td>
      <td>1.004619e+00</td>
      <td>1.004619e+00</td>
      <td>1.004619e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.730600e+00</td>
      <td>-2.587317e+00</td>
      <td>-1.295173e+00</td>
      <td>-2.304901e+00</td>
      <td>-2.216319e+00</td>
      <td>-2.303661e+00</td>
      <td>-3.651728e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-6.910070e-01</td>
      <td>-5.783308e-01</td>
      <td>-1.134279e+00</td>
      <td>-7.330721e-01</td>
      <td>-5.967510e-01</td>
      <td>-8.438929e-01</td>
      <td>-6.075818e-02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.046241e-02</td>
      <td>-1.777798e-02</td>
      <td>8.478979e-02</td>
      <td>-1.898017e-01</td>
      <td>-2.313245e-01</td>
      <td>-1.434383e-01</td>
      <td>4.418707e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.539452e-01</td>
      <td>5.891522e-01</td>
      <td>8.544549e-01</td>
      <td>5.211135e-01</td>
      <td>5.287614e-01</td>
      <td>7.748033e-01</td>
      <td>6.397014e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.423333e+00</td>
      <td>2.480516e+00</td>
      <td>1.951403e+00</td>
      <td>3.339565e+00</td>
      <td>3.846829e+00</td>
      <td>2.081532e+00</td>
      <td>9.776809e-01</td>
    </tr>
  </tbody>
</table>

As we can see, our data is so imbalanced, there are so many data in class 3 and just a few data in class 5, it's would have a bad effect on training process
but we can balence these data by SMOTE Oversampling
let's check out our data before Oversampling

![OverSampling Diagram](https://github.com/Amin1384Movahedi/multiclass-classifier/blob/main/OverSampling_diagram.png)

Now let's check out the correlation between variables and choose just two features for train our models at first and see the result, later we would choose more features from our dataset to train our machine learning and deep learning models

![Data Correlation](https://github.com/Amin1384Movahedi/multiclass-classifier/blob/main/corr.png)

we choose 'Outside Dry-Bulb Temperature' and 'Direct Normal Solar' columns to train the models

## Let's start training our machine learning models with KNN

the result of training the KNN model with a hyper parameter tunner is the max accuracy of 0.676 and 5 n_neighbors

There is the confusion matrix and calculated Precision, Recall and f1_score

![KNN Confusion Matrix Heatmap](https://github.com/Amin1384Movahedi/multiclass-classifier/blob/main/KNN_Confusion_Matrix_Heatmap.png)

<pre>
+-----------+--------------------+
|  metrices |       values       |
+-----------+--------------------+
| Precision | 0.6111402925882564 |
|   Recall  | 0.6153846153846154 |
|  f1_score | 0.6088187595879904 |
+-----------+--------------------+
</pre>

## Now it's time to train the Random Forest model

the best accuracy of random forest model we got is 0.676 with 5 n_estimators

here is the confusion matrix of trained random forest

![RandomForest Confusion Matrix Heatmap](https://github.com/Amin1384Movahedi/multiclass-classifier/blob/main/RandomForest_Confusion_Matrix_Heatmap.png)

the maximum accuracy that we got as our result is 0.753
the Precision, Recall and f1_score of our trained model

<pre>
+-----------+--------------------+
|  metrices |       values       |
+-----------+--------------------+
| Precision | 0.6165445665445666 |
|   Recall  | 0.5846153846153846 |
|  f1_score | 0.5930002191540653 |
+-----------+--------------------+
</pre>

## it's turn for Naive Baysian machine learning model

the accuracy we get from training naive baysian is 0.476, it's actualy nat a good result, naive baysian wasn't a optimal model for our dataset

![Naive Baysian Confusion Matrix Heatmap](https://github.com/Amin1384Movahedi/multiclass-classifier/blob/main/Naive_Baysian_Confusion_Matrix_Heatmap.png)

as you see, the Precision and Recall is the lowest values we got till now

<pre>
+-----------+---------------------+
|  metrices |        values       |
+-----------+---------------------+
| Precision |  0.5009594270463836 |
|   Recall  | 0.47692307692307695 |
|  f1_score | 0.44413637644406884 |
+-----------+---------------------+
</pre>

## The next machine learning algorithm is Decision Tree model

we trained this model with our dataset and we get the 0.553 as our accuracy

there is this model's confusion matrix

![Decision Tree Confusion Matrix Heatmap](https://github.com/Amin1384Movahedi/multiclass-classifier/blob/main/Decision_Tree_Confusion_Matrix_Heatmap.png)

and there is this model's Precision, Recall and f1-score

<pre>
+-----------+--------------------+
|  metrices |       values       |
+-----------+--------------------+
| Precision | 0.7354071569456184 |
|   Recall  | 0.7076923076923077 |
|  f1_score | 0.7145262533568382 |
+-----------+--------------------+
</pre>

## Finally, it's turn for Neural Network

at first, we build a model look like this...

<pre>
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 10)                30        
                                                                 
 dense_1 (Dense)             (None, 64)                704       
                                                                 
 dense_2 (Dense)             (None, 128)               8320      
                                                                 
 dense_3 (Dense)             (None, 64)                8256      
                                                                 
 dense_4 (Dense)             (None, 5)                 325       
                                                                 
=================================================================
Total params: 17635 (68.89 KB)
Trainable params: 17635 (68.89 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

</pre>

we trained our model with 100 epochs, but we get Test accuracy: 0.569, it's not good
let's increase our features and check our model performance again.

in this time, we build a model look like this...

<pre>
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_17 (Dense)            (None, 5)                 45        
                                                                 
 dense_18 (Dense)            (None, 10)                70        
                                                                 
 dense_19 (Dense)            (None, 64)                768       
                                                                 
 dense_20 (Dense)            (None, 128)               8448      
                                                                 
 dense_21 (Dense)            (None, 256)               33280     
                                                                 
 dense_22 (Dense)            (None, 128)               33024     
                                                                 
 dense_23 (Dense)            (None, 10)                1300      
                                                                 
 dense_24 (Dense)            (None, 5)                 55        
                                                                 
=================================================================
</pre>

and we trained this model with 200 epochs and we got 0.830 as our accuracy result
let's see this model performance

![NN Model Accuracy Graph](https://github.com/Amin1384Movahedi/multiclass-classifier/blob/main/NN_Accuracy_Graph.png)

there is our Neural Network confusion matrix

![NN Confusion Matrix](https://github.com/Amin1384Movahedi/multiclass-classifier/blob/main/NN_Confusion_Matrix_Heatmap.png)

also we have the calculated Precision, Recall and f1_score of our trained model

<pre>
+-----------+--------------------+
|  metrices |       values       |
+-----------+--------------------+
| Precision | 0.8387494858083093 |
|   Recall  | 0.8307692307692308 |
|  f1_score | 0.8303098192130451 |
+-----------+--------------------+
</pre>
