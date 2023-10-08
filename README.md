# multiclass-classifier

We have a dataset that collected by Mr.Parham ghadermazi 
(Email: parham.ghadermazi@gmail.com)

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
