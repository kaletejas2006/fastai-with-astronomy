# Deep Learning for Galaxy Zoo challenge with Tensorflow and TPUs

[Galaxy Zoo](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) was a machine learning competition on Kaggle wherein competitors had to build a model that could predict galaxy morphology features based on its `JPG` image. In a [previous notebook](https://colab.research.google.com/drive/1i6ghXgyQPcyLn5Q9-c7QbIsMEpY4vqQQ), we used the `fastai` library to build a `ResNet18` model for this task. Due to the number of training images and the complexity of the model, we could not fit it for all the images. We instead had to rely on a small sample of images (about 10%) in order to train a model in reasonable amount of time.

With this notebook, the aim is to see if a combination of the TensorFlow package and *Tensor Processing Units* (TPUs) (specialised hardware built for machine learning by Google) can help us train `ResNet18` or even deeper models on the entire training dataset in a reasonable amount of time.

The `fastai` package is built on top of another deep learning package called *PyTorch*. Thus, with this notebook, we will be exploring some TensorFlow syntax (Keras to be specific) and understanding how to configure a notebook to run on TPUs.

## Step 0: Prerequisites
Let us begin with by mounting our Google Drive (where the data is stored) and loading the required packages.


```python
from google.colab import files, drive
import sys

drive.mount("/content/gdrive", force_remount=True)
root_dir = "/content/gdrive/My Drive/Colab Notebooks/fastai_2022/data/galaxy-zoo-the-galaxy-challenge"
sys.path.append(root_dir)
```

    Mounted at /content/gdrive


Let us now do a quick check to ensure that we are able to access the contents of this directory in the notebook.


```python
import os
os.listdir(root_dir)
```




    ['images_test_rev1',
     'images_training_rev1',
     'central_pixel_benchmark.csv',
     'all_ones_benchmark.csv',
     'all_zeros_benchmark.csv',
     'training_solutions_rev1.csv']




```python
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
```

## Step 1: Load data

We know that the training images are in the directory `images_training_rev1`. Let us begin by creating a `tf.data.Dataset` object that can access all the training images. 

`tf.data.Dataset` is an API that allows us to write input data pipelines effectively. It provides a convenient way for multiple functions/operations like:
- Create a data source from input data.
- Apply transformations to preprocess the data.
- Iterate over the data to process its elements.

This API can handle data of multiple types like images, text, etc.


```python
root_dir: Path = Path(root_dir)
training_img_dir: Path = Path(root_dir/"images_training_rev1")
```


```python
list_ds: Dataset = tf.data.Dataset.list_files(f"{str(training_img_dir)}/*.jpg", shuffle=False)
```

Let us shuffle the images. By doing so, we can evaluate the performance of our model as there should not be a significant discrepancy in its results with different shuffled images. With `reshuffle_each_iteration` set to `False`, we prevent shuffling with every epoch of our model training.


```python
image_count: int = len(list(training_img_dir.glob("*.jpg")))
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
```

Now we list the paths of first five images.


```python
for f in list_ds.take(5):
    print(f.numpy())
```

    b'/content/gdrive/My Drive/Colab Notebooks/fastai_2022/data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/131465.jpg'
    b'/content/gdrive/My Drive/Colab Notebooks/fastai_2022/data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/744668.jpg'
    b'/content/gdrive/My Drive/Colab Notebooks/fastai_2022/data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/135172.jpg'
    b'/content/gdrive/My Drive/Colab Notebooks/fastai_2022/data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/573937.jpg'
    b'/content/gdrive/My Drive/Colab Notebooks/fastai_2022/data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/123647.jpg'


We know that the outputs for these images are available as columns in `training_solutions_rev1.csv`. Let us first load and peek at this table.


```python
training_outputs: pd.DataFrame = pd.read_csv(root_dir/"training_solutions_rev1.csv")
training_outputs.head()
```





  <div id="df-7fbb8eea-f5ac-4192-a048-086a642d73bd">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GalaxyID</th>
      <th>Class1.1</th>
      <th>Class1.2</th>
      <th>Class1.3</th>
      <th>Class2.1</th>
      <th>Class2.2</th>
      <th>Class3.1</th>
      <th>Class3.2</th>
      <th>Class4.1</th>
      <th>Class4.2</th>
      <th>...</th>
      <th>Class9.3</th>
      <th>Class10.1</th>
      <th>Class10.2</th>
      <th>Class10.3</th>
      <th>Class11.1</th>
      <th>Class11.2</th>
      <th>Class11.3</th>
      <th>Class11.4</th>
      <th>Class11.5</th>
      <th>Class11.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100008</td>
      <td>0.383147</td>
      <td>0.616853</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.616853</td>
      <td>0.038452</td>
      <td>0.578401</td>
      <td>0.418398</td>
      <td>0.198455</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.279952</td>
      <td>0.138445</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.092886</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.325512</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100023</td>
      <td>0.327001</td>
      <td>0.663777</td>
      <td>0.009222</td>
      <td>0.031178</td>
      <td>0.632599</td>
      <td>0.467370</td>
      <td>0.165229</td>
      <td>0.591328</td>
      <td>0.041271</td>
      <td>...</td>
      <td>0.018764</td>
      <td>0.000000</td>
      <td>0.131378</td>
      <td>0.459950</td>
      <td>0.000000</td>
      <td>0.591328</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100053</td>
      <td>0.765717</td>
      <td>0.177352</td>
      <td>0.056931</td>
      <td>0.000000</td>
      <td>0.177352</td>
      <td>0.000000</td>
      <td>0.177352</td>
      <td>0.000000</td>
      <td>0.177352</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100078</td>
      <td>0.693377</td>
      <td>0.238564</td>
      <td>0.068059</td>
      <td>0.000000</td>
      <td>0.238564</td>
      <td>0.109493</td>
      <td>0.129071</td>
      <td>0.189098</td>
      <td>0.049466</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.094549</td>
      <td>0.000000</td>
      <td>0.094549</td>
      <td>0.189098</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100090</td>
      <td>0.933839</td>
      <td>0.000000</td>
      <td>0.066161</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7fbb8eea-f5ac-4192-a048-086a642d73bd')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7fbb8eea-f5ac-4192-a048-086a642d73bd button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7fbb8eea-f5ac-4192-a048-086a642d73bd');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




The values in the `GalaxyID` column match the filenames. So, let us modify these values to full paths for the corresponding files.


```python
training_outputs = (training_outputs.assign(GalaxyImageFile=lambda x: [f"{str(training_img_dir)}/{id}.jpg" for id in x['GalaxyID'].to_list()])
                    .drop("GalaxyID", axis=1))
training_outputs.head()
```





  <div id="df-fb0b9d50-e63f-4c2b-b9fe-b0b15eb3ad5b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class1.1</th>
      <th>Class1.2</th>
      <th>Class1.3</th>
      <th>Class2.1</th>
      <th>Class2.2</th>
      <th>Class3.1</th>
      <th>Class3.2</th>
      <th>Class4.1</th>
      <th>Class4.2</th>
      <th>Class5.1</th>
      <th>...</th>
      <th>Class10.1</th>
      <th>Class10.2</th>
      <th>Class10.3</th>
      <th>Class11.1</th>
      <th>Class11.2</th>
      <th>Class11.3</th>
      <th>Class11.4</th>
      <th>Class11.5</th>
      <th>Class11.6</th>
      <th>GalaxyImageFile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.383147</td>
      <td>0.616853</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.616853</td>
      <td>0.038452</td>
      <td>0.578401</td>
      <td>0.418398</td>
      <td>0.198455</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.279952</td>
      <td>0.138445</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.092886</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.325512</td>
      <td>/content/gdrive/My Drive/Colab Notebooks/fasta...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.327001</td>
      <td>0.663777</td>
      <td>0.009222</td>
      <td>0.031178</td>
      <td>0.632599</td>
      <td>0.467370</td>
      <td>0.165229</td>
      <td>0.591328</td>
      <td>0.041271</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.131378</td>
      <td>0.459950</td>
      <td>0.000000</td>
      <td>0.591328</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>/content/gdrive/My Drive/Colab Notebooks/fasta...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.765717</td>
      <td>0.177352</td>
      <td>0.056931</td>
      <td>0.000000</td>
      <td>0.177352</td>
      <td>0.000000</td>
      <td>0.177352</td>
      <td>0.000000</td>
      <td>0.177352</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>/content/gdrive/My Drive/Colab Notebooks/fasta...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.693377</td>
      <td>0.238564</td>
      <td>0.068059</td>
      <td>0.000000</td>
      <td>0.238564</td>
      <td>0.109493</td>
      <td>0.129071</td>
      <td>0.189098</td>
      <td>0.049466</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.094549</td>
      <td>0.000000</td>
      <td>0.094549</td>
      <td>0.189098</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>/content/gdrive/My Drive/Colab Notebooks/fasta...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.933839</td>
      <td>0.000000</td>
      <td>0.066161</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>/content/gdrive/My Drive/Colab Notebooks/fasta...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fb0b9d50-e63f-4c2b-b9fe-b0b15eb3ad5b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-fb0b9d50-e63f-4c2b-b9fe-b0b15eb3ad5b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fb0b9d50-e63f-4c2b-b9fe-b0b15eb3ad5b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Next, we split the data into training and validation sets.


```python
val_factor: float = 0.2
x_train: list
y_train: np.ndarray
x_valid: list
y_valid: np.ndarray
x_train, x_valid, y_train, y_valid = train_test_split(
    training_outputs.loc[:, "GalaxyImageFile"].to_list(), 
    training_outputs.drop("GalaxyImageFile", axis=1).values, 
    test_size=val_factor, random_state=0)
```


```python
num_outputs: int = int(y_train.shape[1])
num_outputs
```




    37



Here `x_{train|test}` holds Galaxy IDs that correspond to image file names and `y_{train|test}` holds outputs for each image. Let us now create tensors for the IDs.


```python
train_id_ds: Dataset = tf.data.Dataset.from_tensor_slices(x_train)
valid_id_ds: Dataset = tf.data.Dataset.from_tensor_slices(x_valid)
```

Now, let us define a function to load the images. As TensorFlow's `ResNet50` model was trained of images of size *224 x 224 x 3`, we modify the image height and width accordingly. 

Along with it, let us define the desired dimensions of the image to be used for training the model and the number of images to process in each batch.


```python
batch_size: int = 32
img_height: int = 224  # 180
img_width: int = 224  # 180
```


```python
def decode_img(img: tf.Tensor) -> tf.Tensor:
    # Convert the compressed string to a 3D uint8 tensor.
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size. The output is a 3-D float Tensor
    # of shape `[new_height, new_width, channels]`.
    return tf.image.resize(img, [img_height, img_width])
```


```python
def load_image(fp: tf.Tensor) -> tf.Tensor:
    file_contents: tf.Tensor = tf.io.read_file(fp)
    img: tf.Tensor = decode_img(file_contents)
    return img
```


```python
AUTOTUNE = tf.data.AUTOTUNE
```


```python
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_img_ds: Dataset = train_id_ds.map(load_image, num_parallel_calls=AUTOTUNE)
valid_img_ds: Dataset = valid_id_ds.map(load_image, num_parallel_calls=AUTOTUNE)
```

Next, we create tensors of the outputs.


```python
train_output_ds: Dataset = tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.float32))
valid_output_ds: Dataset = tf.data.Dataset.from_tensor_slices(tf.cast(y_valid, tf.float32))
```

Finally, let us combine the images and outputs into `(image, outputs)` pairs.


```python
train_ds: Dataset = tf.data.Dataset.zip((train_img_ds, train_output_ds))
valid_ds: Dataset = tf.data.Dataset.zip((valid_img_ds, valid_output_ds))
```

Let us print some information about one of the input-output pairs.


```python
for image, output in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", output.numpy())
```

    Image shape:  (224, 224, 3)
    Label:  [0.393467   0.53702    0.069513   0.         0.53702    0.
     0.53702    0.37311667 0.16390334 0.11397068 0.24111661 0.0207236
     0.1612091  0.595436   0.404564   0.02983227 0.36363474 0.
     0.03331107 0.         0.09933838 0.2974203  0.13205461 0.03331107
     0.         0.         0.         0.         0.12437209 0.12437209
     0.12437209 0.12437209 0.12437209 0.         0.         0.
     0.12437209]


*Note*: We tried to load the data using the code structure and syntax provided in the [Tensorflow tutorial](https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control) but as the value passed by `map()` in each iteration is a *symbolic tensor*, we could not use it fetch the outputs from a dataframe directly.

Next, we configure the dataset for improved performance i.e.:
- Ensure it is well-shuffled
- Make it batched
- Ensure that batches are available as soon as possible.


```python
def configure_for_performance(ds: Dataset) -> Dataset:
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
```


```python
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(valid_ds)
```

As a final check, let us see one of the training images with its outputs.


```python
image_batch, label_batch = next(iter(train_ds))


plt.figure(figsize=(10, 10))
for i in range(1):
    print(f"Output: {label_batch[i].numpy().tolist()}")
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    plt.axis("off")
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-25-05ae24809b3a> in <module>
    ----> 1 image_batch, label_batch = next(iter(train_ds))
          2 
          3 
          4 plt.figure(figsize=(10, 10))
          5 for i in range(1):


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
        834   def __next__(self):
        835     try:
    --> 836       return self._next_internal()
        837     except errors.OutOfRangeError:
        838       raise StopIteration


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
        820           self._iterator_resource,
        821           output_types=self._flat_output_types,
    --> 822           output_shapes=self._flat_output_shapes)
        823 
        824       try:


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
       2918       _result = pywrap_tfe.TFE_Py_FastPathExecute(
       2919         _ctx, "IteratorGetNext", name, iterator, "output_types", output_types,
    -> 2920         "output_shapes", output_shapes)
       2921       return _result
       2922     except _core._NotOkStatusException as e:


    KeyboardInterrupt: 


## Step 2: Train model

Having loaded the data, let us now define a neural network and train it for the data. In our [fastai notebook](https://colab.research.google.com/drive/1i6ghXgyQPcyLn5Q9-c7QbIsMEpY4vqQQ), we used the `ResNet18` model but the same model is not easily available in TensorFlow. Instead, we will use the `ResNet50` model. Both are neural networks consisting of *residual units* in the hidden layers. The main difference between them is that `ResNet18` consists of 18 layers while `ResNet50` consists of 50 layers.

TensorFlow provides a service called [TensorFlow Hub](https://tfhub.dev/) which allows us to easily download models pretrained on a larger dataset. By downloading the *feature vector* version of model (also called the *headless version*), we get the entire pretrained except for the output layer. We can thus use this layer and attach an output layer and/or other layers in between to train our model.

To fetch the feature vector, we first search for the required model on TensorFlow Hub and then copy the URL provided for downloading the features.


```python
resnet_50_url: str = "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"

feature_extractor_model: str = resnet_50_url
```

Next, we wrap the feature extractor in a Keras layer using `hub.KerasLayer()`. While doing so, we specify the dimensions of our images and set the `trainable` attribute to `False` to prevent model training from affecting features in this pretrained model.


```python
feature_extractor_layer: hub.KerasLayer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(img_height, img_width, 3),
    trainable=False
)
```

Let us get a batch of training images from our dataset. A batch consists of 32 images of dimensions *224 x 224 x 3*.


```python
image_batch: tf.Tensor
labels_batch: tf.Tensor
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
```

    (32, 224, 224, 3)
    (32, 37)


Let us now pass this batch through the feature extractor layer to understand the structure of its output. We see that, for each image, the output of the layer consists of 2048 values.


```python
feature_batch: tf.Tensor = feature_extractor_layer(image_batch)
print(feature_batch.shape)
```

    (32, 2048)


We know that each of our images has a list of 37 real values as outputs. So, let us create a simple network by attaching a fully-connected layer of 37 hidden units to the feature extractor layer.


```python
num_outputs
```




    38




```python
model: Sequential = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(num_outputs)
])

model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     keras_layer_1 (KerasLayer)  (None, 2048)              23561152  
                                                                     
     dense_4 (Dense)             (None, 37)                75813     
                                                                     
    =================================================================
    Total params: 23,636,965
    Trainable params: 75,813
    Non-trainable params: 23,561,152
    _________________________________________________________________


Before we train the model, let us generate predictions from it. The predictions themselves will be meaningless but we can ensure that our code from data loading to model prediction is working fine. Once again, we predict on a batch of images and see that the result is a `Tensor` with 37 outputs for each image.


```python
predictions_to_test: tf.Tensor = model(image_batch)
predictions_to_test.shape
```




    TensorShape([32, 37])



Before we initiate training, let us first compile our model. We will use the *Adam* optimiser, *Mean Absolute Error* as the loss function, and the *Root Mean Squared Error* to use as the metric.


```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

Let us now define a TensorBoard callback to automatically capture loss and metric values during model training and show them in TensorBoard.


```python
log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1)  # Enable histogram computation for every epoch.
```

Finally, let us fit the model by specifying the training and validation datasets and fitting the model for 10 epochs.


```python
NUM_EPOCHS: int = 10

history: dict = model.fit(train_ds,
                          validation_data=val_ds,
                          epochs=NUM_EPOCHS,
                          callbacks=tensorboard_callback)
```

    Epoch 1/10
      59/1540 [>.............................] - ETA: 6:08:11 - loss: 45.4240 - root_mean_squared_error: 88.3485


```python
%tensorboard --logdir logs/fit
```

## References
- [Load and preprocess images: Using tf.data for finer control](https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control)
- [Get string value from a tensor in `Dataset.map()`](https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string)
- [galaxy_zoo_Xception](https://www.kaggle.com/code/hironobukawaguchi/galaxy-zoo-xception)
- [Transfer learning with TensorFlow Hub](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub#download_the_headless_model)


```python

```
