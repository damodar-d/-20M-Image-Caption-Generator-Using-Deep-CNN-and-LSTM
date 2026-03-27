# About this Project
This is a supervised transfer learning project for generating image captions by using Google's pre-trained Xception model <strong>( 20M parameters)</strong> 

# Before trying out
* Having virtual environment activated beforehand is strongly recommended.
* Use the ```requirements.txt```, install the dependencies : 
<p align="center"><code>pip install -r requirements.txt</code></p>


# How to try out
1. Navigate to ```src/inference```
2. <strong>Copy an image you want to generate caption for inside this folder. </strong>
3. Type following command to get the result.
<p align="center"><code>python predict.py --image < image-name > </code></p>
