# VP-UWIQE: Visual Positional Underwater Image Quality Evaluator

## Abstract

Underwater images (UWIs) often suffer from poor illumination, color casts, blur and noise due to scattering and light absorption in water. These degradations significantly affect visual quality and reduce the performance of subsequent image-processing tasks in consumer electronics. Therefore, underwater image quality assessment (UWIQA) is essential for real-time monitoring and for optimizing post-processing enhancement methods. To
address this, a no-reference (NR) UWIQA model is proposed that effectively assess the quality of degraded UWIs by exploiting the visual and positional characteristics of the underwater scene.  With the encoded visual and positional attributes, the patch weights are computed using an attention aggregation module. These attention weights enable the model to assign relative importance to each patch based on the salient information it contains, as degradations in UWIs are spatially non-uniform. Finally, a weighted sum of all patch features is computed to obtain the global feature vector, which is then fed into a trained support vector regressor to estimate the overall quality of the degraded UWI. Extensive experiments conducted on UID2021 and SAUD datasets demonstrate that the predicted scores of the proposed model exhibit superior linearity, ordinal relationships and monotonicity compared to existing traditional, deep learning and UWIQA approaches.

## Block Diagram

![alt text](Block_Diagram.png)

## Steps to import and run the project

1. Clone the project

```bash
git clone https://github.com/RohanRaviKumar/VPUWIQE.git
```

2. Install dependences

```bash
pip install -r requirements.txt
```

3. Run the web app

```bash
#For UID code
python uid_app.py

#For SAUD code
python saud_app.py
```

Open this link `http://127.0.0.1:5000/` on your browser and upload a `.png`, `.jpg`, or `.jpeg` image to get a predicted quality score.