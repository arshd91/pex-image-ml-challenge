# pex-image-classifier
Transfer Learning for coding challenge by Pex.

1. Generate data using generate_data.py:
    
    ``
    python3 generate_data.py [path-to-images] [path-to-data-dump] [extension]
    ``
    
    Example:
    ``
     python3 generate_data.py Pex-ML-Challenge/cleaned_data Pex-ML-Challenge/ jpg
    ``

2. Train using Transfer Learning over pre-trained ResNet-18 Model

    Training/Validation Loss vs Epochs Curves:
    ![](loss.png)

    Training/Validation Accuracy vs Epochs Curves:
    ![](accuracy.png)

3. Inference using inference.py:
    ``
    python3 inference.py [model-path] [test_image_path]
    ``

    Example:
    ``
    python3 inference.py pex-challenge-model.pyt test_image.jpg
    ``
    
    Output:
    ![](inference.png)


    



