# 🐱🐶 Cats vs Dogs Classification

This project uses a **Convolutional Neural Network (CNN)** in TensorFlow/Keras to classify images of cats and dogs.  
The dataset is sourced from Kaggle (`salader/dogs-vs-cats`), resized to **256×256**, normalized, and split into training & validation sets.  

**Model Architecture:**  
`Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense(128) → Dense(64) → Dense(1)`  

**Training Configuration:**  
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Metric: Accuracy  
- Epochs: 10  
- Batch size: 32  

Trained for **10 epochs**, achieving high accuracy. Includes code for predicting custom images after training.
