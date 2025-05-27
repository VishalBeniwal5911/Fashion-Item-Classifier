
# 👕 Clothes Name Prediction using Neural Network

This project uses a simple neural network to predict the type of clothing item (e.g., T-shirt, trouser, sneaker) from image data. The dataset is uploaded as a ZIP file containing a CSV, and the model is built and trained using TensorFlow and Keras.

## 📂 Project Structure

- `zipfile` to unzip uploaded dataset
- `pandas` and `numpy` for data handling
- `matplotlib` and `seaborn` for visualization
- `sklearn` for preprocessing and evaluation
- `tensorflow.keras` for building the neural network

## 🧠 Model Architecture

- Input Layer: 784 neurons (flattened 28x28 image)
- Dense Layer 1: 128 neurons, ReLU activation
- Dense Layer 2: 64 neurons, ReLU activation
- Output Layer: 10 neurons, Softmax activation

## 🧾 Steps in the Notebook

1. **Upload ZIP File**  
   Upload a ZIP file containing the CSV dataset (like Fashion MNIST).

2. **Unzip and Read Data**  
   Extract and load the CSV into a DataFrame.

3. **Preprocess Data**  
   Normalize pixel values and convert labels to categorical.

4. **Train-Test Split**  
   80% for training, 20% for testing.

5. **Build and Train Model**  
   Train a simple neural network using Keras.

6. **Evaluate Model**  
   Predict and generate a confusion matrix and classification report.

7. **Visualize Predictions**  
   Display 10 sample predictions with actual and predicted labels.

## 🧪 Output

- 📊 Confusion Matrix for evaluation  
- 📄 Classification Report (Precision, Recall, F1-Score)  
- 🖼️ Visual output of 10 sample predictions

## 📦 Dependencies

Install these Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## 🔍 Sample Labels

```
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
```

## 💡 Usage

Run the code in **Google Colab**. Make sure to upload your ZIP file containing the CSV dataset when prompted.
