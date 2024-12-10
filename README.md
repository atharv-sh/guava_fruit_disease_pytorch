
# **Guava Fruit Disease Detection using PyTorch**

**Project Overview**

This repository contains a PyTorch-based model for detecting diseases in guava fruits. The model is trained on a dataset of images of healthy and diseased guava fruits.

**Requirements**

* Python 3.12+
* PyTorch
* torchvision
* NumPy
* Matplotlib

**Installation**

**Clone the repository:**
   ```bash
   git clone https://github.com/atharv-sh/Guava_fruit_disease_pytorch.git
   ```

**Dataset**

* **Data Collection:**
  - Collect images of healthy and diseased guava fruits.
  - Categorize images into different disease classes (e.g., anthracnose, bacterial blight, etc.).
* **Data Preprocessing:**
  - Resize images to a standard size.
  - Normalize pixel values.
  - Create a data loader for efficient batch processing.

**Model Architecture**

* **Convolutional Neural Network (CNN):**
  - Utilize a deep CNN architecture to extract relevant features from the image data.
  - Employ convolutional layers, pooling layers, and fully connected layers.
* **Transfer Learning:**
  - Consider using a pre-trained model (e.g., ResNet, VGG) and fine-tune it on the guava fruit disease dataset.

**Training**

* **Data Augmentation:**
  - Apply techniques like random rotations, flips, and color jittering to increase data diversity.
* **Loss Function:**
  - Use cross-entropy loss to optimize the model's predictions.
* **Optimizer:**
  - Employ an optimizer like Adam or SGD to update model parameters.
* **Training Loop:**
  - Iterate over the training dataset, compute loss, and update model weights.

**Evaluation**

* **Test Dataset:**
  - Evaluate the model's performance on a separate test dataset.
* **Metrics:**
  - Calculate accuracy, precision, recall, and F1-score to assess the model's effectiveness.

**Usage**

1. **Load the trained model:**
   ```python
   model = torch.load('model.pth')
   ```
2. **Preprocess the input image:**
   - Resize and normalize the image.
3. **Make predictions:**
   ```python
   with torch.no_grad():
       output = model(input_image)
       predicted_class = torch.argmax(output, dim=1)
   ```

**Additional Considerations**

* **Data Quality:**
  - Ensure the quality of the dataset by removing low-quality images and inconsistencies.
* **Model Complexity:**
  - Balance model complexity with computational resources and overfitting.
* **Hyperparameter Tuning:**
  - Experiment with different hyperparameters (e.g., learning rate, batch size, optimizer) to optimize performance.
