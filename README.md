# Handwritten-Text-Detection
To implement handwritten text prediction using machine learning, i have followed these steps:

1. **Data Collection**: Dataset found from MNIST kaggle.com,its of handwritten text samples. This dataset should include both handwritten text images and their corresponding labels.

2. **Data Preprocessing**: Preprocess the dataset to prepare it for training. This step includes PCA(Principal Component Analysis)  to reduce the dimension of the data.bcz its having 784 dimensions(28x28),for(15,20,25 and 30 components).

3. **Feature Extraction**: Extract relevant features from the preprocessed handwritten text images. 

4. **Data Split**: Split the dataset into a training set and a testing set. The training set will be used to train the machine learning model, while the testing set will be used to evaluate its performance.

5. **Model Selection**: Choose an appropriate machine learning model for handwritten text prediction. Some popular choices include Support Vector Machines (SVM), k-Nearest Neighbors (KNN), Random Forest and Decision Tree.

6. **Model Training**: Train the selected model using the training dataset. This involves feeding the extracted features and their corresponding labels into the machine learning algorithm.

7. **Model Evaluation**: Evaluate the trained model's performance using the testing dataset. Calculate relevant metrics such as accuracy, precision, recall, and F1-score,ROC-AUC Curve to assess how well the model predicts handwritten text.

8. **Model Fine-tuning**: Depending on the evaluation results, you may need to fine-tune your model by adjusting hyperparameters, exploring different feature extraction techniques, or trying alternative models.

9. **Prediction**: Once model trained and evaluated, I used it to make predictions on new handwritten text samples. Provide the unseen data to the model, and it will predict the corresponding labels.

10. **Deployment**: Integrate the trained model into your application or system for practical use. This may involve creating an interface or API that allows users to interact with the model and receive predictions on their handwritten text inputs.(In Progress)

