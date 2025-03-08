# MedCLIP-Based Medical Image Classification

## Project Overview
This project utilizes **MedCLIP**, a Contrastive Language-Image Pretraining model, for **medical image classification**. The dataset used was sourced from **Kaggle**, and the model was trained and fine-tuned to enhance classification performance. There are 2 labels Normal and Pneumonia. The dataset was highly imbalanced and was thus balanced firstly, and then further finetuning was done to make it adaptable to data.

## Achievements
- **Initial Model Performance**: Without fine-tuning, the model achieved **86% accuracy**.
- **Fine-Tuned Performance**: After hyperparameter tuning and optimization, the model reached **97% training accuracy** and **95% testing accuracy**.

## Features
- **Medical Image Classification**: Uses MedCLIP for accurate categorization of medical images.
- **Pretrained Model Adaptation**: Fine-tuned MedCLIP for improved performance on the Kaggle dataset.
- **Hyperparameter Optimization**: Enhanced model generalization and accuracy.
- **Efficient Training Pipeline**: Implements best practices for training deep learning models.

## Technologies Used
- **Machine Learning**: PyTorch, MedCLIP (for medical image classification)
- **Dataset Source**: Kaggle (Medical Imaging Dataset)
- **Optimization Techniques**: Transfer Learning, Contrastive Learning
- **Data Handling**: Python, Pandas, NumPy (for preprocessing and analysis)

## Performance Improvement
| Stage               | Training Accuracy | Testing Accuracy |
|---------------------|------------------|------------------|
| Without Fine-Tuning | 86%              | -                |
| After Fine-Tuning  | 97%              | 95%              |

## Fine-Tuning Strategies
- **Hyperparameter Optimization**: Adjusted learning rate, batch size, and regularization.
- **Data Augmentation**: Introduced transformations to improve model generalization.
- **Transfer Learning**: Utilized pretrained MedCLIP model with additional fine-tuning layers.
- **Early Stopping**: Monitored validation loss to prevent overfitting.

## Future Enhancements
- **Enhance dataset diversity** to improve generalization.
- **Optimize inference speed** for real-time classification.
- **Integrate explainability techniques** to make model predictions more interpretable.


