# Introduction 

In the US, breast cancer makes up one-third of all new cancer cases in women every year    
(American Cancer Society, 2025). When breast cancer is caught early, the 5-year relative survival rate in the US is 99% (National Breast Cancer, 2025). The survival rate remains over 90% for most countries when caught early, but drops to only 30% if caught after metastasis, when the cancer spreads to other parts of the body (Hofeld and Schaffner, 2024). The best way to detect breast cancer is through a mammogram (Cancer Research UK, 2025).  Unfortunately, 12.5% of mammogram images are misclassified during screening, partly due to human error (American Cancer Society, 2022). However, human error can be avoided or reduced by training an AI model to classify mammogram images. In the UK alone, there are roughly 32 breast cancer deaths a day (Cancer Research UK, 2019). Every malignant tumor misclassified as a benign tumor is a person who may not receive the care they need in time. Our goal is to develop a model that accurately classifies breast masses as benign or malignant based on mammogram images. 

# Methodology and Data

**Getting Started**  
While researching which models would be effective for classifying masses in breast tissue, we found Elkorany et al.’s paper from 2023 on breast cancer diagnosis, which mentions three pre-trained convolutional neural network (CNN) models that performed well when classifying breast cancer images: VGG16, InceptionV3, and ResNet (Elkorany et al., 2023). We first used a smaller dataset called MIAS to test out which of these 3 would be faster to train on, which we downloaded from Kaggle (https://www.kaggle.com/datasets/kmader/mias-mammography/data).   
The MIAS data set has 332 1024 x 1024 pixel mammogram images, with information on malignant status and breast density. During our initial testing, we created separate models using the three pre-trained CNNs as feature extractors and a classifying layer. When attempting to classify breast density, we found that the MIAS data set was too small to produce meaningful results.

![image](https://github.com/user-attachments/assets/6bd8f5b3-e5b0-471b-a6de-1df188adfee3)



However, we did find that ResNet trained significantly faster, leading us to choose ResNet as our pretrained model.  
	We decided to use the Chinese Mammography Database (CMMD) for our models because it was more organized than comparable datasets, newer, and less experimented on. We hoped that using newer, less analyzed data would allow us to find novel results. The CMMD is a data set of mammography images published in 2022 with 5,563 images of mammograms from 1,775 patients with biopsy-confirmed diagnosis information. The dataset is comprised of 1416 benign and 4147 malignant images. \[https://www.kaggle.com/datasets/tommyngx/cmmd2022. 

**Preprocessing**  
Before we could train our model on the CMMD data, the data set needed to be cleaned. To clean it, we removed all label columns, except for abnormality, classification, and file location. During initial testing, we discovered that training on the uncropped images was highly ineffective because the model would only guess malignant, even with regularization techniques applied.   

![image](https://github.com/user-attachments/assets/58806d28-a7cb-4321-bed3-a780ad69f351)


We suspected that large portions of the black background surrounding the breast in the images made it difficult for the model to identify important information. To crop the images, we identified the border between the white breast image and the black background. After cropping, we padded and resized the images to 600 by 600\.

**Example of Cropping**

![image](https://github.com/user-attachments/assets/6324cbb6-3185-49f3-8386-79a82ed58147)


**Project Design (remember to link to photos)**  
Along with the ResNet-based transfer learning model, we created a smaller custom CNN made from scratch. The smaller model is valuable because at runtime it evaluates quicker than bigger models due to the simpler design. 

**Evaluations Methods**

We evaluated our models using four primary metrics. 

**Accuracy**  
Accuracy tells us how much of the test set our model correctly predicts. It is calculated by dividing the number of images correctly predicted in our test set by the total number of images in the test set. The distribution of classes within the CMMD data set was very uneven (benign: 1416, malignant: 4174), causing our models to struggle with overpredicting the majority class. However, accuracy can mask the poor performance of a model that predicts only the majority class, so we needed to utilize other metrics in our early tests before we balanced the classes.

**F1 Score**  
F1 score is calculated using precision and recall.

![image](https://github.com/user-attachments/assets/edcdce4a-3593-43e6-a5a1-7e06d00baa26)


Precision is calculated by dividing the number of correct positive identifications by the total number of samples the model labeled as positive, demonstrating how often a model misidentifies a negative sample. In our case, this would be mislabeling a benign sample as malignant, which, though it could create unnecessary work for a human reviewer, is not life-threatening.

Recall is calculated by dividing the number of true positive identifications by the total number of positive samples, demonstrating how often a model misidentifies the positive class. In our case, this would be mislabeling a malignant sample as benign, which could be life-threatening. We therefore focused on maximizing recall over precision.

The F1 score is the harmonic mean of precision and recall. It outputs a number that balances the two variables and is a helpful metric for analyzing them both at once. A high F1 score means the model does well at both catching positives and not mislabeling negatives.

**ROC Curve and AUC**

The ROC (Receiver Operating Characteristic) curve compares the true positive rate (recall) to the false positive rate across different classification thresholds. A *threshold* is the cutoff probability the model uses to decide whether a sample is classified as malignant or benign. For example, a threshold of 0.5 means the model labels any sample with a predicted probability over 50% as malignant. Lowering the threshold increases sensitivity (recall) but can also increase false positives.

The AUC (Area Under the Curve) summarizes the ROC curve in a single number, where a score of 1.0 is perfect classification, and .5 is no better than random guessing.  

![image](https://github.com/user-attachments/assets/25146e05-8644-435e-bca9-2d8f5854c891)


**Confusion Matrix**

A confusion matrix compares the model's predictions to the correct labels, demonstrating the specific classes the model misidentifies. In our case, we were able to visualize the rate of false negatives that could have been hidden when using accuracy alone.

![image](https://github.com/user-attachments/assets/ba038af6-b846-435e-9d12-2a4fc3861ede)


**Baselines**

We used three baselines to evaluate our models performance: majority guessing, random guessing, and logistic regression.

**Majority Guessing**

The original CDDM dataset comprises 5563 images, with 1416 benign and 4147 malignant.   
This means that a model that guessed malignant 100% of the time would have a 74.5% accuracy. This imbalance required us to investigate other metrics when a model had a high accuracy and seemed promising.  
On the models where we artificially balanced the data, the majority guess accuracy was, of course, 50%.

**Random Guessing**

Since our data is 74.5% malignant, a random guesser should guess malignant 74.5% of the time, and benign 26.5% of the time. This would make the accuracy (0.745×0.745) \+ (0.265×0.265) which equals 62.5%. Creating a model with lower accuracy would be pointless. 

On the models where we artificially balanced the data, the random guess baseline was again 50%.

**Logistic Regression**

To create our final baseline we made a logistic regression model, a simple binary classifier that predicts class probability. We flattened the images to 256x256 pixels and fed it to the Scikit-learn logistic regression model. By treating each pixel as an independent input feature, logistic regression provides a baseline for more complicated models like CNNs which can recognize features and should be better at image classification. 

Our logistic regression model obtained an accuracy of 68.43%, a malignant F1 score of 79, and an AUC of .62.

![image](https://github.com/user-attachments/assets/9147a7ec-dc42-4e53-89ff-626d007e254c)


The confusion matrix showed that it was not only guessing the majority class, but closer examination revealed that it guessed malignant 74.4% of the time, almost the exact same percent as there were malignant images in the data set. This, combined with the low benign precision score, made it seem that it was learning to mirror the class distribution.

![image](https://github.com/user-attachments/assets/f820eb85-322d-483d-8253-073e7a608b6b)


With this baseline in mind, we aimed for a model with higher precision for the benign class to prove that our model was learning real features of the data, not just the class distribution.

# Custom CNN Architecture: Results And Discussion

For our custom CNN architecture, we went through an iterative process. Our first goal was to overfit the data to prove that our model was learning real features of the data and not simply guessing the majority class. We used two convolutional layers, a dense layer and an output layer with no regularization.

**Custom Model: No Regularization Techniques**  

![image](https://github.com/user-attachments/assets/f5a08a01-b0c1-4cdf-a51b-59ef78170bc4)
![image](https://github.com/user-attachments/assets/8087d214-91a3-405e-8450-6c0d451a7dac)
![image](https://github.com/user-attachments/assets/c06faafe-d473-439d-819c-91eba22e7742)


This architecture obtained an accuracy of 98%, and a malignant F1 score of .83. By having such high training accuracy compared with the valuation accuracy, it was clear the model was learning each individual image in our training set, proving that it was learning image specific features- not just guessing the majority class. 

**Regularization**  
After overfitting the data we aimed to create a CNN model that still learned real features of the data but was no longer over-fitting.  
Our first architecture had two convolutional layers and a dense layer at the end. The small number of layers allows us to train faster given our limited computational abilities. It’s also a good base for us to build upon with more layers and regularization techniques. In all following custom CNN models, batch size was 32\.

![image](https://github.com/user-attachments/assets/697b5ded-97d0-4c30-a8e1-2f0c0dc88de9)


The custom CNN with no regularization techniques obtained an accuracy of 72%, an AUC of .62 and a malignant F1 score of .82. While these results were better than random guessing, the model was clearly over-predicting malignant. This led us to try out different regularization techniques. First, we tried by adding two 0.5 dropout layers as well as L2 regularization (0.01).

**Custom Model:  Dropout and L2 Regularization**

![image](https://github.com/user-attachments/assets/ff357ed7-d70c-432e-a7fa-401cf277006b)

The custom CNN with dropout and L2 regularization obtained an accuracy of 66%, an AUC of .64 and a malignant F1 score of .76. Though the accuracy was lower, the benign recall was higher than without regularization, showing that the model was learning not to rely as heavily on majority class guessing. These results led us to continue adding more regularization techniques. 

Our next attempt used dropout, L2 regularization (0.01) and class weights. Because CMMD had a significantly imbalanced dataset (1416 benign and 4174 malignant images), the model was learning much more about malignant cases and had worse accuracy when identifying the underrepresented benign class. The imbalanced classes also rewarded the model for leaning to guess the majority class. We used class weights to tell the model to pay more attention to the underrepresented class, which is a benign class in our case. 

**Custom Model: Dropout, L2 Regularization, and Class Weights**

![image](https://github.com/user-attachments/assets/b37695b2-09b0-46d2-ba5b-65b41557e58a)



Using dropout, L2 regularization, and class weights obtained an accuracy of 62%, an AUC of .62 and a malignant F1 score of .73 . Again, while overall accuracy decreased, benign recall increased to .45.  
Finally, we tried dropout with l2 and oversampling.

**Custom Model: Dropout, L2 Regularization, Oversampling**  
Our final attempt at balancing the classes was to create more benign data by augmenting the benign class pictures to get the same number of pictures for each class. We augmented/created 2758 benign images with rotation, zoom, flip, and horizontal and vertical shift. Everything else about our model remained the same. The following was our confusion matrix:
![image](https://github.com/user-attachments/assets/5fa3a86c-3bc4-4b8e-8e31-aae68717197a)
![image](https://github.com/user-attachments/assets/ff19f3f2-998e-4cfb-a6cb-89deb985ebb2)


The custom CNN with dropout, L2 regularization and oversampling obtained an accuracy of 68%, an AUC of .66 and a malignant F1 score of .78. The benign recall score increased to .47.  This model had a higher accuracy than previous attempts while the benign recall score showed it was learning not to rely as heavily on majority class guessing.  This was our final scratch CNN model, since computational limits made training times unfeasible.

**Custom Model on MIAS Dataset: Dropout, L2 Regularization, Oversampling**  
Finally, we evaluated the previous model against the MIAS dataset, in order to evaluate the generalizability. 

![image](https://github.com/user-attachments/assets/428ab70a-da00-4afe-af09-4b8bc07119db)


The custom CNN trained with  Dropout, L2 Regularization, Oversampling had an accuracy of 44%, an AUC of 49% and a F1 malignant score .57 on the MIAS dataset. These poor results demonstrated that the model had learned CMMD specific features and was not very generalizable.

#  Transfer Learning using ResNet50 

We used a pre-trained model ResNet50 on our dataset to extract features and finetune. Transfer learning is useful when there are computational limitations or limited data. Since we experienced long training time and overfitting with our from-scratch model, this was our next step. Transfer learning can also help us better generalize on different datasets since the pre-trained model has been trained on different categories of data. ResNet50 is a CNN model with 50 layers, it excels at image classification. It’s made up of convolutional layers with batch normalization and ReLU activation, which makes it powerful at learning edges, textures, and shape.   
![image](https://github.com/user-attachments/assets/c82a6087-214c-4165-862d-32651ba17f6a)


**ResNet 50**  
We froze the base model ResNet50, meaning the weights are not trainable. We then attached a new output layer in order to match our binary classification. The ResNet50 aims to extract features from our dataset, which we can pass to a few fine tuned layers in order to make the model better at our specific goal of identifying between malignant and benign.   
![image](https://github.com/user-attachments/assets/b8bd0412-7a43-4bc0-97b0-2305dc4798a2)

We also applied data augmentation to our training set to improve optimization and generalization. The test set was not augmented.  
![image](https://github.com/user-attachments/assets/8724b2e6-4dde-4c0d-8943-fd69a8d9259b)

We used ReduceLROnPlateu to decrease learning rate when the value loss stops improving, which means our weights will be updated more slowly and doesn’t make aggressive changes. We also include class weights from the previous scratch model. 

![image](https://github.com/user-attachments/assets/3d6e1743-f741-431c-8eca-90ec1851c73d)


Our model was much better at detecting malignant cases, obtaining an F1 score of 0.73, compared to benign cases which got a score of 0.46. It also had more false negatives than false positives, which is undesirable for a cancer detection model, since false negatives mean cancerous tumors marked incorrectly as benign.

**ResNet50 with Fine Tuning**  
To hopefully decrease the false negatives, we  fine tuned the resulting model by unfreezing the last 30 layers. ResNet50 is trained on the ImageNet dataset which contains many natural images, very different from our black and white breast mammograms. If we have a nicely sized dataset but different types of data, it’s better for us to unfreeze deeper layers so that the model learns more complex features specific to our task. 

![image](https://github.com/user-attachments/assets/c0532460-20c2-41ee-b08c-fbb21b0626fd)


With fine-tuning, our model got better at predicting malignant cases,  jumping from 0.73 to 0.79 F1 score, while getting no better at predicting benign cases. The overall accuracy score and AUC score also increased, showing that fine tuning helped ResNet50 learn more about our specific task. 

**Oversampling with ResNet50**  
Our dataset CMMD had a significantly imbalanced dataset:  

Benign count: 1416  
Malignant count: 4174  

Because of that, the model was learning much more about malignant cases and had worse accuracy when it came to the underrepresented benign class. Just as we did with the scratch CNN model, our solution was to augment the benign class pictures to get the same number of pictures for each class. We augmented 2758 benign images with rotation, zoom, flip, and horizontal and vertical shift to create the additional benign data. No other features of the model were changed. 


![image](https://github.com/user-attachments/assets/87eac629-8458-4072-8cf7-06250f0d9a28)

**Performance on the augmented dataset**  
![image](https://github.com/user-attachments/assets/49297092-9b4f-4241-86dc-f9e519f865b4)

![image](https://github.com/user-attachments/assets/5fc828c2-e201-42cc-a69c-73826263752e)

![image](https://github.com/user-attachments/assets/51110eaa-af35-4b9b-90c3-6d96ac5c1413)

Our accuracy increased to 78.9% and AUC to 87%. It was predicting both classes equally well, which can be visualized from the F1 scores and confusion matrices.

**Performance on original data**  
After performing these tests, we realized that the model was including the augmented benign data that we had created in the final test set. This skews the evaluation since it should only be performed on original data. When we removed the augmented data from the test set, the model proved to be less efficient at detecting benign cases but it had gotten better at detecting malignant cases.   
![image](https://github.com/user-attachments/assets/ce54b7c4-67d6-4b78-b6aa-49ec4a617e9b)

So by enhancing the underrepresented dataset, we taught the model more about the malignant cases. Our accuracy improved from 69.9% in fine-tuned ResNet50 to 72.8% in this model. But our AUC score almost stayed the same. 

**Fine-tuning the oversampled ResNet50**   
![image](https://github.com/user-attachments/assets/1f5f88a3-87c1-4f51-b2dd-49efd8f20eee)

Using the same fine-tuning measures as before had minimal improvements, with 72.8% accuracy improving to 74.6%, and 68.5% AUC improving to 69.2%. We think the model trained on the oversampled data is overfitted to the augmented benign images. It may have learned specific features the augmented photos had, like rotations and the padding from the rotations. 

**Our best model: ResNet50 Oversample with No Fine Tuning**   

![image](https://github.com/user-attachments/assets/a62681d2-e86b-4ee5-89b7-7c46b758ad9e)


Our best results came from rerunning the ResNet model with oversampling and no finetuning. This model ran for 45 hours and performed 60 epochs of 209 images each. Running the evaluation on the test set (with the augmented benign images removed) got an accuracy of 76%, a malignant recall of 95% and a AUC of .70. We believe the model might have run better without the fine tuning because we didn’t have enough data to accurately train the 30 layers of the ResNet model we unfroze. Future investigation using less fine tuning may lead to better results. 

# Challenges

Computational limits prevented us from properly training the fully custom CNN. As such, the results are not quite as impressive as our ResNet transfer model, and we believe there is more potential for the scratch model. Additionally, due to inexperience storing such a large variety of models there was some code lost that had to be reproduced after the fact, which led to delays and lost work.   
The large discrepancy in class size between malignant and benign was another challenge. We had a lot of issues with our models guessing primarily the majority class, and our methods for addressing the imbalance were not fully successful. Our best model still highly prioritized the malignant class because the augmented benign data seemed to be less helpful for training.  
In addition, our original plan involved using the CBIS-DDSM dataset ([https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)). However, the dataset was so large that it was difficult for us to run it on our computers. As well, the data was so unorganized that it was extremely difficult for us to clean properly.

**Conclusion**

In this project, we explored the use of CNN’s to determine the difference between malignant and benign tumors, potentially supplementing a radiologist's ability to diagnose breast cancers. Our work does show that there is merit to the idea of using CNNs to aid diagnosis of breast cancers. 


AI Disclosure: This github contains code produced by Generative AI
