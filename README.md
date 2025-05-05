# Custom CNN Architecture: Results And Discussion

For our custom CNN architecture, we went through an iterative process. Our first goal was to overfit the data to prove that our model was learning real features of the data and not simply guessing the majority class. We used two convolutional layers, a dense layer and an output layer with no regularization.

**Custom Model: No Regularization Techniques**

****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdCuNfm8vF6FeBTYLjlrNCWZUe-CTTJW85AI59DNGrZzYL5-vrhC12k4iVn7swczKwsVo_xj1KiwET7P6ricG3BQfZ6jWfY9Jc912v4ox5UhGYFZJCoSd9s9CQjqLw_y1Xbd_37mg?key=DDkPovtG6eUAdQ0LEiRgPMLX)****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeS3CHqOA0B_YK34gRIH97yWR2-mjJEkD0SCJtHtQJn92L4cVrNxvJ4QSlrvelM2U_w0QxaDv4gPjdLBozLL7uf0Iqn4QEeD3ep2cbpQsqLqZX1mfYAir2llD0pu7RRu2phiEpoVg?key=DDkPovtG6eUAdQ0LEiRgPMLX) ****

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdt2VQFsbBSQET6tL33ukhhH-Vfjo6sR-ZIDnni32mxp-q1f4T6ePl-nrs5kVffiDxKbG-wDeVCtJPwRDodU54QHHI-n5KZvVbn164aClimbVoJ4Dw6R-DH2G3GpfgYduBKLeUvog?key=DDkPovtG6eUAdQ0LEiRgPMLX)

This architecture obtained an accuracy of 98%, and a malignant F1 score of .83. By having such high training accuracy compared with the valuation accuracy, it was clear the model was learning each individual image in our training set, proving that it was learning image specific features- not just guessing the majority class. 

**Regularization**

After overfitting the data we aimed to create a CNN model that still learned real features of the data but was no longer over-fitting.

Our first architecture had two convolutional layers and a dense layer at the end. The small number of layers allows us to train faster given our limited computational abilities. It’s also a good base for us to build upon with more layers and regularization techniques. In all following custom CNN models, batch size was 32.

The custom CNN with no regularization techniques obtained an accuracy of 72%, an AUC of .62 and a malignant F1 score of .82. While these results were better than random guessing, the model was clearly over-predicting malignant. This led us to try out different regularization techniques. First, we tried by adding two 0.5 dropout layers as well as L2 regularization (0.01).

**Custom Model:  Dropout and L2 Regularization**

The custom CNN with dropout and L2 regularization obtained an accuracy of 66%, an AUC of .64 and a malignant F1 score of .76. Though the accuracy was lower, the benign recall was higher than without regularization, showing that the model was learning not to rely as heavily on majority class guessing. These results led us to continue adding more regularization techniques. 

Our next attempt used dropout, L2 regularization (0.01) and class weights. Because CMMD had a significantly imbalanced dataset (1416 benign and 4174 malignant images), the model was learning much more about malignant cases and had worse accuracy when identifying the underrepresented benign class. The imbalanced classes also rewarded the model for leaning to guess the majority class. We used class weights to tell the model to pay more attention to the underrepresented class, which is a benign class in our case. 

\
\
\
\
\
\
\
\


**Custom Model: Dropout, L2 Regularization, and Class Weights**

Using dropout, L2 regularization, and class weights obtained an accuracy of 62%, an AUC of .62 and a malignant F1 score of .73 . Again, while overall accuracy decreased, benign recall increased to .45.

Finally, we tried dropout with l2 and oversampling.

**Custom Model: Dropout, L2 Regularization, Oversampling**

Our final attempt at balancing the classes was to create more benign data by augmenting the benign class pictures to get the same number of pictures for each class. We augmented/created 2758 benign images with rotation, zoom, flip, and horizontal and vertical shift. Everything else about our model remained the same. The following was our confusion matrix:![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXf81z6kL3N2ywpqceWbWtgPWb3Jt-zy6BFNC5TgF_U91pQTjyqcth6PT5lWYrfPs0EGXv0OASrclizwA-h9_lAGjCeVFRe32NXLdl5klNCb-F-X0QyMaQaOwIQ6-MAdM5Fe1NN0mg?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXew_NiDy-T21Usufjuhp1lawYsiny8fqww_73uS8hRppQhx7u6c1hCk-eIscVJfZKdB87hGkhuLaCnIP8xit6k1LAkcRa3gkSDEvLg6GYc-QAiEkTHpVkjM5GD4g8mEP29cC0vU5Q?key=DDkPovtG6eUAdQ0LEiRgPMLX)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfEYDu5z8FQJtpQ9WnQLfphGAIsgNw3ILyR1h4vSy2z9wcInm0_Eqkfug1cIQFuPmV1aQ35InBuCSsaboMYQW3S17e5XYpOT2j0XEogz13GdEx0gpSeTB0rZaXhjyof24S5_SzTSg?key=DDkPovtG6eUAdQ0LEiRgPMLX)

The custom CNN with dropout, L2 regularization and oversampling obtained an accuracy of 68%, an AUC of .66 and a malignant F1 score of .78. The benign recall score increased to .47.  This model had a higher accuracy than previous attempts while the benign recall score showed it was learning not to rely as heavily on majority class guessing.  This was our final scratch CNN model, since computational limits made training times unfeasible.

**Custom Model on MIAS Dataset: Dropout, L2 Regularization, Oversampling**

Finally, we evaluated the previous model against the MIAS dataset, in order to evaluate the generalizability. 

The custom CNN trained with  Dropout, L2 Regularization, Oversampling had an accuracy of 44%, an AUC of 49% and a F1 malignant score .57 on the MIAS dataset. These poor results demonstrated that the model had learned CMMD specific features and was not very generalizable.

**Transfer Learning using ResNet50** 

We used a pre-trained model ResNet50 on our dataset to extract features and finetune. Transfer learning is useful when there are computational limitations or limited data. Since we experienced long training time and overfitting with our from-scratch model, this was our next step. Transfer learning can also help us better generalize on different datasets since the pre-trained model has been trained on different categories of data. ResNet50 is a CNN model with 50 layers, it excels at image classification. It’s made up of convolutional layers with batch normalization and ReLU activation, which makes it powerful at learning edges, textures, and shape. 

[![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcD8i9whxz0ilRXLXt8TxNawU1NGllnBqyoaYkFlWG0M978EnRW-8rnq_MNC8-AMtktyLRE1hXfpcYBzVRXjyJk5wZEuzUMKqcpSHVSnVEAGejFj_V2VjEeSK9KJw1Dz5sd-M_1Qw?key=DDkPovtG6eUAdQ0LEiRgPMLX)](https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758/)

**ResNet 50**

We froze the base model ResNet50, meaning the weights are not trainable. We then attached a new output layer in order to match our binary classification. The ResNet50 aims to extract features from our dataset, which we can pass to a few fine tuned layers in order to make the model better at our specific goal of identifying between malignant and benign. 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdD7cwBipI0b9aOmloWGdYNg8iMT4fjCUmA4V9uhxVU8DyLlz3zQ2hW9EJU662QPFiR5lXvgT2TfZetCFhDEKpC7LC0Nq3dQHH7h6-9mCxY4MHCyplly47cQmGaKQAymmC7yNS7pg?key=DDkPovtG6eUAdQ0LEiRgPMLX)

We also applied data augmentation to our training set to improve optimization and generalization. The test set was not augmented.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfEEZNnfyCJkecetVKJU6XQXNS4jwzAVVy-M1wx27C5sHE1Hl7G5LCMkwi8DHaNzjmMoAyVzow_EJ5aDHwit2932kEMutJdMIQdq2YoystZe7Wad-NrWED4cBI_q83CJPup-pnqhA?key=DDkPovtG6eUAdQ0LEiRgPMLX)\
![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcViS5gTpUi531_c0AdKkZBXp2IbLBNVa-ZuStOyNpu6c3NgeLYzb6LO4OIegW32l6YxCo7dExWw4JYlE6hoqbvfLYtFbCJScJc9xw-JioxGGT0DpRXM_7FrW49My1rMtu86sZ3KQ?key=DDkPovtG6eUAdQ0LEiRgPMLX)

We used ReduceLROnPlateu to decrease learning rate when the value loss stops improving, which means our weights will be updated more slowly and doesn’t make aggressive changes. We also include class weights from the previous scratch model. 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXevohz2ZQGo4VoFq_Tf54dGgxsmIn6VWsSTNWBOAECCsrjZU5uQUakBC-hcYKvsznc30Eu_LuSN1bbTAwZipE-Ep7yUCz3PQY7z1OeD7TH2ssgeTmAgQXr1fZPBj6AXoY9QO1k7TQ?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfAP5SjS8d9HtFsgLam0zNpz7ENCKpQaRjfS41OAZSEGCjrF5_ObvfVOPtZkgH3gpADT6gWNenf2K72geoz44w-sbeTrBbF7-VhKeSCnj2huW5-Nn26oQpyYqXs5t-MZXC2GtLWQQ?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXevohz2ZQGo4VoFq_Tf54dGgxsmIn6VWsSTNWBOAECCsrjZU5uQUakBC-hcYKvsznc30Eu_LuSN1bbTAwZipE-Ep7yUCz3PQY7z1OeD7TH2ssgeTmAgQXr1fZPBj6AXoY9QO1k7TQ?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdEri95qzBqyDdqlLXCnPCt6GqUpvD45x9g_dGjXQWmwopAyWUWCT1FI2HZrt3q8n9atq1Uc9eMMWc4OcSxOC7b5u33SsS-Oq1FMpZj7Jt4dnE4WFRSykRvs2I597HnpZHYop7YMg?key=DDkPovtG6eUAdQ0LEiRgPMLX)

\


Our model was much better at detecting malignant cases, obtaining an F1 score of 0.73, compared to benign cases which got a score of 0.46. It also had more false negatives than false positives, which is undesirable for a cancer detection model, since false negatives mean cancerous tumors marked incorrectly as benign.

**ResNet50 with Fine Tuning**

To hopefully decrease the false negatives, we  fine tuned the resulting model by unfreezing the last 30 layers. ResNet50 is trained on the ImageNet dataset which contains many natural images, very different from our black and white breast mammograms. If we have a nicely sized dataset but different types of data, it’s better for us to unfreeze deeper layers so that the model learns more complex features specific to our task. 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcHPVglyf1JhfZOzhJv-Sa_PinyHRm13-gE1ueo5ytkJ1YRPlFDc2UrLeNaiZ8xOfxguXUaMp3iS6u2jYqPPDRDyp9t7eC3Oaoyl8GohDq7NVtVFCU2GRK4UuhsqlmqHdhw1wCZ7A?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfhGYsk5BlibGaCDf861ot5jE9lRWO6xFn7dLu3vgn6H5uYDj6lrVd6zUF7wpialoYR3YwM2D5HSkXJJKFZV_wR9Q8RyEYqgSGzz8A0TSFq9v302MMHB-Uu8fpIi1JKgzJxSquuGQ?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcHPVglyf1JhfZOzhJv-Sa_PinyHRm13-gE1ueo5ytkJ1YRPlFDc2UrLeNaiZ8xOfxguXUaMp3iS6u2jYqPPDRDyp9t7eC3Oaoyl8GohDq7NVtVFCU2GRK4UuhsqlmqHdhw1wCZ7A?key=DDkPovtG6eUAdQ0LEiRgPMLX)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXc6yJgoxIjETM4Y5SxBjigAwl9Ax8ISg62WnWdqiNQeWGdPbH6D2yNjByia-xGCCnbjsfXGuRMIuThQ5gvem1giYsUZl6wzGhp7rxYSdrntVvyqkpo88WGywQ94h07JG_-Km012hA?key=DDkPovtG6eUAdQ0LEiRgPMLX)

With fine-tuning, our model got better at predicting malignant cases,  jumping from 0.73 to 0.79 F1 score, while getting no better at predicting benign cases. The overall accuracy score and AUC score also increased, showing that fine tuning helped ResNet50 learn more about our specific task. 

**Oversampling with ResNet50**

Our dataset CMMD had a significantly imbalanced dataset:

Benign count: 1416

Malignant count: 4174

Because of that, the model was learning much more about malignant cases and had worse accuracy when it came to the underrepresented benign class. Just as we did with the scratch CNN model, our solution was to augment the benign class pictures to get the same number of pictures for each class. We augmented 2758 benign images with rotation, zoom, flip, and horizontal and vertical shift to create the additional benign data. No other features of the model were changed. 

- Final training set: 8348 images (Benign: 4174, Malignant: 4174)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdOviTajBMKq8Nv4Khd6rKMa7Bo_Zb-tByaI_duX2vVQ3pEirrEZWuO_Wz7aWvaUfw_1sDkyd9KKv1Uh0883k4SAd-8zdhHWJbUQzB4AQ2BvMO_bwtZ_cnPdlugne_GX4YSM979vA?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdCxbKIGcpBcHk3QewK2HV93hoQwhFBl7Osqy96qivX6hMhNlQWWmP5Vp3e1xNYIIVT2L2MvFJqL2WW0PFvTcfyoX9EOuGRg5zIo-9L-27PfFYDnBg0GjW7zb4j4pzoZypp854y?key=DDkPovtG6eUAdQ0LEiRgPMLX)                                              ![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXc615A_6UTq-vg7EAGwx5JKhtQ0lZwvsRam5FobMTdnpErl5yzJXXoKSIB-PYgBzomte38grDVupo86Vrh1oJVPW078KTO4yDKyDBpxLNGc99FqvmNzubTC1lt9LScdz9qzC2ZzTQ?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe_HQ8s38gqRXwqlgM12Ho4SN_aAxmHSncptUXS9IBG_B_3ngKDsY4Jlx67JB1HPrXdrciuCNtdUjknUZybHAJaPb5-pD4QqGTgClTIeTjS7oKQg15gtIU0YNUkVErG16jfXuJVmg?key=DDkPovtG6eUAdQ0LEiRgPMLX)

THE MODEL RAN FOR 18 HOURS

**Performance on the augmented dataset**

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXex9yjN5UnEedsJWV3WhzjPixdv5WEiUWRDHWOLGkxrE6HMcRri2h12kNlGHEiylKGp5L-D-UBjDKCC694DBE_OrZadkbuvESLLlBjRidRyebZGy-8wrGt1CWho_B-cWojjk2iwnw?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcZgJt7ypsFzv3LY-aYsNLy1sAqy3vqsYQgEEqr3LEIN0ehNfnswhpNzf6S9bMs9I7G6LRwQAtgFoGcFISusPqD4UpySHYezTbRzG3kOKsxcIm1VSNxYLpo1XnfqauZI6wrQRBE6A?key=DDkPovtG6eUAdQ0LEiRgPMLX)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXf5G-UilYNzCnUzh0MabZ3sbjRQU2TGO0LnMf6DWbsxibo9PYoE3w7bVLe4h4znYDJRv2zAy9P7VENH2A671pF7ciNppUxfGbza1Ywa80CjxPm00Nlg3xHxUGqgfEQsxWVySmFI?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfMAWZPZubEFhOR1Ek5uq7ii_BsreGug0rwVuBApuOmEpbQFSqq__GV3FDfKpzA3r8VZCCXSL9lctCGvaw5kygonYIrhiic0PdpVFjQcI6W-yB4r4QtsA-au0b8JRtFQODdIRGN_A?key=DDkPovtG6eUAdQ0LEiRgPMLX)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXebITnIYZmNBFRwp8Of_RpA1TjBBK9324Xl9w222QB22kYCCGfqPk0bzA6LiQH25FlQviUyjYGZZqJUejxZRLDOUzPvEU4ThuTZ8XKjhKhmJGk5ZOCXMTEjJ6s8QKpwWCu6A9qPzg?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXerv97ZuPbaluFpZzketQWsw3cSesAF2uvjqfagVlrXK3H8PLY2LE9YgxJZ07FXEDyIvoWQtYSJsVJeluHmXnzqAS-RoIGTzymc7u5cXEvxmKDi3kbQucDjJoX1vR18Tv2yIXgP?key=DDkPovtG6eUAdQ0LEiRgPMLX)

Our accuracy increased to 78.9% and AUC to 87%. It was predicting both classes equally well, which can be visualized from the F1 scores and confusion matrices.

**Performance on original data**

After performing these tests, we realized that the model was including the augmented benign data that we had created in the final test set. This skews the evaluation since it should only be performed on original data. When we removed the augmented data from the test set, the model proved to be less efficient at detecting benign cases but it had gotten better at detecting malignant cases. 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcWsOek_RrGy9pd7lnYtdzhaldCmK_Omwb3OQz-KLskM3ShCiq5iCxOkrN7yXbielvfKKyLSCMsOprMjQ_l_BLjneXLro6X6dcJX0P8DLmCzRFacbpewBNRzdTwPzFXrsFs2fULCA?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXecpbVpc2gTez44IitcOsG1CxU4Ndp_ZYNlCzER5T8b55FB8hcWQhAouapG4hwhDCjvh0CJkPl9Y_7jrJxPz3ySz6CM9GPOhBS8utPlGcyyIIE2vx0PvOGDbICVYBwTXp-MttTD?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXehRXXwQChrhvq1olHI6Oroy_KNMd4BJYcl1w_nJW-uQIJswe2Nlzh6dtJn_mbPLIvg5lD7Q5rYdX7y6o3eyhvtRQYNmrmECAquWLLQ5-0F9o1dl3Q4EiFSYJGJtNTNh2Xtq_pDjA?key=DDkPovtG6eUAdQ0LEiRgPMLX)

So by enhancing the underrepresented dataset, we taught the model more about the malignant cases. Our accuracy improved from 69.9% in fine-tuned ResNet50 to 72.8% in this model. But our AUC score almost stayed the same. 

**Fine-tuning the oversampled ResNet50** 

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcr3dAkY1_GsHk6AaO_WPTgb4B2m0Z0t37D9k25JYndFcQw4sOD1NZelvvU_8h6uLAt5tJVbzbjY9CtIALV3GhR6hRf--OUtCDfKk2Ve92fklqFj5XYj98xUOai4QCKrmX-WwacaA?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXd6MyavBA-dLeSRz-Vpa4AwUt14BdRpOxz82OAuBt_AgGsg4klpSmmlGcW3l5vnCwV-7uA2yhAuvE8Sv2bhC_GxYvW-01PfkQFqa9X0873i4zoQvT8eACJ6_oYKCJGUQTJaLFqY5Q?key=DDkPovtG6eUAdQ0LEiRgPMLX)![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfGCnZkYYxdd_knaEm4m8h2kD_EYorl1BGAUcfvU32KZ_Ih4oklNg8phuAHUBrST2ICOpaaNAeKuWLShE-xlEFjaDE3smO-lCf9IgakbDWIYgCgYpeT2bejGZrzcpd-_03te9Zlhw?key=DDkPovtG6eUAdQ0LEiRgPMLX)

Using the same fine-tuning measures as before had minimal improvements, with 72.8% accuracy improving to 74.6%, and 68.5% AUC improving to 69.2%. We think the model trained on the oversampled data is overfitted to the augmented benign images. It may have learned specific features the augmented photos had, like rotations and the padding from the rotations. 

**Our best model: ResNet50 Oversa**![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdrE_yXLRsegMMAqK18c_vB4p3-Pky1bQluS46xWYx1aWkgja_8ogOFeKYUZAIeHIwMAWDmz5LRSbkuzScRn6RIdQqMF6ABB7tdJjwcxg_YxOszbKS0jkZt4yf3Wj9xO-VHM1IsPQ?key=DDkPovtG6eUAdQ0LEiRgPMLX)**mpled**

 ****![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXey5somuicy2KlV0rbQNeJ-qlIeiHss19gxb123NJDq7O8Mb8oEvRPxKGMb31BhsM45S9tpZTjJVXMZj619MD2pjFOxnvzYJSa8ziAE8vlc1MBnyZM2rzs-Oh16eHRw0bLWElcRbg?key=DDkPovtG6eUAdQ0LEiRgPMLX)****

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcRn93W4EHpLNxrmUqOZAnMzgebqbx_05Mci0gEgHr3ggyJBv0p43PxUuPmPwEbTLlQ6fYogjZZzO3_dJCUqksVRQbHdUy8N-v1_kv-UwhN7QbL3wkTNTk7X20eC7Irjdi2hFA0?key=DDkPovtG6eUAdQ0LEiRgPMLX)

Challenges

Computational limits prevented us from properly training the fully custom CNN. As such, the results are not quite as impressive as our ResNet transfer model, and we believe there is more potential for the scratch model. Additionally, due to inexperience storing such a large variety of models there was some code lost that had to be reproduced after the fact, which led to delays and lost work. 

**Conclusion**

\


In this project, we explored the use of CNN’s to determine the difference between malignant and benign tumors, potentially supplementing a radiologist's ability to diagnose breast cancers. Our work does show that there is merit to the idea of using CNNs to aid diagnosis of breast cancers. 
