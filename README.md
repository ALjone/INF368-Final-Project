# INF368-Final-Project
This was made for a final project for INF368 at the University of Bergen. We replicate two models, GanBert and LAMBADA, based on the papers below. These are generative models that help in low-data situations. GanBert uses unlabeled data while LAMBADA synthesizes its own data based on GPT2. These are then used to try to improve classification performance on the IMDB dataset and the Medical Text dataset. We then compare the results to a baseline using a BERT classifier. 

## Papers
- [Do Not Have Enough Data? Deep Learning to the Rescue!](https://arxiv.org/pdf/1911.03118.pdf)
- [GAN-BERT: Generative Adversarial Learning for
Robust Text Classification with a Bunch of Labeled Examples](https://aclanthology.org/2020.acl-main.191.pdf)

## Data
[IMDB data set](http://ai.stanford.edu/~amaas/data/sentiment/) contains 50,000 documents with 2 categories.
[IMDB data set](https://www.kaggle.com/chaitanyakck/medical-text) contains 14438 documents with 5 categories.

## Setup
| variable  | value   |
|---|---|
| # per label  | 5, 10, 25, 50  |
| batch size  |  5 |
| learning rate |  5e-4 |
| seed  | 0  |

## How to run the analysis
- Download the notebooks in the "Notebooks" folder and run them in Google Colab.
 1. Run Bert.ipynb for the baseline results.     
 2. Run GanBert.ipynb for the Gan-Bert results.
 3. Run Lambada.ipynb for the Lambada results.     
 
 Note. if one need to generate the data, one can run the GenerateData.ipynb file.  

## Sources
- https://github.com/mcelikkaya/medium_articles/blob/main/gtp2_training.ipynb
- https://github.com/crux82/ganbert-pytorch
- https://www.tensorflow.org/text/tutorials/classify_text_with_bert#about_bert
