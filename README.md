# INF368-Final-Project
Final project for INF368

## Papers
- [Do Not Have Enough Data? Deep Learning to the Rescue!](https://arxiv.org/pdf/1911.03118.pdf)
- [GAN-BERT: Generative Adversarial Learning for
Robust Text Classification with a Bunch of Labeled Examples](https://aclanthology.org/2020.acl-main.191.pdf)

## Data
[IMDB data set](http://ai.stanford.edu/~amaas/data/sentiment/) contains 50,000 documents with 2 categories.

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
