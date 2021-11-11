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
1. Get the data. IMBD and Medical Text are already supplied, but if you wish to use other datasets, you need to generate the required files yourself. The required files are six different files that should all be in data/YourDatasetName/. These are on the form of:
      - These four files each one column called text with the text, and a column with labels called label. The number at the end indicates the number of datapoints per label:
            - train_labeled_5.csv
      
            -train_labeled_10.csv
      
            -train_labeled_25.csv
      
            -train_labeled_50.csv
      
      -The unlabeled data must also have two columns, text and label. The label should be blank, although it doesn't really matter as long as it exists:
      
            -train_unlabeled.csv
      
      -The test data should also be on the form of two columns, text and label. 
      
            -test.csv
      
      It is very important that the names are exactly as written, on the specified form.
      
2. Generate results for the different datasets by:
   1. Running Bert.ipynb for the baseline results.     
   2. Running GanBert.ipynb for the Gan-Bert results.
   3. Running Lambada.ipynb for the Lambada results.     
   Each notebook will have a variable you need to change in order to point it to the correct dataset. This will have to be changed manually.
  
3. Run Results.ipynb to concatenate the results and save them to a single .csv file along with a plot. They can now be compared.

## Sources
- https://github.com/mcelikkaya/medium_articles/blob/main/gtp2_training.ipynb
- https://github.com/crux82/ganbert-pytorch
- https://www.tensorflow.org/text/tutorials/classify_text_with_bert#about_bert
