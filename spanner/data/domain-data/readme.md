*Source: https://www.kaggle.com/datasets/bhavikardeshna/yahoo-email-classification?resource=download*

## About Dataset
The Yahoo! Answers topic classification dataset is constructed using the 10 largest main categories. Each class contains 140,000 training samples and 6,000 testing samples. Therefore, the total number of training samples is 1,400,000, and testing samples are 60,000 in this dataset. From all the answers and other meta-information, we only used the best answer content and the main category information.

- Society & Culture
- Science & Mathematics
- Health
- Education & Reference
- Computers & Internet
- Sports
- Business & Finance
- Entertainment & Music
- Family & Relationships
- Politics & Government

The Yahoo! Answers topic classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the above dataset. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015)

## Format
The files train.csv and test.csv contain all the training samples as comma-separated values. There are 4 columns in them,

- Class Index (1 to 10)
  - Society & Culture
  - Science & Mathematics
  - Health
  - Education & Reference
  - Computers & Internet
  - Sports
  - Business & Finance
  - Entertainment & Music
  - Family & Relationships
  - Politics & Government
- Question Title
- Question Content
- Best Answer.
  
The text fields are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".