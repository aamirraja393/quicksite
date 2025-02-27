---
title: Machine Learning for Fraud Detection
author: Aamir Raja
date: 2025-02-15
layout: post
tags: [data science, fraud detection, machine learning, artificial intelligence]
---
----
## How common is fraud?

Fraud is prevalent throughout society with common news articles showing up on the BBC such as “Victims lose £7m to romance fraud in one year” (Sinclair, 2025). Or, “Ghost broking: Young and vulnerable people targeted by insurance scam” (Pandey, 2022).

Fraud runs rampant throughout our society, yet it appears that it falls through the cracks of our financial fraud detection systems, commonly being detected one step too late.

The impacts of fraud can be disastrous for all of those who have had the unfortunate luck of dealing with fraudsters. Fraud can easily cost companies millions of pounds and can cause pensioners to lose all their savings which they have spent decades accumulating. This can be observed as the approximate cost of fraud alone is equivalent to £6.8 billion in the United Kingdom and Wales (Cleverly & Tugendhat, 2024).

Statistics such as these made me interested in understanding what types of fraud take place within society and made me ponder what can be done to minimise it and or ideally negate it. Hence, I took it upon myself to try and research such issues further to investigate what the data science solutions would be to resolve such issues, as data is a powerful tool when used in the correct hands, so I wished to analyse fraudulent transaction-based data and observe how I could deploy my knowledge of machine learning to help combat a very prevalent real-world finance issue.

In this article, I shall be exploring and discussing the machine learning (ML) methods I deployed for fraudulent transaction detection the issues surrounding detecting fraudulent transactions when deploying machine learning models for such purposes, and any general insights I may have in relation to such topics to share my understanding of such topics.

------

## What are the different types of fraud?
Fraud can be defined as “criminal deception intended to result in personal or financial gain”. I believe this definition encompasses the different types of fraud that we may be discussing throughout this article, which can take the form of:

- Insurance fraud 
- Pension fraud.
- E-commerce fraud.
- Credit card fraud.
- Mobile fraudulent transactions.
- Romance fraud.

Fraud is a vast domain for criminals to tap into for significant financial gain, and fraudulent behaviour may come from one individual who is running their operations, or a group of individuals working with one another to ensure that the scam runs as smoothly as possible to minimise the probability of detection and possible criminal charges being pressed against them.

Credit card fraud alone is a prominent cause of concern for the banking industry and global financial health. In the Single Euro Payment area, we can observe that credit card fraud amounted to 1.8 billion Euros in 2019 and the total loss from credit card fraud amounted to $21.84 billion, which was observed globally that year (Raghavan & Gayar, 2019).

Credit card fraud is a transactional type of fraud that can also occur within seconds, as fraudsters are adapting to common fraud prevention strategies deployed by financial institutions, and although financial institutions are adapting their strategies to attempt to keep up with fraudsters sometimes this alone is not enough, as displayed by our statistics above.

Credit card fraud also has many forms and can fall into multiple domains such as:

- Account takeover.
- Application fraud.
- Lost or stolen credit card.
- Non-received items fraud.
-----

## Why is it so difficult to simply stop these fraudsters?

The primary challenge in fraud prevention is fraud detection; without it, fraud goes unnoticed and fraudsters will be able to steal the information or data they need to pull off their quick heist and reap the financial rewards from such lucrative pursuits.

Data scientists find it especially difficult to get access to significant quantities of sensitive, personal financial data to analyse and deploy the most efficient or effective machine learning (ML) fraud detection methods.

Financial data contains a lot of sensitive information, and using such data would be very helpful in creating machine learning models for credit card fraud detection. However, compliance rules must be maintained in a skilled manner. Also, areas such as GDPR, and data privacy must be respected, and consent must be provided for such information to be used for the training and development of machine learning models. These policies make fraudulent transaction detection a highly sensitive and restricted field, limiting data scientists’ access to crucial data and potentially limiting ML model deployment and development.

In addition, security concerns play a significant role in handling such sensitive information. This is because financial data may contain very personal data about individuals’ date of birth, income, address, marital status, mortgage payments, and more. Therefore, access remains extremely limited to maintain privacy and this is an area data scientists must respect and work around, but this is an area that does add an element of difficulty to such ML modelling tasks.

Furthermore, fraud detection datasets appear to be extremely imbalanced which adds to the tasks of data scientists. This dataset will usually contain a significant amount of non-fraudulent transactions and an infinitesimal proportion of fraudulent transactions, as they’re sprinkled throughout a dataset which can sometimes contain millions of transactions or more. Therefore, data scientists must utilise effective methods for balancing the data before they attempt to detect and minimise fraud.

------

## What is machine learning and how is it used for fraud detection?

Machine learning revolves around exposing machines to human data, which over time will allow the machine to detect patterns, and learn the way that humans learn, adapt, and evolve their fraudulent behaviours all of which is usually done through the use of artificial intelligence (IBM, 2025).

Machine learning is a common tool used by data scientists across a variety of industries to resolve fraud detection and to aid with fraud prevention, and some of the ML models that typically tend to be used for such measures are:

- Logistic regression.
- Random forest.
- Artificial neural networks (ANNs).
- K-means clustering.
- XGBoost.
- Support vector machines (SVMs).
- K-nearest neighbours.
- Convolutional neural networks (CNNs)

Machine learning models can also fall into the categories of supervised and unsupervised learning models.

------

## Unsupervised learning

Unsupervised machine learning models can identify patterns and structures within data without being provided with a focus variable, unlike supervised machine learning, which requires labelled data. These models do not need labelled data because the algorithms search for shared characteristics between instances or patterns within the data, proceeding to cluster and group them accordingly.

Unsupervised machine learning techniques are commonly used in fraud detection, product segmentation, image recognition, scientific discovery, and movie recommendations on streaming platforms like Netflix.

Some of the ML models that typically fall into this category are:

- K-means clustering
- Principal component analysis (PCA)
- Autoencoders
------

## Supervised learning
This type of machine learning is defined by the use of input and output variables. Within this, the model is trained using input data and output data, and it works through the use of labelled data, which is data that contains features(input) and targets (output). The models deployed attempt to understand the relationship between the input data that is labelled and the output data used to train.

This allows the model to provide predictions or classifications when given unseen instances.

ML models that fall into this category will typically be:

- XGBoost
- Logistic regression
- Random forest
- Linear regression
- Support vector machines (SVMs)
- Decision trees

Overall, models used for ML will be used for classification or regression challenges.

Fraud detection and prevention in nature would be considered to be a classification problem that is being attempted to be solved through the use of ML models. Therefore, it’s only appropriate we explain this below:

-------

## Classification problem

Classification problems will range around issues such as fraud detection because as a Data Scientist, you will be provided with significant amounts of data that haven’t been placed into any category and therefore require an ML model to do this for you. E.g. Fraudulent transaction data will have two types of data referred to as the majority class (legitimate transactions) and the minority class (fraudulent transactions). Hence, it will be the ML model’s job to predict the likelihood of a transaction being fraudulent based on trends and patterns shown to it.

-------

## Results

Although I ran tests on my dataset using XGBoost, random forest, logistic regression, and k-means clustering. The results displayed the XGBoost model as the most effective at fraud transaction detection.

One of the most effective models that were shown throughout the literature was XGBoost and through my findings, I was also able to replicate this with my original findings.

I then attempted to further optimise the model through the use of hyperparameter optimisation, which is a series of tweaks that an individual can make for each of their ML models to further improve their performance and accuracy when it comes to fraud detection.

-------

## Conclusion:

Overall, incentives for fraud have never been higher and so fraudsters will continue to innovate and adapt their fraudulent transaction to overcome typical fraud detection methods used by industries such as finance. However, finance industries simultaneously have been attempting to invest heavily in research and development to counter the attempts of fraudsters and detect fraud to protect their businesses and ensure the safety of their customers’ transactions.

ML models for fraudulent transaction detection appear to be improving with the significant improvements in artificial intelligence, hardware capabilities, and significantly greater supplies of highly skilled data scientists in the workplace to help thwart such fraudulent behaviour.

From my research and observations, XGBoost appears to be incredibly effective at detecting fraudulent transactions with the random forest model being a close contender in second place. Both models demonstrated strong performance due to their ability to optimise a wide range of hyperparameters, which we did. However, XGBoost was the most effective for such classification tasks, due to robust default hyperparameters and built-in boosting framework built into the model.

------

## Also!

If you enjoyed this article, please feel free to read my other articles where I regularly post about new data science topics and content to help inform you of the latest data science trends and foundational topics.

Have a great week ahead! 👋

-------

## References:
Cleverly, J. and Tugendhat, T. (2024) Major campaign to fight fraud launched, GOV.UK. Available at: https://www.gov.uk/government/news/major-campaign-to-fight-fraud-launched#:~:text=The%20estimated%20cost%20of%20fraud,a%20decline%20now%20being%20observed. (Accessed: 15 February 2025).

IBM (2025) What is machine learning (ML)?, IBM. Available at: https://www.ibm.com/think/topics/machine-learning (Accessed: 15 February 2025).

Raghavan, P. & Gayar, N.E., 2019. Fraud detection using machine learning and deep learning. 2019 International Conference on Computational Intelligence and Knowledge Economy (ICCIKE), Dubai, United Arab Emirates, pp. 334–339. doi:10.1109/ICCIKE47802.2019.9004231.

Pandey, M. (2022) Ghost broking: Young and vulnerable people targeted by insurance scam, BBC News. Available at: https://www.bbc.com/news/newsbeat-61992772 (Accessed: 15 February 2025).

Sinclair, E. (2025) Romance fraud: Victims in Surrey and Sussex lose £7M in one year, BBC News. Available at: https://www.bbc.co.uk/news/articles/czrldx6zk8ko (Accessed: 15 February 2025).

