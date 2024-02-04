# deep-learning-challenge
Module 21 Challenge (Python / Neural Networking) - Wassim Deen

# Table of Contents
1. [Notes](#notes)
2. [Final Repository Structure](#final-repository-structure)
3. [Scenario Description](#scenario-description)
4. [Summary of Challenge](#summary-of-challenge)
5. [Neural Network Model Performance Report](#neural-network-model-performance-report)

# Notes
- Three attempts of optimising the model were conducted as it was not able to meet the target performance of higher than 75% accuracy rating.
- The Neural Network Model Performance Report can be found [here](#neural-network-model-performance-report) in this README.md file.
- Steps #1 & #2 were completed in the first attempt of improving the model performance (`./Attempt #1 (Manual)/AlphabetSoupCharity.ipynb`)
- Results of each model iteration are exported as an HDF5 file within their respective folder.

# Final Repository Structure
```
├── README.md
├── Attempt #1 (Manual)
├── Attempt #2 (Auto-Optimisation)
└── Attempt #3 (Manual Optimisation)
    ├── AlphabetSoupCharity_Optimisation2.ipynb
    └── AlphabetSoupCharity_Optimisation2.h5

```

# Scenario Description
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organisations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organisation, such as:

- `EIN` and `NAME` — Identification columns
- `APPLICATION_TYPE` — Alphabet Soup application type
- `AFFILIATION` — Affiliated sector of industry
- `CLASSIFICATION` — Government organisation classification
- `USE_CASE` — Use case for funding
- `ORGANIZATION` — Organisation type
- `STATUS` — Active status
- `INCOME_AMT` — Income classification
- `SPECIAL_CONSIDERATIONS` — Special considerations for application
- `ASK_AMT` — Funding amount requested
- `IS_SUCCESSFUL` — Was the money used effectively


# Summary of Challenge
- Step 1: Preprocess the Data
    - Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, you’ll need to preprocess the dataset.
    - Follow the instructions to complete the preprocessing steps:
        1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
            - What variable(s) are the target(s) for your model?
            - What variable(s) are the feature(s) for your model?
        2. Drop the `EIN` and `NAME` columns.
        3. Determine the number of unique values for each column.
        4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
        5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
        6. Use `pd.get_dummies()` to encode categorical variables.
        7. Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.
        8. Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.

- Step 2: Compile, Train, and Evaluate the Model
    - Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organisation will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
        1. Continue using the Jupyter Notebook in which you performed the preprocessing steps from Step 1
        2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
        3. Create the first hidden layer and choose an appropriate activation function.
        4. If necessary, add a second hidden layer with an appropriate activation function.
        5. Create an output layer with an appropriate activation function.
        6. Check the structure of the model.
        7. Compile and train the model.
        8. Create a callback that saves the model's weights every five epochs.
        9. Evaluate the model using the test data to determine the loss and accuracy.
        10. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

- Step 3: Optimise the Model
    - Using your knowledge of TensorFlow, optimise your model to achieve a target predictive accuracy higher than 75%.
    - Use any or all of the following methods to optimise your model:
        - Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
            - Dropping more or fewer columns.
            - Creating more bins for rare occurrences in columns.
            - Increasing or decreasing the number of values for each bin.
        - Add more neurons to a hidden layer.
        - Add more hidden layers.
        - Use different activation functions for the hidden layers.
        - Add or reduce the number of epochs to the training regimen.
    
    1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimisation.ipynb`.
    2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
    3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimising the model.
    4. Design a neural network model, and be sure to adjust for modifications that will optimise the model to achieve higher than 75% accuracy.
    5. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimisation.h5`.

- Step 4: Write a Report on the Neural Network Model
    - For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
    - The report should contain the following:
        1. <b>Overview of the analysis</b>: Explain the purpose of this analysis.
        2. <b>Results</b>: Using bulleted lists and images to support your answers, address the following questions:
            - Data Preprocessing
                - What variable(s) are the target(s) for your model?
                - What variable(s) are the features for your model?
                - What variable(s) should be removed from the input data because they are neither targets nor features?
            - Compiling, Training, and Evaluating the Model
                - How many neurons, layers, and activation functions did you select for your neural network model, and why?
                - Were you able to achieve the target model performance?
                - What steps did you take in your attempts to increase model performance?
        3. <b>Summary</b>: Summarise the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.


# Neural Network Model Performance Report

## Overview of the Analysis
-------

To achieve Alphabet Soup's goal in acquiring a tool to select applicants for funding, a Neural Network Model (Tensorflow / Keras) was built & trained using their Charity CSV Dataset.
Built as a binary classifier, this analysis centres around evaluating its performance of accurately predicting whether applicants will be successful if funded by Alphabet Soup.

The 'Charity' Dataset consists of 34299 datapoints with the following features:

- `EIN` — Identification Column
- `NAME` — Identification Column
- `APPLICATION_TYPE` — Alphabet Soup application type
- `AFFILIATION` — Affiliated sector of industry
- `CLASSIFICATION` — Government organisation classification
- `USE_CASE` — Use case for funding
- `ORGANIZATION` — Organisation type
- `STATUS` — Active status
- `INCOME_AMT` — Income classification
- `SPECIAL_CONSIDERATIONS` — Special considerations for application
- `ASK_AMT` — Funding amount requested
- `IS_SUCCESSFUL` — Was the money used effectively (0 is 'No', 1 is 'Yes')

## Results
-------

### Data Preprocessing
-------

The goal is to have the binary classifier predict whether applicants will be **successful** if funded by Alphabet Soup. For the Neural Network Model, this means:

- Target Variable = `IS_SUCCESSFUL`
- all of the following variables are features for the model:
    - `APPLICATION_TYPE`
    - `AFFILIATION`
    - `CLASSIFICATION`
    - `USE_CASE`
    - `ORGANIZATION`
    - `STATUS`
    - `INCOME_AMT`
    - `SPECIAL_CONSIDERATIONS`
    - `ASK_AMT`

In addition, the following were performed as part of the pre-processing:

- `EIN` / `NAME` - Columns were dropped (no clear value as features or targets for the model)
- `APPLICATION_TYPE` - Unique values with a count of less than 500 were binned together as 'Other' (reduce paramaters for the model)
- `CLASSIFICATION` - Unique values with a count of less than 1883 were binned together as 'Other' (reduce paramaters for the model)

### Compiling, Training, and Evaluating the Model
-------

For this exercise, three attempts were made in an effort to optimise the model to achieve a target predictive accuracy higher than 75%.

### Attempt #1

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (relu)               (None, 250)               11000     
                                                                 
 dense_1 (relu)             (None, 150)               37650     
                                                                 
 dropout (Dropout)          (None, 150)                 0         
                                                                 
 dense_2 (relu)             (None, 30)                4530      
                                                                 
 dense_3 (relu)             (None, 30)                 930       
                                                                 
 dense_4 (sigmoid)          (None, 1)                  31        

```

```
268/268 - 1s - loss: 0.5792 - accuracy: 0.7234 - 628ms/epoch - 2ms/step
Loss: 0.5791993737220764, Accuracy: 0.7233819365501404

```

- For this first attempt:
    - First two layers have the largest number of neurons (given the large dataset dealt with)
    - Dropout layer was added after the 2nd layer (reduce overfitting)
    - Additional hidden layers were added (account for the complexity of the data)

- Model Accuracy = 72.34% (**TARGET PERFORMANCE NOT ACHIEVED**)


### Attempt #2

```
{'activation': 'tanh',
 'first_units': 100,
 'num_layers': 2,
 'units_0': 100,
 'units_1': 25,
 'units_2': 150,
 'tuner/epochs': 12,
 'tuner/initial_epoch': 4,
 'tuner/bracket': 4,
 'tuner/round': 2,
 'tuner/trial_id': '0103'}

 ```

 ```
268/268 - 1s - loss: 0.5547 - accuracy: 0.7286 - 627ms/epoch - 2ms/step
Loss: 0.5546512007713318, Accuracy: 0.7286297082901001

```

- For this second attempt:
    - In the Features dataset (`X`), the `SPECIAL_CONSIDERATIONS_N` column was dropped
        - The `SPECIAL_CONSIDERATIONS_Y` is more than sufficient on its own (1 is 'Yes', 0 is 'No')
    - Through the Hyperband Tuner instance (`hb_tuner`), automation was used to find the best hyperparameters for the model.
    - From the tuner with limits applied, the best model was found with the following hyperparameters:
        - Input Layer = 100 Neurons (`tanh`)
        - Hidden Layer #1 = 25 Neurons (`tanh`)
        - Hidden Layer #2 = 150 Neurons (`tanh`)
        - Epochs = 12

- Model Accuracy = 72.86% (**TARGET PERFORMANCE NOT ACHIEVED**)


### Attempt #3

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_5 (relu)             (None, 250)                10250     
                                                                 
 dense_6 (relu)             (None, 150)                37650     
                                                                 
 dropout_1 (Dropout)        (None, 150)                  0         
                                                                 
 dense_7 (relu)             (None, 50)                  7550      
                                                                 
 dense_8 (relu)             (None, 50)                  2550      
                                                                 
 dense_9 (sigmoid)          (None, 1)                    51        
                                                                 
=================================================================

```

```
268/268 - 1s - loss: 0.5688 - accuracy: 0.7257 - 506ms/epoch - 2ms/step
Loss: 0.5688203573226929, Accuracy: 0.7257142663002014

```


- In this final attempt:
    - Optimisation process was performed manually (excessive time required through automation)
    - `INCOME_AMT` - Unique values categorically marked as 5M or more were changed to 'Over 5M' (reduce paramaters for the model)
    - In the Features dataset (`X`), the `SPECIAL_CONSIDERATIONS_N` column was dropped
        - The `SPECIAL_CONSIDERATIONS_Y` is more than sufficient on its own (1 is 'Yes', 0 is 'No')
    - Count of neurons for the last two hidden layers were slightly increased.

- Model Accuracy = 72.57% (**TARGET PERFORMANCE NOT ACHIEVED**)


## Summary
---

Out of all three attempts, the second iteration (using the Hyperband Tuner instance) ultimately produced the best accuracy result (**72.86%**) but not by a large margin. Despite this, the models from across my attempts have failed to meet the desired performance of greater than 75% accuracy.

Since the Charity Dataset is made up of 34299 datapoints with a large number of dimensions, it is fair to suggest the number of neurons and layers for the binary classifier should increase given the complexity of the data. However, doing so will risk the model overfitting.

Finding the best parameters through automation will also be extremely resource intensive both in time and computation, and that applies even for running a model with additional layers and neuron count. As the end goal is to build a reliable binary classifier, different algorithms besides neural networking can be explored to achieve this.

With some features from the Charity dataset not having clear linear relationships with the target variable (`IS_SUCCESSFUL`), I would recommend building a Random Forest model as it is known to effectively handle non-linear relationships and less likely to overfit.