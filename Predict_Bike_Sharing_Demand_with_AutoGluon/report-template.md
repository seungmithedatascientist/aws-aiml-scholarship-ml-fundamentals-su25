# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Seungmi Kim (kimsm6397@gmail.com)

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
The raw predictions had negative values. Since negative values are not valid for bike rental counts, it was required to post-process the predictions by setting any values less than zero to zero. 

### What was the top ranked model that performed?
The top-ranked model in the initial training was WeightedEnsemble_L3, with the score value of -53.157140.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
For the exploratory data analysis, I first made the histogram of the distribution of each feature. It was found that bike usage varied by time of day. Therefore, a new feature `hour` was created by extracting the hour from the `datetime` column. Also, some categorical features like `season` and `weather` were properly encoded using the `category` dtype. 

### How much better did your model preform after adding additional features and why do you think that is?
The Kaggle score improved from 1.80123 to 0.60461. This is likely because the `hour` feature captured temporal patterns related to work commute and leisure times. 

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Hyperparameter tuning led to a modest improvement in Kaggle score from 0.60461 to 0.60006. Although this drop is not very significant, this fine-tuning helped find better-performing model configurations, also suggesting further systematic hyperparameter tuning.

### If you were given more time with this dataset, where do you think you would spend more time?
If I have more time, I would explore additional time-based features such as day of the week, weekdays or weekends, and weather-hour interactions. 

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|None|None|None|1.80123|
|add_features|None|None|None|0.60461|
|hpo|num_trials=10|scheduler=local|searcher=random|0.60006|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
This project has steadily improved model performance using different methods. Including time-based features (`hour`) had the most significant impact, which suggests real-world pattern in bike usage. It is also intuitive that bicycle usage patterns for work commmute and leisures significnantly vary for different times of the day. Hyperparameter tuning provided smaller but meaningful gains. AutoGluon made it simple to iterate the process quickly while utilizing ensemble learning method.

