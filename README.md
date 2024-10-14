# Calories Burned Prediction

The **Calories Burned Prediction** project aims to predict the calories burned by individuals based on their exercise data. By utilizing a dataset containing user-specific information such as gender, age, height, weight, duration of exercise, heart rate, and body temperature, we can effectively build a model to estimate the calories burned during physical activities.

## Objectives

- **Data Exploration**: Analyze the datasets to understand the relationships between different features and calories burned.
- **Data Preprocessing**: Clean and prepare the data for modeling.
- **Modeling**: Implement a regression model using XGBoost to predict calories burned.
- **Evaluation**: Assess the performance of the model using appropriate metrics and visualize the results.

## Dataset

### 1. Exercise Data (`exercise.csv`)

| User_ID  | Gender | Age | Height | Weight | Duration | Heart_Rate | Body_Temp |
|----------|--------|-----|--------|--------|----------|------------|-----------|
| 14733363 | male   | 68  | 190.0  | 94.0   | 29.0     | 105.0      | 40.8      |
| 14861698 | female | 20  | 166.0  | 60.0   | 14.0     | 94.0       | 40.3      |
| ...      | ...    | ... | ...    | ...    | ...      | ...        | ...       |

### 2. Calories Data (`calories.csv`)

| User_ID  | Calories |
|----------|----------|
| 14733363 | 231.0    |
| 14861698 | 66.0     |
| ...      | ...      |

## Procedure

1. **Data Loading**:
   - Import the necessary libraries and load the datasets.

    ```python
    import numpy as np
    import pandas as pd
    ```

2. **Data Exploration**:
   - Use methods like `head()`, `describe()`, and visualizations to understand the datasets.

    ```python
    calories_data.describe()
    sns.countplot(calories_data['Gender'])
    sns.distplot(calories_data['Age'])
    ```

3. **Data Preprocessing**:
   - Handle missing values, convert categorical variables to numerical formats, and concatenate the datasets.

    ```python
    calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)
    ```

4. **Feature Selection**:
   - Define the feature matrix (X) and target vector (Y).

    ```python
    X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
    Y = calories_data['Calories']
    ```

5. **Train-Test Split**:
   - Split the dataset into training and testing sets.

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    ```

6. **Model Training**:
   - Train the XGBoost Regressor on the training data.

    ```python
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    ```

7. **Prediction**:
   - Predict the calories burned for the test data.

    ```python
    test_data_prediction = model.predict(X_test)
    ```

8. **Evaluation**:
   - Calculate the Mean Absolute Error (MAE) and visualize the results using scatter plots and residual plots.

    ```python
    mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
    ```

9. **Results Visualization**:
   - Create plots to compare actual vs predicted values and visualize residuals.

    ```python
    plt.scatter(Y_test, test_data_prediction)
    ```

## Conclusion

This project demonstrates how machine learning can be applied to predict calories burned during physical activities, providing insights that can assist individuals in their fitness journeys. Future work may involve enhancing the model with additional features or employing different machine learning algorithms to improve accuracy.
