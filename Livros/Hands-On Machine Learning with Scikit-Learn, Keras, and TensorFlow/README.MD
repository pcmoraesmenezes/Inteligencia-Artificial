# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

## Chapter 1 - Machine Learning Landscape

Machine Learning is the process of a machine learning from data. This process don't happen by itself, so if I download a page in web my machine will have the data, but not the knowledge. The machine will need to learn from the data to extract knowledge from it. The process of learning from data is called training.

The process of a machine learning training can be describe as:

1. Get the data

2. Train a model

3. Evaluate the model

   **If** the model is not good, **then** we need to tune the model or get more data.
   
   **Else** continue with the current model.


The great advantage of an machine learning process is that humans can learn from the machine learning process. The humans can inspect the solution and understand how the machine is solving the problem. This is called **data mining**.

There are many types of machine learning algorithms, but they can be grouped in eight main categories:

1. Supervised Learning: The training data is labeled. The algorithm tries to learn the relationship between the features and the labels.

2. Unsupervised Learning: The training data is not labeled. The algorithm tries to learn the relationship between the features.

3. Semisupervised Learning: The training data is partially labeled.

4. Reinforcement Learning: The algorithm learns by interacting with the environment. It receives rewards and penalties for the actions it takes.

5. Batch Learning: The model is trained with all the data at once.

6. Online Learning: The model is trained with data instances one at a time.

7. Instance-Based Learning: The model learns the training data by heart and generalizes to new data by comparing it to the training data.

8. Model-Based Learning: The model learns the training data and generalizes to new data by using a model.


The most common supervised learning tasks are:

1. Classification: The model tries to predict a class label.

2. Regression: The model tries to predict a continuous value.


### Exercises

1. How would you define Machine Learning?

R: Machine Learning is the process of a machine learning from data.

2. Can you name four types of problems where it shines?

R: Machine Learning shines in problems where the solution is too complex for traditional approaches, where the solution changes over time, where the solution requires a lot of fine-tuning, and where the solution requires a large amount of data.

3. What is a labeled training set?

R: A labeled training set is a training set that contains the desired solution for each instance.

4. What are the two most common supervised tasks?

R: The two most common supervised tasks are classification and regression.

5. Can you name four common unsupervised tasks?

R: The four most common unsupervised tasks are clustering, visualization, dimensionality reduction, and association rule learning.

6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?

R: Reinforcement Learning.

7. What type of algorithm would you use to segment your customers into multiple groups?

R: Clustering.

8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?

R: Supervised learning.

9. What is an online learning system?

R: An online learning system is a system that learns incrementally, as data streams in.

10. What is out-of-core learning?

R: Out-of-core learning is the process of training a model on a large dataset that cannot fit in a computer's main memory.

11. What type of learning algorithm relies on a similarity measure to make predictions?

R: Instance-based learning.

12. What is the difference between a model parameter and a learning algorithm's hyperparameter?

R: A model parameter is a parameter that the model learns from the training data, while a learning algorithm's hyperparameter is a parameter that the learning algorithm uses to control the learning process.

13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?

R: Model-based learning algorithms search for an optimal value of the model's parameters. The most common strategy they use to succeed is to minimize a cost function. They make predictions by using the model's parameters.

14. Can you name four of the main challenges in Machine Learning?

R: The four main challenges in Machine Learning are insufficient quantity of training data, nonrepresentative training data, poor-quality data, and irrelevant features.

15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?

R: The model is overfitting the training data. Three possible solutions are to simplify the model, to gather more training data, and to reduce the noise in the training data.

16. What is a test set and why would you want to use it?

R: A test set is a set of instances that are used to evaluate the model's performance. You would want to use a test set to estimate the model's performance on new instances.

17. What is the purpose of a validation set?

R: The purpose of a validation set is to tune the model's hyperparameters.

18. What is a train-dev set and when would you use it?

R: A train-dev set is a set of instances that are used to evaluate the model's performance on the training data. You would use a train-dev set when you suspect that the model is overfitting the training data.

19. What can go wrong if you tune hyperparameters using the test set?

R: If you tune hyperparameters using the test set, you risk overfitting the test set.

---

## Chapter 2 - End-to-End Machine Learning Project

This chapter is about to set a project from scratch and go through all the steps of a machine learning project. In this chapter we're going to work with the California Housing Prices dataset from the StatLib.

### Frame the Problem

In a ML/DL project, we can set some steps to follow to make the project easier to manage.

We can set the following steps:

1. Define the objective of the project.
2. Understanding the Problem.
3. Identify the type of problem.
4. How to get the data.
5. Choose a Evaluate Metrics.
6. Exploratory Data Analysis.
7. Data Preprocessing.
8. Selecting a Model.
9. Training the Model.
10. Fine-Tuning the Model.
11. Presenting the Solution.

These steps are not mandatory, but they can help guide the project.

#### Define the Objective of the Project

In this step, you must question what is the goal of the project? What is the business objective? How will the solution be used? What are the current solutions/workarounds? How should the problem be framed? How should the performance be measured? Are there any constraints (business, legal, etc.)?

- In this case, we know that the stakeholders want to predict the median housing price in any district in California.
- Additionally, it's important to note the **real-world application**: real estate agents or investors will use the results to optimize their investments, so the model must fit within a broader **pipeline**, where data will flow from various systems into a model, be processed, and then feed into a database for decision-making.

#### Understanding the Problem

In this step, you must question what is the current solution? How is the problem solved today? What assumptions are made by the current solution? What are the limitations?

- The current solution involves hiring experts who manually gather and process data, investing a lot of time and money.
- **Understanding the limitations**: Manual predictions can be slow, prone to bias, and less scalable, which is why automation is desirable.

#### Identify the Type of Problem

In this step, you must question what type of problem you're dealing with. Is it supervised, unsupervised, or reinforcement learning? Is it classification, regression, or something else? 

- Stakeholders want to predict a **value**, making this a **regression problem**.
- **Optional Consideration**: While this project is framed as a regression, it could be interesting to consider whether the problem might benefit from categorizing investment decisions (e.g., high-risk vs. low-risk areas), which would turn it into a classification problem.

#### How to Get the Data

In this step, you must question how to get the data, how much is needed, what kind of data is required, and whether it's in the correct format. 

- **In this case**, the data is available in the StatLib repository and is already clean and in CSV format. It’s not large, so it fits comfortably in memory.
- **Considerations for scalability**: Even though the dataset is currently manageable, consider whether new data sources or updates will be required over time. How will this data pipeline handle future growth?

#### Choose and Evaluate Metrics

In this step, you must decide how to evaluate the model. What are the performance measures and the cost function?

- **RMSE** and **MSE** are great choices for a regression problem. However, consider also using **Mean Absolute Error (MAE)**, especially if the dataset contains outliers.
  
    - **MAE formula**: 
    
    $MAE(X, h) = \frac{1}{m} \sum_{i=1}^{m} |h(x^{(i)}) - y^{(i)}|$
    
  
    - This metric is less sensitive to outliers than RMSE or MSE and can provide a more interpretable error in the context of business decisions (i.e., "On average, the model is off by $X").

#### Exploratory Data Analysis (EDA)

In this step, you need to explore the structure of the data, its distributions, correlations, and outliers. 

- **Visualization is key**: Incorporate graphs like **histograms, scatter plots**, and **correlation matrices** to gain insights into feature relationships and distributions. This helps in identifying patterns that can be exploited by the model.
- **Outliers and Correlations**: Pay special attention to outliers and highly correlated features, as they can distort the performance of your regression model. Outliers might skew metrics like RMSE, so strategies to handle them (e.g., capping or transformation) should be considered.
- **Distribution Checks**: If features have highly skewed distributions, you may need to apply transformations (e.g., log-transform) to stabilize variance.

#### Data Preprocessing

In this step, handle missing values, outliers, categorical features, scaling, and feature engineering. 

- **Feature Engineering**: Besides handling missing values and scaling, consider creating new features that capture interactions between existing ones. For example, combining latitude and longitude into a feature that better represents a district’s **location** might enhance predictions.
- **Scaling**: Depending on your model choice (e.g., linear regression, neural networks), feature scaling is crucial. Use **Min-Max Scaling** or **Standardization** based on your model’s sensitivity to different ranges of features.

#### Selecting a Model

In this step, choose the best model to fit the problem. 

- Start with simple models like **linear regression** to establish a baseline.
- As you progress, experiment with more complex models like **decision trees, random forests**, or **neural networks**, depending on the complexity of your dataset.
- **Cross-validation**: Use **k-fold cross-validation** to ensure the model generalizes well to unseen data and isn’t overfitting.

#### Training the Model

In this step, focus on training the model using the preprocessed data.

- Ensure you implement **validation sets** to prevent overfitting. After training, evaluate the model on the validation set and check whether performance deteriorates (indicative of overfitting).

#### Fine-Tuning the Model

In this step, improve model performance by adjusting hyperparameters. 

- Use techniques like **Grid Search** or **Random Search** to find optimal hyperparameters.
- **Analyze Learning Curves**: Plot training and validation errors over time to see if your model is underfitting or overfitting. This can guide you to adjust your model's complexity or the amount of data needed.

#### Presenting the Solution

In this step, focus on how to deliver the results to stakeholders.

- **Dashboard or Visualization**: Consider delivering predictions through **visual dashboards** or reports that make it easy for real estate investors to interpret the results. Tools like Power BI, Tableau, or even web dashboards might be appropriate.
- **Post-deployment monitoring**: Consider how the model’s performance will be tracked over time. Create mechanisms to update the model with new data or provide alerts if its predictions become unreliable.

### Understanding the Pipeline

A pipeline is a sequence of data processing components. Each component is called a **data transformation**. Components typically run **asynchronously**. Each component pulls in a large amount of data, processes it, and spits out the result in another data store. The next component pulls in that data and processes it further.

In a real-world scenario, the ML/DL model is just one part of a broader pipeline. Data will flow from various systems into the model, be processed, and then feed into a database for decision-making.

The flow will look something like this:

1. **Data Ingestion**: Data is collected from various sources (e.g., databases, APIs, files).

2. **Data Preprocessing**: Data is cleaned, transformed, and prepared for the model.

3. **Model Training**: The model is trained on the preprocessed data.

4. **Model Evaluation**: The model’s performance is evaluated on a validation set.

5. **Model Deployment**: The model is deployed to a production environment.

6. **Monitoring and Maintenance**: The model’s performance is monitored, and it’s updated as needed.

