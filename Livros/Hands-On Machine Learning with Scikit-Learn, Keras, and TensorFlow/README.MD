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


## Chapter 3 - Classification

In the previous chapter, we discussed regression problems. Now, let's focus on **classification** problems.

Classification is the process of predicting the class of a given data point. The classes can be **binary** (e.g., spam/not spam) or **multiclass** (e.g., handwritten digit recognition). We will use the well-known **MNIST** dataset, which contains 70,000 small images of handwritten digits collected from students and U.S. Census Bureau employees. Each image is labeled with the digit it represents.

### Binary Classification

A **Binary Classification** task has two possible outcomes. A common example is spam detection (spam/not spam). Another typical binary classification task is identifying the presence or absence of a condition — for example, breast cancer detection (has cancer/does not have cancer).

For this example, we will create a binary classifier to detect whether an image contains the digit **5** or not.

### Performance Measures

Evaluating a classifier is often more challenging than evaluating a regressor. Several performance metrics are available, and it is essential to understand them to choose the right one for your problem.

Our goal in a machine learning project is to ensure that the model generalizes well to **new data**. Therefore, a good performance measure is critical. Techniques like **cross-validation** help ensure that the model performs well even when trained on different subsets of data.

#### Confusion Matrix

The **Confusion Matrix** is a helpful tool for understanding the performance of a classifier in terms of correct and incorrect predictions:

| True Negative | False Positive |
|---------------|----------------|
| False Negative| True Positive  |

Each cell in the matrix represents a count of predictions made by the model, comparing the actual label with the predicted label. This matrix is particularly useful for identifying which types of errors the model is making, such as reducing **False Negatives** in medical diagnosis problems.

#### Precision, Recall, and F1 Score

- **Precision**: Measures how accurate the positive predictions are.
  - Formula: $Precision = \frac{TP}{TP + FP}$
  
- **Recall**: Measures how well the model captures all actual positive instances.
  - Formula: $Recall = \frac{TP}{TP + FN}$
  
- **F1 Score**: The harmonic mean of Precision and Recall. It is useful when we want to balance both.
  - Formula: $F1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$

Each metric has its own importance depending on the context. For example, in spam detection, Precision is more critical, while in medical diagnoses, Recall is often prioritized.

#### Precision/Recall Tradeoff

The **Precision/Recall Tradeoff** occurs because improving one metric often worsens the other. For instance, increasing Precision can lead to a drop in Recall. In many cases, the goal is to find a balance that maximizes the F1 Score, which considers both Precision and Recall.

#### ROC Curve and AUC

The **ROC Curve** (Receiver Operating Characteristic) plots the True Positive Rate (Recall) against the False Positive Rate. One important metric derived from the ROC curve is the **AUC** (Area Under the Curve), which provides a summary of the model's overall performance. A higher AUC (closer to 1.0) indicates better performance.


### Multiclass Classification

In a **Multiclass Classification** task, the model must classify instances into three or more classes. For example, classifying news articles into categories like sports, politics, or technology.

Some algorithms like **Random Forest** and **Naive Bayes** can handle multiclass classification directly. Others like **Support Vector Machines (SVM)** and **Linear Classifiers** are strictly binary classifiers. However, there are strategies to perform multiclass classification using binary classifiers, such as **One-versus-All (OvA)** or **One-versus-One (OvO)**.

#### OvA and OvO Strategies

- **OvA**: Train a binary classifier for each class. When classifying a new instance, select the class with the highest score among all classifiers. This strategy is efficient for large datasets.

- **OvO**: Train a binary classifier for each pair of classes. If there are N classes, you need N * (N - 1) / 2 classifiers. When classifying a new instance, the class that wins the most duels is the predicted class. This strategy is more efficient for smaller datasets.

It is possible to select the strategy based on the algorithm's scalability and the dataset's size. For example, **SVM** scales poorly with the size of the training set, so OvO is preferred for SVM.

To do this use the `OneVsOneClassifier` or `OneVsRestClassifier` classes from Scikit-Learn.

### Error Analysis

After training a model, it is essential to analyze its errors to understand where it is failing and why. This process can provide insights into how to improve the model.

One common technique is to analyze the **confusion matrix** and identify which classes are often confused. This can help determine whether the model is making systematic errors, such as confusing similar classes.

Another approach is to visualize the model's errors. For example, plotting instances that the model misclassified can provide insights into why it failed. This can help identify patterns that the model is not capturing.

### Multilabel Classification

In a **Multilabel Classification** task, each instance can have multiple classes. For example, classifying images of people into categories like "happy," "smiling," or "wearing glasses."

To handle multilabel classification, you can use the `KNeighborsClassifier` or `RandomForestClassifier` from Scikit-Learn. These classifiers can output multiple binary labels for each instance.

### Multioutput Classification

In a **Multioutput Classification** task, each label can be multiclass. For example, removing noise from images can be seen as a multioutput classification task, where each pixel can have multiple values.

To handle multioutput classification, you can use the `KNeighborsClassifier` or `RandomForestClassifier` from Scikit-Learn. These classifiers can output multiple binary labels for each instance.

### Exercises

1. Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set. Hint: the KNeighborsClassifier works quite well for this task; you just need to find good hyperparameter values (try a grid search on the weights and n_neighbors hyperparameters).

2. Write a function that can shift an MNIST image in any direction (left, right, up, or down) by one pixel. Then, for each image in the training set, create four shifted copies (one per direction) and add them to the training set. Finally, train your best model on this expanded training set and measure its accuracy on the test set. You should observe that your model performs even better now! This technique of artificially growing the training set is called **data augmentation** or **training set expansion**.

3. Tackle the Titanic dataset. A great place to start is on Kaggle.

4. Build a spam classifier (a more challenging exercise).

## Chapter 4 - Training Models

This chapter was focused to explain some of the most common algorithms used in machine learning.

### Linear Regression

In the chapter two I've saw a little about linear regression, now I'm going to understand the math behind it.

Basic a model can be explained as a function that maps the input features to the predicted output, for example life satisfaction = f(GDP per capita). This type of function is linear.

A linear model makes a prediction by computing a weighted sum of the input features, plus a constant called the bias term (also called the intercept term). The equation is:

$\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$

Where:

- $\hat{y}$ is the predicted value.

- n is the number of features.

- $x_i$ is the i-th feature value.

- $\theta_j$ is the j-th model parameter (including the bias term $\theta_0$ and the feature weights $\theta_1, \theta_2, ..., \theta_n$).

As dealling with machine learning and deep learning approaches is common to use vectors and matrices to represent the equations, because it's easier and faster to compute. In the case of a linear regression model, we can represent the equation as:

$\hat{y} = h_{\theta}(x) = \theta^T \cdot x$

Where:

- $\theta$ is the model's parameter vector, containing the bias term $\theta_0$ and the feature weights $\theta_1, \theta_2, ..., \theta_n$.

- $x$ is the instance's feature vector, containing $x_0$ to $x_n$, with $x_0$ always equal to 1.

- $\theta^T$ is the transpose of $\theta$.

- $h_{\theta}$ is the hypothesis function, using the model parameters $\theta$.

The process of training a model is to find the parameters that minimize the cost function. In the case of a linear regression model, the cost function is the Mean Squared Error (MSE) function:

$MSE(X, h_{\theta}) = \frac{1}{m} \sum_{i=1}^{m} (\theta^T \cdot x^{(i)} - y^{(i)})^2$

Where:

- $m$ is the number of instances in the dataset.

- $x^{(i)}$ is the i-th instance's feature vector.

- $y^{(i)}$ is the i-th instance's target value.

- $\theta^T \cdot x^{(i)}$ is the prediction for the i-th instance using the model parameters $\theta$.

The goal is to find the value of $\theta$ that minimizes the MSE function. 

We can get the value of $\theta$ by using the Normal Equation:

$\hat{\theta} = (X^T \cdot X)^{-1} \cdot X^T \cdot y$

Where:

- $\hat{\theta}$ is the value of $\theta that minimizes the cost function.

- y is the vector of target values containing $y^{(1)}$ to $y^{(m)}$.

#### Computational Complexity

Using the Normal Equation to compute we need to compute a the inverse of a matrix multiplication $(X^T \cdot X)$. Our matrix $X$ has a shape of $(m, n)$, where $m$ is the number of instances and $n$ is the number of features. Our matrix $X^T$ has a shape of $(n, m)$. So the matrix multiplication $X^T \cdot X$ has a shape of $(n, n)$. By doing the inverse of this matrix we have a new matrix with the same shape $(n, n)$. The computational complexity of inverting a matrix is about $O(n^{2.4})$ to  $O(n^3)$.

The Normal Equation gets very slow when the number of features grows large. However, the equation is linear with regards to the number of instances in the training set, so it handles large training sets efficiently, provided they can fit in memory. 

The good new is that once you have trained your Linear Regression model, predictions are very fast: the computational complexity is linear with regards to both the number of instances you want to make predictions on and the number of features.

See the implementation of a linear regression in this [file](https://github.com/pcmoraesmenezes/Inteligencia-Artificial/blob/main/Livros/Hands-On%20Machine%20Learning%20with%20Scikit-Learn%2C%20Keras%2C%20and%20TensorFlow/codes/training_models/linear_regression.py)

### Gradient Descent

The gradient descent method is a optimization algorithm that finds the minimum of a function. The idea is to tweak the parameters iteratively to minimize a cost function.

You can think of this as rolling a ball down a hill. The ball will eventually reach the lowest point. The parameters are updated in the opposite direction of the gradient of the cost function.

#### Epochs and Learning Rate

Those two parameters are crucial for this type of optimization algorithm.

The first one, **Epochs**, is the number of times the algorithm goes through the dataset. The second one, **Learning Rate**, is the size of the steps the algorithm takes to find the minimum of the function.

A high number of epochs leads to more time training the model, but it can lead to a better model. However, a high number of epochs does not necessary implies a better model. 

The second one, the learning rate, is a hyperparameter that can be tuned. A high learning rate can make the algorithm to diverge, while a low learning rate can make the algorithm to take too long to converge.

Those two parameters are crucial to be tuned to get a good model, and they complete each other. A high learning rate can be used with a low number of epochs, while a low learning rate can be used with a high number of epochs.

#### Early Stopping

Since we've discussed about the number of epochs and the learning ratio, we can talk about the early stopping

The early stopping is a technique to stop the training of the model when the model is not improving anymore. We set a **PATIENCE** and a **TOLERANCE**. The patience is the number of epochs that the model can be stopped without improving, while the tolerance is the minimum improvement that the model must have to not be stopped.

Those two parameters can be combinned with the learning rate and the number of epochs to get a good model.

For example, we can set a high number of epochs and a low learning rate value, and set a arbitrary value for the patience and tolerance. If the model reachs the patience we can update dinaamically the learning rate to see if the model improves. If the model does not improve, we can stop the training.

---

There're three types of a cost function:

1. No convex function: The function has many local minimums, so the algorithm can get stuck in a local minimum.

2. Convex function: The function has only one minimum, so the algorithm can find the global minimum.

3. Convex function with a high learning rate: The algorithm can diverge.

![Convex Function](/images/gradient_descent1.jpg)

As we can see the image above, the cost function is like a bowl. The algorithm starts at the top of the bowl and goes down to the bottom. 

![Convex Function with a high learning rate](/images/gradient_descent_2.png)

The second image shows a cost function that is more like a irregular terrain. The algorithm can get stuck in a local minimum and not diverge to the global minimum.

By using a gradient descent algorithm we must ensure that all features have a similar scale. If the features have different scales, the algorithm will take a long time to converge.

#### Batch Gradient Descent

The Batch Gradient Descent algorithm computes the gradients of the cost function with regards to the model parameters $\theta$ for the entire training set $X$ at each step. The algorithm is slow when the training set is large. The calculation of the gradients is based on partial derivatives of the cost function in relation to the model parameters.

The Batch Gradient Descent algorithm is given by the equation:

$\frac{\partial}{\partial\theta_{j}} MSE(\theta) = \frac{2}{m} \sum_{i=1}^{m} (\theta^T \cdot x^{(i)} - y^{(i)})x^{(i)}_{j}$

Where:

- $m$ is the number of instances in the dataset.

- $x^{(i)}$ is the i-th instance's feature vector.

- $y^{(i)}$ is the i-th instance's target value.

- $\theta^T \cdot x^{(i)}$ is the prediction for the i-th instance using the model parameters $\theta$.

- $x^{(i)}_{j}$ is the j-th feature value of the i-th instance.

The algorithm is given by the equation:

We can use a vectorized form of the equation to compute the gradients of the cost function:

$\nabla_{\theta} MSE(\theta) = \frac{2}{m} X^T \cdot (X \cdot \theta - y)$

Where:

- $\nabla_{\theta} MSE(\theta)$ is the gradient vector of the cost function.

- $X$ is the matrix of feature values.

- $y$ is the vector of target values.

By adding a learning rate $\eta$ to the equation we have the equation:

$\theta^{(next step)} = \theta - \eta \nabla_{\theta} MSE(\theta)$

Where:

- $\theta^{(next step)}$ is the next step of the model parameters.

- $\eta$ is the learning rate.

- $\nabla_{\theta} MSE(\theta)$ is the gradient vector of the cost function.


#### Stochastic Gradient Descent

Batch Gradient Descent, while effective, has a significant drawback: it uses the entire dataset to compute gradients. This can make the algorithm very slow, especially for large datasets. Stochastic Gradient Descent (SGD) offers an efficient alternative by computing the gradients of the cost function using randomly selected instances of the dataset. This makes it well-suited for large datasets.

However, SGD has its own limitations. Unlike Batch Gradient Descent, which progresses smoothly towards the minimum, SGD introduces randomness due to its reliance on individual data points. This randomness can cause the algorithm to fluctuate around the minimum instead of converging exactly. Additionally, it can sometimes get stuck in local minima.

That said, this same randomness can be an advantage in more complex scenarios. For cost functions with irregular terrains, SGD is less likely to settle into a local minimum and has a better chance of escaping to find the global minimum.

To improve SGD's performance, a learning schedule can be employed. A learning schedule is a strategy that gradually reduces the learning rate as the algorithm progresses towards the minimum. By starting with a higher learning rate to make large strides and gradually decreasing it to fine-tune the solution, the algorithm can balance exploration and convergence. This approach, inspired by the Simulated Annealing optimization technique, enhances SGD's ability to find the global minimum.

#### Mini-batch Gradient Descent

Mini-batch Gradient Descent is a compromise between Batch Gradient Descent and Stochastic Gradient Descent. Instead of computing the gradients based on the entire dataset (Batch GD) or a single instance (SGD), Mini-batch GD computes the gradients on small random sets of instances called mini-batches.

Mini-batch GD offers the best of both worlds: it is faster than Batch GD and more stable than SGD. The algorithm's progress is less erratic than SGD, as the mini-batches provide a smoother convergence towards the minimum. Additionally, Mini-batch GD can leverage hardware optimization for matrix operations, making it faster than pure Python implementations.

However, Mini-batch GD is more likely to get stuck in local minima than SGD, as the algorithm's path is less erratic. To mitigate this, a learning schedule can be used to gradually reduce the learning rate as the algorithm approaches the minimum.

---

All of the algorithms have their own advantages and disadvantages. The Batch Gradient Descent is slow, but it can find the global minimum. The Stochastic Gradient Descent is fast, but it can get stuck in a local minimum. The Mini-batch Gradient Descent is a compromise between the two algorithms.

It's possible to reach the implementation of the three algorithms in this [file](/codes/training_models/gradient_descent_models.py)

### Polynomial Regression

Some kind of data is not linear. When we're facing with this kind of problem we can use a polynomial regression. The idea is to add powers of each feature as new features, then train a linear model on this extended set of features.

Facing a ML problem requires sometimes a try-error approach. Some problems is easily solved by linear models, while others require some suitable approach. For Polynomial Regression our focus is on the parameter `degree` of the model. The higher the degree, the more complex the model will be. Because of that, use high levels of degree can cause overfitting to the model. Otherwise use low levels when facing complex problems can cause underfitting.

This can be called as `Learning Curves`. The learning curves are plots of the model's performance on the training set and the validation set as a function of the training set size (or the training iteration). This can help to understand if the model is overfitting or underfitting.

![Learning Curves](/images/high_regression_degree.png)

![Learning Curves](/images/learning_curve.png)

One cool caracteristic of the polynomial regression is that it can find the patterns in the data (a linear model can't). This happens because the polynomial regression also uses the features combinations to find the patterns till the degree of the model. For example, a `degree=3` the model will use the features $a$, $b$, $a^2$, $b^2$, $ab$, $a^3$, $b^3$, $a^2b$, $ab^2$.

You can find a implementation of a polynomial regression in this [file](/codes/training_models/polynomial_regression.py)

--- 

#### Tradeoff between Bias and Variance

The **Bias/Variance Tradeoff** is a fundamental concept in machine learning. It describes the tradeoff between a model's ability to minimize bias (error from erroneous assumptions) and variance (sensitivity to small fluctuations in the training data).

- **Bias**: Bias is the error from erroneous assumptions in the learning algorithm. High bias can cause underfitting, where the model is too simple to capture the underlying structure of the data.

- **Variance**: Variance is the error from sensitivity to small fluctuations in the training data. High variance can cause overfitting, where the model is too complex and learns the noise in the training data.

- **Irreducible Error**: Irreducible error is the error from the noisiness of the data itself. It cannot be reduced by the model.

The goal is to find the right balance between bias and variance to minimize the model's generalization error. This is achieved by tuning the model's complexity, using techniques like cross-validation, regularization, and learning curves.

#### Regularized Linear Models

Regularization is a technique used to prevent overfitting by constraining a model's complexity. Regularized linear models add a regularization term to the cost function, which penalizes large model parameters. This encourages the model to fit the data well while keeping the model weights small.

##### Ridge Regression

**Ridge Regression** (also called Tikhonov regularization) adds a regularization term to the cost function equal to the L2 norm of the weight vector. This forces the model to fit the data well while keeping the weights as small as possible.

The regularization is used only during the cost function calculation, not during the prediction. The regularization term is added to the cost function only during the training phase.

The Ridge Regression cost function is given by:

$J(\theta) = MSE(\theta) + \alpha \frac{1}{2} \sum_{i=1}^{n} \theta_i^2$

Where:

- $\alpha$ is the regularization hyperparameter.

- $\theta_i$ is the i-th model parameter.

If $\alpha$ is set to 0, Ridge Regression is just Linear Regression. If $\alpha$ is set to a very high value, all weights end up very close to zero, and the result is a flat line going through the data's mean.

The Ridge Regression model is trained using the following equation:

$\hat{\theta} = (X^T \cdot X + \alpha A)^{-1} \cdot X^T \cdot y$

Where:

- $A$ is the identity matrix with a 0 in the top-left cell.

It's possible to use the Ridge regularization in gradient descent algorithms. The regularization term is added to the gradients of the cost function.

![Ridge Regression](/images/ridge_regression.png)

#### Lasso Regression

**Lasso Regression** (Least Absolute Shrinkage and Selection Operator) is another regularized version of Linear Regression. Lasso Regression adds a regularization term to the cost function equal to the L1 norm of the weight vector. This forces the model to fit the data well while keeping the weights as small as possible.

The Lasso Regression cost function is given by:

$J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} |\theta_i|$

One particularity of the Lasso Regression is that it tends to eliminate the weights of the least important features. This can be seen as an automatic feature selection technique.

By using a gradient descent algorithm, the regularization term is added to the gradients of the cost function. Plus a dynamic alpha value can be used to update the learning rate.

![Lasso Regression](/images/lasso_regression.png)

#### Elastic Net

**Elastic Net** is a middle ground between Ridge Regression and Lasso Regression. It combines both regularization terms, adding a mix ratio $r$ to the cost function.

The Elastic Net cost function is given by:

$J(\theta) = MSE(\theta) + r \alpha \sum_{i=1}^{n} |\theta_i| + \frac{1 - r}{2} \alpha \sum_{i=1}^{n} \theta_i^2$

![Elastic Net](/images/elastic_net.png)

### Logistic Regression

**Logistic Regression** is a classification algorithm used to estimate the probability that an instance belongs to a particular class. If the estimated probability is greater than 50%, the model predicts the instance belongs to that class (positive class, labeled as 1). Otherwise, it predicts the instance belongs to the negative class (labeled as 0).

Logistic Regression computes a weighted sum of the input features, plus a bias term, and outputs the logistic of the result. The logistic function is a sigmoid function that outputs a number between 0 and 1.

The probability that the model predicts an instance belongs to the positive class is given by:

$\hat{p} = h_{\theta}(x) = \sigma(\theta^T \cdot x)$

A logistic function is given by:

$\sigma(t) = \frac{1}{1 + e^{-t}}$

#### How to train a Logistic Regression model

The objective of training a Logistic Regression model is to set the model parameters $\theta$ so that the model estimates high probabilities for positive instances and low probabilities for negative instances. The cost function used in Logistic Regression is the **Log Loss** function.

The Log Loss function is given by:

$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)})]$

Where:

- $m$ is the number of instances in the dataset.

- $\hat{p}^{(i)}$ is the model's estimated probability that the i-th instance belongs to the positive class.

- $y^{(i)}$ is the target value of the i-th instance (1 if the instance belongs to the positive class, 0 otherwise).

The Log Loss function penalizes the model when it estimates a low probability for a positive instance or a high probability for a negative instance. The model parameters $\theta$ are trained to minimize the cost function.

The Log Loss function does not have a closed-form solution, so it cannot be computed directly. However, it is a convex function, so Gradient Descent (or any other optimization algorithm) is guaranteed to find the global minimum.

The gradients of the Log Loss function are given by:

$\frac{\partial}{\partial\theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\sigma(\theta^T \cdot x^{(i)}) - y^{(i)})x^{(i)}_{j}$

The model parameters $\theta$ are updated using the following equation:

$\theta^{(next step)} = \theta - \eta \nabla_{\theta} J(\theta)$

#### Decision Frontier

![Decision Frontier](/images/decision_frontier.png)

A decision boundary is a surface that separates instances of different classes. In the case of Logistic Regression, the decision boundary is linear. This means the model predicts a positive class for instances on one side of the boundary and a negative class for instances on the other side.

In this plot, the decision boundary is the line where the model's estimated probability is 50%. Instances on one side of the line are classified as positive, while instances on the other side are classified as negative.

### Softmax Regression

**Softmax Regression** (also called Multinomial Logistic Regression) is a generalization of Logistic Regression to support multiple classes. Instead of predicting just two classes (positive and negative), Softmax Regression can predict multiple classes.

This happens by computing a score for each class, then applying the softmax function to estimate the probability that an instance belongs to each class. The class with the highest probability is the one predicted by the model.

The score for class $k$ is given by:

$s_k(x) = x^T \cdot \theta^{(k)}$

After computing the score for each class, the Softmax function is applied to estimate the probability that the instance belongs to class $k$:

$\hat{p}_k = \sigma(s(x))_k = \frac{e^{s_k(x)}}{\sum_{j=1}^{K} e^{s_j(x)}}$

Where:

- $K$ is the number of classes.

- $s(x)$ is a vector containing the scores of each class for the instance $x$.

- $\sigma(s(x))_k$ is the estimated probability that the instance $x$ belongs to class $k$.

The Softmax Regression model predicts the class with the highest estimated probability:

$\hat{y} = \text{argmax}_k \sigma(s(x))_k$

The model is trained using the **Cross-Entropy** cost function, which penalizes the model when it estimates a low probability for the target class. The Cross-Entropy cost function is given by:

$J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{p}_k^{(i)})$

Where:

- $m$ is the number of instances in the dataset.

- $K$ is the number of classes.

- $y_k^{(i)}$ is the target probability that the i-th instance belongs to class $k$.

- $\hat{p}_k^{(i)}$ is the model's estimated probability that the i-th instance belongs to class $k$.

The gradients of the Cross-Entropy cost function are given by:

$\nabla_{\theta^{(k)}} J(\Theta) = \frac{1}{m} \sum_{i=1}^{m} (\hat{p}_k^{(i)} - y_k^{(i)})x^{(i)}$

![Softmax Regression](/images/softmax_regression.png)

### Exercises

1. What Linear Regression training algorithm can you use if you have a training set with millions of features?

    - If you have a training set with millions of features, you may want to use a training algorithm that can handle large datasets efficiently. One option is to use **Stochastic Gradient Descent** (SGD) or **Mini-batch Gradient Descent**. These algorithms are well-suited for large datasets because they process instances individually or in small batches, making them faster than Batch Gradient Descent.

2. Suppose the features in your training set have very different scales. What algorithms might suffer from this, and how? What can you do about it?

    - Algorithms that are sensitive to feature scales, such as **Gradient Descent** and **Regularized Linear Models**, may suffer when features have very different scales. This is because the algorithm may take a long time to converge or may not converge at all. To address this issue, you can scale the features using techniques like **Min-Max Scaling** or **Standardization**. These techniques ensure that all features have similar scales, making the algorithm converge faster and more reliably.

3. Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?

    - No, Gradient Descent cannot get stuck in a local minimum when training a Logistic Regression model. This is because the cost function for Logistic Regression is convex, meaning it has only one global minimum. Gradient Descent is guaranteed to find the global minimum for convex cost functions, so it will always converge to the optimal solution for Logistic Regression.

4. Do all Gradient Descent algorithms lead to the same model, provided you let them run long enough?

    - No, not all Gradient Descent algorithms lead to the same model, even if you let them run long enough. The choice of learning rate, batch size, and other hyperparameters can affect the final model. For example, **Stochastic Gradient Descent** (SGD) and **Mini-batch Gradient Descent** may converge to a different solution than **Batch Gradient Descent**. Additionally, the learning rate schedule and regularization techniques can also influence the final model.

5. Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?

    - If the validation error consistently goes up when using Batch Gradient Descent, it is likely that the model is overfitting the training data. This means the model is learning the noise in the training data and is not generalizing well to new data. To fix this issue, you can try the following techniques:
      - **Early Stopping**: Stop training when the validation error starts to increase.
      - **Regularization**: Add regularization to the model to prevent overfitting.
      - **Feature Selection**: Remove irrelevant features from the dataset.
      - **Cross-Validation**: Use k-fold cross-validation to evaluate the model's performance on different subsets of the data.

6. Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?

    - No, it is not a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up. This is because the validation error can fluctuate due to noise in the data. Instead, you can use techniques like **Early Stopping** to stop training when the validation error consistently increases over several epochs. This helps prevent overfitting and ensures the model generalizes well to new data.

7. Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?

    - Among the Gradient Descent algorithms we discussed, **Stochastic Gradient Descent** (SGD) will reach the vicinity of the optimal solution the fastest. This is because SGD updates the model parameters after processing each instance, making it faster than Batch Gradient Descent and Mini-batch Gradient Descent. However, SGD may not converge to the optimal solution due to its erratic path. To make the other algorithms converge faster, you can use techniques like **Learning Rate Schedules**, **Regularization**, and **Feature Scaling**. These techniques help the algorithms converge more efficiently and reliably.

8. Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?

    - If there is a large gap between the training error and the validation error in Polynomial Regression, it is likely that the model is overfitting the training data. This means the model is too complex and is learning the noise in the training data. To solve this issue, you can try the following techniques:
      - **Reduce the Model Complexity**: Use a lower degree for the polynomial features to reduce the model's complexity.
      - **Regularization**: Add regularization to the model to prevent overfitting.
      - **Feature Selection**: Remove irrelevant features from the dataset to reduce the model's complexity.

9. Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter $\alpha$ or reduce it?

    - If the training error and the validation error are almost equal and fairly high in Ridge Regression, the model likely suffers from high bias. This means the model is too simple to capture the underlying structure of the data. To address this issue, you can try the following techniques:
      - **Increase the Model Complexity**: Use a higher degree for the polynomial features to increase the model's complexity.
      - **Reduce the Regularization Hyperparameter**: Reduce the value of $\alpha$ to allow the model to fit the data better.

10. Why would you want to use:

    - Ridge Regression instead of plain Linear Regression (i.e., without any regularization)?
      - You would want to use Ridge Regression instead of plain Linear Regression when the model is overfitting the training data. Ridge Regression adds a regularization term to the cost function, which penalizes large model parameters. This helps prevent overfitting and ensures the model generalizes well to new data.

    - Lasso Regression instead of Ridge Regression?
      - You would want to use Lasso Regression instead of Ridge Regression when you want to perform feature selection. Lasso Regression adds an L1 regularization term to the cost function, which tends to eliminate the weights of the least important features. This can be useful when you have a large number of features and want to reduce the model's complexity.

    - Elastic Net instead of Lasso Regression?
      - You would want to use Elastic Net instead of Lasso Regression when you want a middle ground between Ridge Regression and Lasso Regression. Elastic Net combines both L1 and L2 regularization terms, allowing you to control the mix ratio $r$. This gives you more flexibility in choosing the regularization technique that best fits your data.

11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or one Softmax Regression classifier?

    - If you want to classify pictures as outdoor/indoor and daytime/nighttime, you should implement two Logistic Regression classifiers. This is because the classes are not mutually exclusive, meaning an instance can belong to both classes (e.g., outdoor and daytime). Softmax Regression is used for multiclass classification tasks where the classes are mutually exclusive, so it is not suitable for this scenario.

12. Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn).

## Chapter 5 - Support Vector Machines

**Support Vector Machines (SVM)** are a powerful and versatile machine learning algorithm. They are capable of performing linear or nonlinear classification, regression, and outlier detection tasks. SVMs are particularly well-suited for complex classification tasks with small to medium-sized datasets.

### Linear SVM Classification

**Linear SVM Classification** is a classification task that separates instances into two classes using a linear decision boundary. The goal is to find the widest possible margin that separates the classes. This margin is defined by the support vectors, which are the instances located on the edge of the margin.

The decision boundary is defined by the hyperplane that maximizes the margin between the classes. The hyperplane is the line that separates the classes, while the margin is the distance between the hyperplane and the support vectors.

The objective of Linear SVM Classification is to find the hyperplane that maximizes the margin while minimizing the number of margin violations (instances that are inside the margin or on the wrong side of the hyperplane).

Think of this as a line that separate the classes. The line is the hyperplane, while the margin is the distance between the line and the closest instance of the classes. A frontier is a line that separates the classes and the margin is the distance between the frontier and the closest instance of the classes.

The decision function of a Linear SVM is given by:

$\hat{y} = \begin{cases} 0 & \text{if } \theta^T \cdot x < t \\ 1 & \text{if } \theta^T \cdot x \geq t \end{cases}$

#### Hard Margin Classification

By defining that all instances must be outside the margin and on the correct side of the hyperplane, we have a **Hard Margin Classification**. This approach is sensitive to outliers and may not work well with noisy data. If the data is not linearly separable, the model may not converge.

#### Soft Margin Classification

To address the limitations of Hard Margin Classification, we can use **Soft Margin Classification**. This approach allows some instances to be inside the margin or on the wrong side of the hyperplane. The objective is to find a balance between maximizing the margin and minimizing the number of margin violations.

The Soft Margin Classification is controlled by the hyperparameter $C$, which determines the tradeoff between margin width and margin violations. A low $C$ value allows more margin violations but a wider margin, while a high $C$ value allows fewer margin violations but a narrower margin.

---

The SVM algorithm is sensitive to feature scales, so it is important to scale the features before training the model. This can be done using techniques like **Min-Max Scaling** or **Standardization**.

Instead of a train using the class `LinearSVC` from `scikit-learn` we can use `SGDClassifier` with **loss=hinge** and **alpha=1/(m*C)**. This is because the `LinearSVC` class is based on the `liblinear` library, while the `SGDClassifier` class is based on the `liblinear` library.

The `SGDClassifier` class can be used to train large datasets efficiently, as it processes instances individually or in small batches. This makes it well-suited for large datasets that do not fit in memory.

### Nonlinear SVM Classification

**Nonlinear SVM Classification** is a classification task that separates instances into two classes using a nonlinear decision boundary. This is achieved by transforming the features into a higher-dimensional space, where the classes are linearly separable.

This can be done by setting a `Polynomial features` or a `Similarity features`. The `Polynomial features` is used to add polynomial features to the dataset, while the `Similarity features` is used to add features based on the similarity of the instances to a landmark.

### Polynomial Kernel

The **Polynomial Kernel** is a kernel trick that allows SVMs to perform nonlinear classification tasks by adding polynomial features to the dataset. This is achieved by using the `PolynomialFeatures` transformer from `scikit-learn` to add polynomial features to the dataset.

We can use the `SVC` class from `scikit-learn` to train a SVM model with a polynomial kernel. The `SVC` class uses the `libsvm` library to train the model, which is more efficient than the `LinearSVC` class for large datasets.

### Similarity features

Another way to deal with nonlinear classification tasks is to add features based on the similarity of the instances to a landmark. This is achieved by using the **Similarity Features** technique, which transforms the instances into a higher-dimensional space where the classes are linearly separable.

The Similarity Features technique is useful for clustering and anomaly detection tasks, where the instances are not linearly separable. By transforming the instances into a higher-dimensional space, the SVM can find a decision boundary that separates the classes.

$\phi_{\gamma}(x, l) = \exp(-\gamma ||x - l||^2)$

Where:

- $\phi_{\gamma}(x, l)$ is the similarity feature vector for the instance $x$ and the landmark $l$.

- $\gamma$ is the similarity parameter.

- $||x - l||$ is the Euclidean distance between the instance $x$ and the landmark $l$.

![Similiarity](/images/simialirity.png)

### Gaussian RBF Kernel

The **Gaussian RBF Kernel** is a kernel trick that allows SVMs to perform nonlinear classification tasks by adding similarity features to the dataset. This is achieved by using the `RBF` kernel from `scikit-learn` to add similarity features to the dataset.

#### Hyperparameters gamma and C

The **gamma** hyperparameter controls the width of the Gaussian RBF kernel. A low gamma value makes the bell-shaped curve wider, while a high gamma value makes it narrower. The gamma hyperparameter can be used to control the model's complexity and prevent overfitting.

The **C** hyperparameter controls the tradeoff between margin width and margin violations. A low C value allows more margin violations but a wider margin, while a high C value allows fewer margin violations but a narrower margin. The C hyperparameter can be used to control the model's bias and variance.

#### Which kernel to use?

When training an SVM model, it is important to choose the right kernel for the task. The **Linear Kernel** is suitable for linear classification tasks, while the **Polynomial Kernel** and **Gaussian RBF Kernel** are suitable for nonlinear classification tasks. Therefore, the choice of kernel depends on the complexity of the data and the desired decision boundary.

### Computational Complexity

`LinearSVC` class is based on the `liblinear` which is an optimization algorithm that supports linear SVMs. The algorithm is efficient for large datasets and can handle millions of instances and features. The computational complexity of the `LinearSVC` class is about $O(m \times n)$.

`SVC` class is based on the `libsvm` library, which is an optimization algorithm that supports nonlinear SVMs. The algorithm is efficient for large datasets and can handle millions of instances and features. The computational complexity of the `SVC` class is about $O(m^2 \times n)$ to $O(m^3 \times n)$.

### Regression with SVM

**Support Vector Machines (SVM)** can also be used for regression tasks. This is achieved by using the **Support Vector Regression (SVR)** algorithm, which is a variant of SVM that supports regression tasks.

The objective of SVR is to find a function that approximates the data points while minimizing the margin violations. The margin violations are controlled by the hyperparameter $\epsilon$, which determines the width of the margin.

### Prediction function and decision function

The prediction function of SVR is given by:

$\hat{y} = w^T \cdot x + b$

If the result is greather than $0$ the model predicts the instance belongs to the positive class, otherwise it predicts the instance belongs to the negative class.

### Training goal

Consider the inclination of the decision function as: normal to the weight vector, $||w||$. If we divide by $2$ the points where the decision function is equal to $1$ and $-1$, will be twice as far from frontier. Divide the inclination by two is the same as multiply the margin by two.

Basically the goal of this model is to minimize $||w||$ to get a large margin and minimize the margin violations.

We need a decidion function to be greather than $1$ for positive instances and less than $-1$ for negative instances.  We can define $t^{(i)} = -1$ for negative instances and $t^{(i)} = 1$ for positive instances, also we can express the restrictions as $t^{(i)}(w^T \cdot x^{(i)} + b) \geq 1$.

So the goal of a SVM linear is a optimization problem that minimize $||w||$ and minimize the margin violations. This can be expressed as:

$\min_{w, b} \frac{1}{2} w^T \cdot w$

Subject to:

$t^{(i)}(w^T \cdot x^{(i)} + b) \geq 1$

To we get a soft margin we must add a slack variable $\zeta^{(i)} >= 0$ to each instance. The slack variable is used to measure the margin violations. The goal is to minimize the margin violations while maximizing the margin.

Now we have two goals: 

1. Get the slack variable as low as possible.

2. Get $1/2 w^T \cdot w$ as low as possible.

For this we have a parameter $C$ that controls the tradeoff between the two goals. A low $C$ value allows more margin violations but a wider margin, while a high $C$ value allows fewer margin violations but a narrower margin.

The optimization problem is given by:

$\min_{w, b, \zeta} \frac{1}{2} w^T \cdot w + C \sum_{i=1}^{m} \zeta^{(i)}$

Subject to:

$t^{(i)}(w^T \cdot x^{(i)} + b) \geq 1 - \zeta^{(i)}$

$\zeta^{(i)} \geq 0$

### Quadratic Programming

The optimization problem of the SVM can be solved using Quadratic Programming. Quadratic Programming is a mathematical optimization technique that finds the minimum of a quadratic function subject to linear constraints.

The Quadratic Programming problem is given by:

$\min_{p} \frac{1}{2} p^T \cdot H \cdot p + f^T \cdot p$

Subject to:

$A \cdot p \leq b$

A way to train a classifier linear SVM with tough margin is to use a QP solver.

### Dual Problem

The optimization problem of the SVM can be expressed as the **Primal Problem** or the **Dual Problem**. The **Primal Problem** is the original optimization problem, while the **Dual Problem** is an equivalent formulation that can be solved more efficiently.

The **Dual Problem** is given by:

$\min_{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha^{(i)} \alpha^{(j)} t^{(i)} t^{(j)} (x^{(i)} \cdot x^{(j)}) - \sum_{i=1}^{m} \alpha^{(i)}$

Subject to:

$\alpha^{(i)} \geq 0$

After solving the **Dual Problem**, the model parameters $w$ and $b$ can be computed using the following equations:

$w = \sum_{i=1}^{m} \alpha^{(i)} t^{(i)} x^{(i)}$

$b = \frac{1}{n_s} \sum_{i=1}^{m} (t^{(i)} - w^T \cdot x^{(i)})$

A problem with the **Dual Problem** is that it requires computing the dot product of all pairs of instances, which can be computationally expensive for large datasets. To address this issue, the **Kernel Trick** can be used to compute the dot product in a higher-dimensional space without actually transforming the instances.

### Kernel Trick

Let's suppose that you want to apply a polynomial transformation of degree $d$ to the train dataset with $d$ dimensions. The transformation is given by:

$\phi(x) = \phi([x_1, x_2]) = [x_1^2, \sqrt{2}x_1x_2, x_2^2]$

for $d=2$. The dot product of the transformation is given by:

$\phi(x)^T \cdot \phi(z) = [x_1^2, \sqrt{2}x_1x_2, x_2^2] \cdot [z_1^2, \sqrt{2}z_1z_2, z_2^2] = x_1^2z_1^2 + 2x_1z_1x_2z_2 + x_2^2z_2^2 = (x_1z_1 + x_2z_2)^2$

This can be reached as $(x^T \cdot z)^2$. 

The dot product of the transformation is equal to the square of the dot product of the original instances. This property is known as the **Kernel Trick**

Common kernels are:

- **Linear Kernel**: $K(x, z) = x^T \cdot z$

- **Polynomial Kernel**: $K(x, z) = (\gamma x^T \cdot z + r)^d$

- **Gaussian RBF Kernel**: $K(x, z) = \exp(-\gamma ||x - z||^2)$

- **Sigmoid Kernel**: $K(x, z) = \tanh(\gamma x^T \cdot z + r)$

The primal problem introduce a problem to computing the prediction function because of the dimension of the transformation. We can use the formula $h(x) = \sum_{i=1}^{m} \alpha^{(i)} t^{(i)} K(x^{(i)}, x) + b$ to compute the prediction function without the transformation.

### SVM online

The **Online SVM** algorithm is an extension of the SVM algorithm that supports online learning. This means the model can be trained on a stream of data instances, one at a time, without needing to store the entire dataset in memory.

The Online SVM algorithm uses the **Hinge Loss** function to update the model parameters after processing each instance. The Hinge Loss function penalizes the model when it makes a margin violation, which helps the model learn from the data.

The Online SVM algorithm is well-suited for large datasets that do not fit in memory and for tasks that require real-time processing of data streams.

The cost function of the Online SVM algorithm is given by:

$J(\theta) = \max(0, 1 - t \cdot (\theta^T \cdot x))$

### Exercises

1. What is the fundamental idea behind Support Vector Machines?

    - The fundamental idea behind Support Vector Machines (SVM) is to find the optimal decision boundary that separates instances into two classes. The decision boundary is defined by the hyperplane that maximizes the margin between the classes. The margin is the distance between the hyperplane and the support vectors, which are the instances located on the edge of the margin.

2. What is a support vector?
  
      - A support vector is an instance located on the edge of the margin that defines the decision boundary in a Support Vector Machine (SVM). The support vectors are the instances that are closest to the hyperplane and have the largest influence on the decision boundary.

3. Why is it important to scale the inputs when using SVMs?

    - It is important to scale the inputs when using SVMs because the algorithm is sensitive to feature scales. SVMs use the distance between instances to define the decision boundary, so features with larger scales can dominate the optimization process. Scaling the inputs ensures that all features have similar scales, making the algorithm more robust and efficient.

4. Can an SVM classifier output a confidence score when it classifies an instance? What about a probability?

    - Yes, an SVM classifier can output a confidence score when it classifies an instance. The confidence score is the distance between the instance and the decision boundary, which indicates how confident the model is in its prediction. However, SVMs do not output probabilities directly, as they are not probabilistic models. To estimate probabilities, you can use Platt scaling or cross-validation.

5. Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?

    - When training an SVM model on a large dataset with millions of instances and hundreds of features, it is generally more efficient to use the **Dual Form** of the SVM problem. The Dual Form is more computationally efficient for large datasets because it involves computing the dot product of all pairs of instances, which can be expensive for large datasets. The Dual Form also allows the use of the **Kernel Trick** to compute the dot product in a higher-dimensional space without actually transforming the instances.

6. Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease $\gamma$? What about C?

    - If an SVM classifier with an RBF kernel underfits the training set, you should increase the value of $\gamma$. The $\gamma$ hyperparameter controls the width of the Gaussian RBF kernel, so increasing $\gamma$ makes the bell-shaped curve narrower, which can help the model fit the data better. You can also try increasing the value of the $C$ hyperparameter, which controls the tradeoff between margin width and margin violations. A higher $C$ value allows fewer margin violations but a narrower margin, which can help the model fit the data better.

7. How should you set the QP parameters ($H$, $f$, $A$, and $b$) to solve the soft margin linear SVM classifier problem using an off-the-shelf QP solver?

    - To solve the soft margin linear SVM classifier problem using an off-the-shelf QP solver, you need to set the QP parameters as follows:
      - $H = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$
      - $f = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$
      - $A = \begin{pmatrix} -t^{(1)} & -t^{(2)} & -t^{(3)} \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$
      - $b = \begin{pmatrix} -1 \\ 0 \\ 0 \end{pmatrix}$


## Chapter 6 - Decision Trees

**Decision Trees** are a versatile machine learning algorithm that can perform both classification and regression tasks. They are easy to interpret and visualize, making them useful for understanding how a model makes predictions.

### Training and Visualizing a Decision Tree

We can use the `tree` module from `scikit-learn` to train a Decision Tree model. The `DecisionTreeClassifier` class is used for classification tasks, while the `DecisionTreeRegressor` class is used for regression tasks.

To visualize the Decision Tree model, we can use the `export_graphviz` function from the `tree` module. This function generates a Graphviz file that can be visualized using the `dot` command-line tool.

The command-line is given by:

```bash

dot -Tpng tree.dot -o tree.png

```

![codes/decision_tree](/codes/decision_tree/iris_tree.png)

### Making Predictions

As it is possible to see in the image above, the Decision Tree model makes predictions by following the decision nodes from the root to the leaf nodes. The model predicts the class of an instance by following the decision nodes based on the feature values of the instance.

One of great advantages of ML models is that they are not black boxes. It is possible to understand how the model makes predictions by visualizing the decision nodes and the feature values that lead to the predictions.

Decision Trees does not require feature scaling or centering, as they are not sensitive to the scale of the features.

#### Hyperparameters

The Decision Tree model has several hyperparameters that can be tuned to improve the model's performance. Some of the most important hyperparameters are:

- **max_depth**: The maximum depth of the Decision Tree. A higher max_depth value makes the model more complex and prone to overfitting.

- **min_samples_split**: The minimum number of samples required to split a node. A higher min_samples_split value prevents the model from splitting nodes with too few samples.

- **min_samples_leaf**: The minimum number of samples required to be a leaf node. A higher min_samples_leaf value prevents the model from creating leaf nodes with too few samples.

- **max_features**: The maximum number of features to consider when splitting a node. A lower max_features value reduces the model's complexity and prevents overfitting.

- **sample_weight**: The weight of each sample in the dataset. This hyperparameter can be used to assign different weights to the samples, which can be useful for imbalanced datasets.

- **gini**: The impurity measure used to split nodes. The Gini impurity is a measure of how often a randomly chosen element would be incorrectly classified. The impurity can be calculated using the formula:

$G_i = 1 - \sum_{k=1}^{n} p_{i, k}^2$

Where:

- $G_i$ is the Gini impurity of node $i$.

- $p_{i, k}$ is the ratio of class $k$ instances in node $i$.

---

A decision tree also can be used to estimate the probability that an instance belongs to a particular class. The probability is given by the ratio of the instances of the class in the leaf node.

### CART Training Algorithm

The **Classification and Regression Tree (CART)** algorithm is a popular training algorithm used to train Decision Tree models. The CART algorithm works by recursively splitting the dataset into subsets based on the feature values of the instances.

The CART algorithm splits the dataset into two subsets at each node by finding the feature and threshold that minimize the impurity of the subsets. The impurity is a measure of how mixed the classes are in the subset, and the goal is to split the dataset in a way that minimizes the impurity.

This can be expressed as a optimization problem that minimize the impurity of the subsets. The optimization problem is given by:

$\min_{k, t_k} \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}$

The CART algorithm continues splitting the dataset until it reaches the maximum depth of the tree or the minimum number of samples per leaf node. This helps prevent the model from overfitting the training data and ensures the model generalizes well to new data.

One of the biggest problems of Decision Trees is that they can be expressed as a NP-Complete problem. This means that the algorithm won't find the optimal solution in a reasonable time.

### Gini Impurity or Entropy

The **Gini Impurity** and **Entropy** are two common impurity measures used to split nodes in a Decision Tree. The Gini Impurity is a measure of how often a randomly chosen element would be incorrectly classified, while the Entropy is a measure of the average information content of a node. Entropy is zero when all instances in a node belong to the same class.

The entropy of a node is given by:

$H_i = -\sum_{k=1}^{n} p_{i, k} \log_2(p_{i, k})$

Gini's coefficient is faster to compute than entropy, but entropy tends to isolate the most frequent class in its own branch of the tree.

### Regression

The Decision Tree algorithm can also be used for regression tasks. This is achieved by using the `DecisionTreeRegressor` class from `scikit-learn` to train a Decision Tree model for regression tasks.

Instead of predicting a class label, the Decision Tree model predicts a continuous value for each instance. The model makes predictions by following the decision nodes from the root to the leaf nodes, based on the feature values of the instance.

![codes/decision_tree](/codes/decision_tree/iris_tree_reg.png)

CART algorithm works as the same way for regression tasks, but instead of minimizing the impurity of the subsets, it minimizes the MSE of the subsets.

### Instability

Decision Trees are prone to overfitting the training data, especially when the model is too complex or the dataset is noisy. One of their problem's is related to decision boundaries that are orthogonal to the axis. This means that the model is sensitive to small variations in the training data, which can lead to overfitting.

To address this issue, we can use techniques like **Pruning**, **Regularization**, and **Ensemble Learning**.

**Pruning** is a technique used to reduce the complexity of the Decision Tree model by removing nodes that do not improve the model's performance. Pruning helps prevent overfitting and ensures the model generalizes well to new data.

**Regularization** is a technique used to prevent overfitting by adding constraints to the model. This can be done by setting hyperparameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` to control the model's complexity.

**Ensemble Learning** is a technique used to combine multiple Decision Tree models to improve the model's performance. This can be done by using techniques like **Random Forests** and **Gradient Boosting** to train multiple Decision Tree models and combine their predictions.

### Exercises

1. What is the approximate depth of a Decision Tree trained (without restrictions) on a training set with one million instances?

    - The approximate depth of a Decision Tree trained on a training set with one million instances is about $O(\log_2(m))$, where $m$ is the number of instances in the dataset. For a dataset with one million instances, the depth of the Decision Tree would be about $O(\log_2(10^6)) = 20$.

2. Is a node's Gini impurity generally lower or greater than its parent's? Is it generally lower/greater, or always lower/greater?

    - A node's Gini impurity is generally lower than its parent's, as the Decision Tree algorithm splits the dataset in a way that minimizes the impurity of the subsets. The impurity is a measure of how mixed the classes are in the subset, so the goal is to split the dataset in a way that reduces the impurity of the nodes.

3. If a Decision Tree is overfitting the training set, is it a good idea to try decreasing `max_depth`?

    - If a Decision Tree is overfitting the training set, it is a good idea to try decreasing the `max_depth` hyperparameter. This reduces the complexity of the model and prevents it from memorizing the training data. By limiting the depth of the Decision Tree, you can prevent overfitting and ensure the model generalizes well to new data.

4. If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?

    - If a Decision Tree is underfitting the training set, it is not a good idea to try scaling the input features. Decision Trees are not sensitive to the scale of the features, so scaling the features will not improve the model's performance. Instead, you can try increasing the model's complexity by tuning the hyperparameters like `max_depth`, `min_samples_split`, and `min_samples_leaf`.

5. If it takes one hour to train a Decision Tree on a training set containing 1 million instances, roughly how much time will it take to train another Decision Tree on a training set containing 10 million instances?

    - If it takes one hour to train a Decision Tree on a training set containing 1 million instances, it will take about 10 hours to train another Decision Tree on a training set containing 10 million instances. The training time of a Decision Tree is proportional to the number of instances in the dataset, so training a model on a larger dataset will take longer.

6. If your training set contains 100,000 instances, will setting `presort=True` speed up training?

    - If your training set contains 100,000 instances, setting `presort=True` will not speed up training. The `presort` parameter is used to presort the data before training the model, which can be computationally expensive for large datasets. For a dataset with 100,000 instances, presorting the data may not provide a significant speedup and can slow down the training process.




## Chapter 7 - Ensemble Learning and Random Forests

**Ensemble Learning** is a machine learning technique that combines multiple models to improve the model's performance. Ensemble Learning can be used to reduce overfitting, increase model accuracy, and improve the model's robustness.

### Voting Classifiers

**Voting Classifiers** are a type of Ensemble Learning technique that combines multiple models to make predictions. There are three types of Voting Classifiers:

- **Hard Voting**: The model predicts the class that receives the most votes from the individual models.

- **Soft Voting**: The model predicts the class with the highest probability based on the individual models' predictions.

- **Weighted Voting**: The model assigns weights to the individual models' predictions based on their performance.

Voting Classifiers can be used for both classification and regression tasks. They work by combining the predictions of multiple models to make more accurate predictions.

It's possible to use the `VotingClassifier` class from `scikit-learn` to train a Voting Classifier model. The `VotingClassifier` class supports different voting strategies, including `hard`, `soft`, and `weighted`.

See the code [here](/codes/ensemble/Voting.py)

### Bagging and Pasting

**Bagging** and **Pasting** are Ensemble Learning techniques that combine multiple models to improve the model's performance. Bagging and Pasting work by training multiple models on different subsets of the training data and combining their predictions.

With **Bagging**, the models are trained on random subsets of the training data with replacement. This means that the same instance can be sampled multiple times in the same subset. Bagging is short for **Bootstrap Aggregating**.

With **Pasting**, the models are trained on random subsets of the training data without replacement. This means that each instance can only be sampled once in the same subset.

It's possible to use the `BaggingClassifier` and `BaggingRegressor` classes from `scikit-learn` to train Bagging models. The `BaggingClassifier` class supports different base estimators, including Decision Trees, Random Forests, and Support Vector Machines.

![Bagging decision boundaries](/images/bagging_decision_frontier.png)

In the image above, it's possible to see the decision boundaries of a decision tree and a bagging classifier. The bagging classifier has a smoother decision boundary, which helps prevent overfitting and improves the model's performance.

### Out-of-Bag Evaluation

**Out-of-Bag (OOB) Evaluation** is a technique used to evaluate the performance of a Bagging model without the need for a separate validation set. The OOB Evaluation works by using the instances that were not sampled in the training data to evaluate the model's performance.

The OOB Evaluation is useful for estimating the model's accuracy and generalization error without the need for a separate validation set. This can help save time and resources when training Bagging models.

### Random Patches and Random Subspaces

**Random Patches** and **Random Subspaces** are variations of Bagging that sample both instances and features to train the models. Random Patches sample both instances and features, while Random Subspaces sample only features.

### Random Forests

**Random Forests** are a type of Ensemble Learning technique that combines multiple Decision Tree models to improve the model's performance. Random Forests work by training multiple Decision Tree models on random subsets of the training data and combining their predictions.

They are trained using **bagging** or **pasting**. It's possible to use it directly by API instead of using the **BaggingClassifier** with the **DecisionTreeClassifier**.

Random Forests can be used for both classification and regression tasks. They work by combining the predictions of multiple Decision Tree models to make more accurate predictions.

### Extra-Trees

**Extra-Trees** are a variation of Random Forests that use random thresholds for each feature to split the nodes. This helps prevent overfitting and improves the model's performance.

Extra-Trees can be used to reduce the variance of the model and improve the model's generalization error. They work by training multiple Decision Tree models on random subsets of the training data and combining their predictions.

### Feature Importance

Random Forests can be used to estimate the importance of each feature in the dataset. The feature importance is a measure of how much each feature contributes to the model's predictions.

The feature importance is calculated by measuring the decrease in impurity or the decrease in the model's performance when a feature is removed from the dataset. Features that lead to a larger decrease in impurity are considered more important.

### Boosting

**Boosting** is an Ensemble Learning technique that combines multiple weak learners to create a strong learner. Boosting works by training multiple models sequentially, where each model corrects the errors of the previous model.

There are several types of Boosting algorithms, including **AdaBoost**, **Gradient Boosting**, and **XGBoost**. These algorithms work by training multiple models sequentially and combining their predictions to improve the model's performance.

### AdaBoost

**AdaBoost** is a type of Boosting algorithm that combines multiple weak learners to create a strong learner. AdaBoost works by training multiple models sequentially, where each model focuses on the instances that were misclassified by the previous model.

AdaBoost can be used for both classification and regression tasks. It works by training multiple models sequentially and combining their predictions to improve the model's performance.

See the code [here](/codes/ensemble/ada_boost.py)

### Gradient Boosting

**Gradient Boosting** is a type of Boosting algorithm that combines multiple weak learners to create a strong learner. Gradient Boosting works by training multiple models sequentially, where each model corrects the errors of the previous model.

Gradient Boosting adds sequential models to the ensemble, where each model focuses on the errors of the previous model. This helps improve the model's performance and reduce the model's bias.

### Stacking

**Stacking** is an Ensemble Learning technique that combines multiple models to improve the model's performance. Stacking works by training multiple models on the training data and combining their predictions to make more accurate predictions.

Stacking can be used to reduce overfitting, increase model accuracy, and improve the model's robustness. It works by training multiple models on the training data and combining their predictions to make more accurate predictions.

