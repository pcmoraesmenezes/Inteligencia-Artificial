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