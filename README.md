
This is my project on understanding the pipeline architecture for MLOps, focusing on the Brazilian E-Commerce Public Dataset by Olist, showcases an advanced implementation of machine learning operationalization techniques. The project leverages the strengths of ZenML, MLflow, and Streamlit to create a seamless pipeline from data preparation to deployment, emphasizing the practical application of ML models in a real-world e-commerce context.
The dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing charges from various dimensions: from order status, price, payment, freight performance to customer location, product attributes and finally, reviews written by customers. The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc.

ZenML Pipeline Implementation:

Data Preparation: The initial phase involves cleaning and structuring the dataset, ensuring it is suitable for analysis and model training. This step is crucial for the subsequent stages of the pipeline, as the quality of data directly impacts model performance.
Feature Engineering: This stage focuses on extracting and selecting the most relevant features from the dataset. By transforming raw data into a format that machine learning models can easily interpret, this step enhances the model's ability to learn and make accurate predictions.
Model Training: With clean and structured data, the model training phase involves selecting appropriate machine learning algorithms and training them on the dataset. This step is iterative, with the model's performance continuously assessed and optimized.
Model Evaluation: After training, the models are rigorously evaluated to determine their accuracy and effectiveness. This phase is critical for ensuring the models meet the desired performance benchmarks before deployment.
Model Deployment: The final step involves deploying the trained and evaluated models into production, making them accessible for real-time predictions and analysis. This phase marks the transition from a development environment to a real-world application.
MLflow for Visualization and Deployment:
MLflow plays a crucial role in this project by offering tools for tracking experiments, visualizing model performance, and managing the deployment lifecycle. It provides a unified interface to monitor various metrics and parameters, facilitating easy comparison between different models and iterations.

Streamlit Application Implementation:
The project culminates in the integration of the ML models into a Streamlit application, enabling users to interact with the models in a user-friendly environment. This application acts as the interface for the end-users, allowing them to input data, receive predictions, and visualize results in a straightforward and intuitive manner.

Overall, this MLOps project exemplifies a comprehensive approach to developing and deploying machine learning models, from the initial data processing stages to the final application implementation. Through the effective use of ZenML for pipeline creation, MLflow for visualization and deployment, and Streamlit for application development, the project demonstrates a sophisticated application of machine learning techniques in addressing real-world e-commerce challenges.

The project presents a comprehensive implementation of MLOps practices using the Brazilian E-Commerce Public Dataset by Olist. The core of the project revolves around utilizing ZenML, MLflow, and Streamlit to build, monitor, and deploy a machine learning model tailored to the dataset's insights.

ZenML is leveraged to create a robust pipeline that streamlines the machine learning workflow, encompassing data preparation, feature engineering, model training, model evaluation, and model deployment. This approach ensures a systematic and efficient management of the model development process, facilitating repeatability and scalability.

MLflow is utilized for its powerful visualization and deployment capabilities, offering a detailed view of the model's performance metrics and operational aspects. It aids in tracking experiments, managing artifacts, and simplifying the deployment process, thereby enhancing the model's reliability and accessibility.

Furthermore, the project incorporates a Streamlit application, making the machine learning model accessible and interactive for end-users. This application serves as a practical demonstration of the model's capabilities, allowing users to engage with the model in a user-friendly environment.

Overall, the project exemplifies a holistic approach to implementing MLOps, from model creation to deployment, highlighting the importance of pipeline management, visualization, and user interaction in the development of machine learning solutions.

- ZenML - makes pipeline for ML models. 
    - data-preparation 
    - feature engineering
    - model training
    - model evaluation
    - model deployment
    combine all above steps to make a pipeline in ZenML

commands:
- pip install "zenml["server"]"
- zenml init
- zenml up
- zenml local host username = default, no password
for mlflow install 
- zenml integration install mlflow -y

for traking uri command
- mlflow ui --backend-store-uri "file:/Users/vivek/Library/Application Support/zenml/local_stores/9a52ae25-e565-4fd0-b55d-bd76fabb70f8/mlruns"

for streamlit app 
- streamlit run streamlit_app.py  
