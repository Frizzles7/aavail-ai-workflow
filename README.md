# IBM AI Enterprise Workflow - Capstone Project

## Overview of Contents

- **app.py**: flask app to predict, train, and access logfiles
- **data**: directory containing data files
- **Dockerfile**: commands to build the Docker image
- **logs**: directory for storing training and prediction logs
- **models**: directory for storing models
- **notebooks**: contains notebook for exploratory data analysis and visualizations
- **requirements.txt**: file containing the packages used in this repo
- **rununittests.py**: python script to run unit tests
- **src**: directory containing python scripts for training and prediction
- **templates**: simple templates for rendering flask app
- **unittests**: directory containing scripts for unit tests

## Testing in Python

To test the flask app:
```bash
~$ python app.py
```
Go to http://0.0.0.0:8080 to see a basic website for this project.

To run the all of the unit tests, including model, logging, and API tests:
```bash
~$ python rununittests.py
```

To run only the model tests:
```bash
~$ python unittests/ModelTests.py
```

To run only the logging tests:
```bash
~$ python unittests/LoggerTests.py
```

To run only the API tests:
```bash
~$ python unittests/ApiTests.py
```

To test the training and prediction of the models:
```bash
~$ python src/model.py
```

## Testing in Docker

To build the Docker container:
```bash
~$ sudo docker build -t capstone .
```

To verify that the image is there:
```bash
~$ sudo docker image ls
```

To run the container:
```bash
~$ sudo docker run -p 4000:8080 capstone
```

Go to http://0.0.0.0:4000/ to verify that the app is running.

To quit, press CTRL+C.

To verify the container is there:
```bash
~$ sudo docker ps -a
```

To remove the container using the container name or id from above step:
```bash
~$ sudo docker rm [container_name_or_id]
```

To remove the image:
```bash
~$ sudo docker image rm capstone
```

