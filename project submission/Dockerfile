FROM public.ecr.aws/lambda/python:3.12

# 1. Install libgomp using 'dnf' (Newer package manager for Python 3.12 image)
RUN dnf install -y libgomp git

# 2. Upgrade pip
RUN pip install --upgrade pip

# 3. Install libraries using BINARIES (Instant install)
RUN pip install pandas numpy scikit-learn xgboost lightgbm==4.5.0 joblib boto3 py7zr --only-binary=:all:

# 4. Copy your code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]