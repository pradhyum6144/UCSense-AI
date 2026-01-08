# UCSense-AI Lambda Docker Image
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for OpenCV and Tesseract
RUN yum update -y && \
    yum install -y \
    tesseract \
    tesseract-langpack-eng \
    leptonica \
    libGL \
    libgthread-2.0.so.0 \
    mesa-libGL \
    && yum clean all

# Set Tesseract path
ENV TESSERACT_CMD=/usr/bin/tesseract

# Copy requirements and install Python dependencies
COPY backend/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Install mangum for Lambda ASGI adapter
RUN pip install --no-cache-dir mangum

# Copy application code
COPY backend/ ${LAMBDA_TASK_ROOT}/

# Copy trained model if available
COPY models/ ${LAMBDA_TASK_ROOT}/models/ 2>/dev/null || true

# Set the handler
CMD ["main.handler"]
