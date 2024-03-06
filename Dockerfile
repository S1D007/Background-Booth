FROM python:3.10

WORKDIR /app

RUN apt-get update
RUN apt-get install libsm6 libxext6 libgl1-mesa-glx -y

COPY requirements.txt /app/

RUN pip install carvekit --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
# install gunicorn
RUN pip install gunicorn
COPY . /app/

EXPOSE 8080

ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "api:app", "--timeout", "60", "--workers", "1"]
