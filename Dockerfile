FROM tensorflow/tensorflow:latest-gpu-py3

ARG NLTK_DATA=/home/app/nltk_data
ENV NLTK_DATA=${NLTK_DATA}

COPY $PWD /home/app/
WORKDIR /home/app/

RUN pip install --upgrade pip \
	&& pip install \
		pandas \
		tqdm \
		numpy \
		nltk \
		scikit-learn 
RUN mkdir -p /home/app/nltk_data
