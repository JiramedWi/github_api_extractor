ARG UBUNTU_VER=22.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.10
ARG PANDAS_VER=2.0.3

FROM ubuntu:${UBUNTU_VER}
# FROM --platform=amd64 ubuntu:${UBUNTU_VER}
# System packages
RUN apt-get update && apt-get install -yq curl wget jq vim

# Use the above args
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
ENV script_name=x_for_docker.py
ENV script_name_2=y_prep_1_for_docker.py
ENV script_name_3=y_prep_2_for_docker.py
ENV script_name_4=y_prep_3_for_docker.py
ENV script_name_5=x_y_last_step_for_docker.py
ENV python_file_optuna_1=pre_train_dataset_for_docker.py
ENV python_file_optuna_2=optuna_traning_for_docker.py
RUN conda update -y conda
SHELL ["/bin/bash", "-c"]
RUN conda init bash


ARG PY_VER
ARG PANDAS_VER
# Install packages from conda
RUN conda install -c anaconda -y python=${PY_VER}
RUN conda install -c anaconda -y \
    pandas=${PANDAS_VER}
WORKDIR /app
COPY ERAWAN_env.yml ERAWAN_env.yml
#RUN conda env create -f ERAWAN_env.yml -n ERAWAN_env -y
RUN conda env create -f ERAWAN_env.yml
RUN source ~/.bashrc && source activate base && \ 
    conda activate ERAWAN_env && \ 
    python -m spacy download en_core_web_sm && \ 
    pip install optuna==3.6.0 spacy pandas textblob==0.17.1 gitpython scikit-learn scipy seaborn beautifulsoup4 imbalanced-learn smote-variants==0.7.3 && \
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
COPY . .
VOLUME /app/resources

# CMD python /src/{{script_name}}
#RUN conda env create -f ERAWAN_env.yml  && conda activate ERAWAN_env && python -m spacy download en_core_web_sm
#ENV script_name=java_script_upgrade.py
# RUN chmod +x ./src/collect_dataset/script_after_build.sh
RUN chmod +x ./src/collect_dataset/script_indexing_optuna.sh
# RUN sed -i 's/\r$//' ./src/collect_dataset/script_pre_processing.sh
RUN sed -i 's/\r$//' ./src/collect_dataset/script_indexing_optuna.sh
# CMD ./src/collect_dataset/script_pre_processing.sh ${script_name} ${script_name_2} ${script_name_3} ${script_name_4} ${script_name_5}
CMD ./src/collect_dataset/script_indexing_optuna.sh ${python_file_optuna_1}

