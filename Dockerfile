ARG UBUNTU_VER=22.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.10
ARG PANDAS_VER=2.0.3

FROM --platform=amd64 ubuntu:${UBUNTU_VER}
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
ENV script_name=java_script_upgrade.py
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
COPY . .
#RUN conda env create -f ERAWAN_env.yml -n ERAWAN_env -y
RUN conda env create -f ERAWAN_env.yml
RUN source ~/.bashrc && source activate base && conda activate ERAWAN_env && python -m spacy download en_core_web_sm && pip install spacy pandas gitpython scikit-learn scipy seaborn beautifulsoup4 imbalanced-learn smote-variants==0.7.3
# RUN conda run -n ERAWAN_env && pip install spacy pandas gitpython scikit-learn scipy seaborn beautifulsoup4 imbalanced-learn smote-variants==0.7.3
VOLUME /app/resources

# CMD python /src/{{script_name}} 
#RUN conda env create -f ERAWAN_env.yml  && conda activate ERAWAN_env && python -m spacy download en_core_web_sm
#ENV script_name=java_script_upgrade.py
#CMD python ./src/collect_dataset/{{script_name}}