FROM pytorch/pytorch

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output /working \
    && chown user:user /opt/app /input /output /working
# RUN mkdir -p /output/images/inferior-alveolar-canal

ENV ND_ENTRYPOINT="/neurodocker/startup.sh"
RUN export ND_ENTRYPOINT="/neurodocker/startup.sh" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        bzip2 \
        ca-certificates \
        curl \
        locales \
        unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && chmod 777 /opt && chmod a+s /opt \
    && mkdir -p /neurodocker \
    && if [ ! -f "$ND_ENTRYPOINT" ]; then \
         echo '#!/usr/bin/env bash' >> "$ND_ENTRYPOINT" \
    &&   echo 'set -e' >> "$ND_ENTRYPOINT" \
    &&   echo 'if [ -n "$1" ]; then "$@"; else /usr/bin/env bash; fi' >> "$ND_ENTRYPOINT"; \
    fi \
    && chmod -R 777 /neurodocker && chmod a+s /neurodocker


RUN /neurodocker/startup.sh

ENV FSLDIR="/opt/fsl-5.0.11" \
    PATH="/opt/fsl-5.0.11/bin:$PATH"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           bc \
           dc \
           file \
           git \
           ssh \
           libfontconfig1 \
           libfreetype6 \
           libgl1-mesa-dev \
           libglu1-mesa-dev \
           libgomp1 \
           libice6 \
           libxcursor1 \
           libxft2 \
           libxinerama1 \
           libxrandr2 \
           libxrender1 \
           libxt6 \
           wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-5.0.11 \
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.11-centos6_64.tar.gz \
    | tar -xz -C /opt/fsl-5.0.11 --strip-components 1 \
    && sed -i '$iecho Some packages in this Docker container are non-free' $ND_ENTRYPOINT \
    && sed -i '$iecho If you are considering commercial use of this container, please consult the relevant license:' $ND_ENTRYPOINT \
    && sed -i '$iecho https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Licence' $ND_ENTRYPOINT \
    && sed -i '$isource $FSLDIR/etc/fslconf/fsl.sh' $ND_ENTRYPOINT \
    && echo "Installing FSL conda environment ..." \
    && bash /opt/fsl-5.0.11/etc/fslconf/fslpython_install.sh -f /opt/fsl-5.0.11

ENV POSSUMDIR=/usr/share/fsl/5.0
ENV FSLOUTPUTTYPE="NIFTI_GZ"
ENV FSLDIR=/usr/share/fsl/5.0
ENV FSLMULTIFILEQUIT=TRUE
ENV FSLTCLSH=/usr/bin/tclsh
ENV FSLWISH=/usr/bin


RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           make \
           gcc \
           clang \ 
           clang-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p niftyseg/src \
    && git clone https://github.com/KCL-BMEIS/NiftySeg.git niftyseg/src \ 
    && cd niftyseg && mkdir build \
    && cd build \
    && cmake ../src \
    && make && make install


USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PATH="/niftyseg/install/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user requirements.txt /opt/app/
# RUN python -m piptools sync requirements.txt
RUN python -m pip install --user -r requirements.txt

RUN git clone https://github.com/MIC-DKFZ/nnUNet.git \
    && cd nnUNet && pip install -e .

COPY --chown=user:user process.py /opt/app/

## Ficar fitxers .py i directoris models
COPY models/ /models
COPY --chown=user:user preprocess.py /opt/app/
COPY --chown=user:user postprocess.py /opt/app/
COPY --chown=user:user data_transformation.py /opt/app/
COPY --chown=user:user reshape_arrays.py /opt/app/
COPY --chown=user:user utils.py /opt/app/
COPY --chown=user:user voting_strategies.py /opt/app/


ENTRYPOINT [ "python", "-m", "process" ]
