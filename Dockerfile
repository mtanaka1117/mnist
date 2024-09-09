# https://qiita.com/kouki77/items/0fcfa470a17673df1541
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt update
RUN apt install -y python3 python3-pip


WORKDIR /mnist