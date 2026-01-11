FROM pytorch/torchserve:latest

WORKDIR /home/model-server

COPY model-store /home/model-server/model-store

COPY config.properties /home/model-server/config.properties
COPY torchserve_handler /home/model-server/torchserve_handler

EXPOSE 8080 8081

ENTRYPOINT ["torchserve"]
CMD ["--start", "--foreground", "--disable-token-auth", "--ts-config", "config.properties", "--model-store", "model-store", "--models", "mymodel=mymodel.mar"]
