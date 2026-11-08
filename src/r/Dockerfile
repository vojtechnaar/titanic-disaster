FROM rocker/r-ver:4.4.2

ENV DATA_DIR=/app/data

WORKDIR /app

# Install packages first so this layer is cached between code changes
COPY install_packages.R .
RUN Rscript install_packages.R

COPY titanic_model.R .

# The data is NOT baked into the image: mount src/data to /app/data at run time
CMD ["Rscript", "titanic_model.R"]
