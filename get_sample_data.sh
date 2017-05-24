#!/bin/sh

mkdir -p ./data/raw/sample

wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip -O ./data/raw/sample/data.zip

unzip -o ./data/raw/sample/data.zip -d ./data/raw/sample

