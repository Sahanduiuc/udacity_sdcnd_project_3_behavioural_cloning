#!/bin/sh

mkdir -p ./data/raw/sample/IMG

wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip -O ./data/raw/sample/data.zip

unzip -o ./raw/sample/data.zip

mv ./data/raw/sample/data/IMG ./data/raw/sample/IMG
mv ./data/raw/sample/data/driving_log.csv ./data/raw/sample/driving_log.csv
rm -rf ./data/raw/sample/data

