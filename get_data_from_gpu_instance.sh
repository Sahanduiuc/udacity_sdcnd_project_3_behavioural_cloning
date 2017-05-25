#!/bin/sh

# run the below first to ensure you can run this script using ssh to access remote
#  -> solution from: https://stackoverflow.com/a/16201608/893766
# 	ssh-copy-id user@1.2.3.4

instance="carnd@${1}"
timestamp=$(date +"%Y%m%d_%H%M%S")

project_dir="repos/udacity_carnd/project_3"

remote_log_dir="${project_dir}/log"
remote_model_dir="${project_dir}/model"
remote_notebook_dir="${project_dir}/notebook"

local_log_dir="${HOME}/${project_dir}/log"
local_model_dir="${HOME}/${project_dir}/model"
local_notebook_dir="${HOME}/${project_dir}/notebook"

mkdir -p "${local_log_dir}" && \
    scp -ri ~/.ssh/id_rsa "${instance}:"~"/${remote_log_dir}/"* "${local_log_dir}/"

mkdir -p "${local_model_dir}" && \
    scp -ri ~/.ssh/id_rsa "${instance}:"~"/${remote_model_dir}/"* "${local_model_dir}/"

scp -ri ~/.ssh/id_rsa "${instance}:"~"/${remote_notebook_dir}/"* "${local_notebook_dir}/"
