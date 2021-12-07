# Install dependencies
pip3 install -r requirements.txt

# Download datasets
gdrive download 11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS
mv widerface.zip data
unzip data/widerface.zip -d data

# Download pretrained models
gdrive download --recursive 1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1
mkdir weights
mv Retinaface_model_v2/* weights
rm -r Retinaface_model_v2

# Build extensions
cd widerface_evaluate 
python3 setup.py build_ext --inplace
cd .. 

# Init clearml for logging
clearml-init
