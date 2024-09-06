# Cancer-Net-SCa-Synth

This is the repo containing scripts and utilities for generating images of benign and malignant skin lesions using the Stable Diffusion model with DreamBooth. The latest version producing the best results uses scripts from HuggingFace which are licensed under the Apache 2.0 license (as of 2024-05-26).


## Project structure
```bash
.
├── data # Training data, recommend symlinking
│   ├── jpeg # JPEG images
│   │   ├── test # Test set
│   │   └── train # Training set
│   ├── test.csv # Metadata CSV for test set
│   └── train.csv # Metadata CSV for training set
├── src # Python source code
│   └── create_training_dataset.py # Script to generate folder of images for training SD using DreamBooth trainer
│   └── train_dreambooth.py # HuggingFace Python script for training SD using DreamBooth trainer
│   └── generate_images.py # Generate images using the trained SD with DreamBooth trainer
│   └── create_generated_csv_file.py # Create a csv file for the generated images
│   └── prepreprocess_data.py # Preprocess data for model training
│   └── train_mobilenetv2_model.py # Train MobileNetV2 model
# Misc individual files
├── requirements.txt
└── README.md
```

## Setup
The following setup should be done on a machine with a Nvidia GPU.

1. Download/prepare the ISIC 2020 skin lesion dataset
The dataset can be downloaded [here](https://www.kaggle.com/c/siim-isic-melanoma-classification/data).

To download from Kaggle:
```bash
kaggle competitions download -c siim-isic-melanoma-classification # Need Kaggle CLI installed
```

then unzip the file and create a symlink to the folder under the name `data/` in the project root:
```bash
ln -s <downloaded folder> data
```

2. Create a directory containing the desired training instances. A helper script can be used to generate this directory, sampling random benign/malignant images as follows:
```bash
python src/create_training_dataset.py \
    malignant \ 
    --source_csv data/train.csv \
    --source_dir data/jpeg/train \
    --target_dir data/generated/train-malignant \
    --subset_size 300
```

```bash
python src/create_training_dataset.py \
    benign \ 
    --source_csv data/train.csv \
    --source_dir data/jpeg/train \
    --target_dir data/generated/train-benign \
    --subset_size 300
```

Please run `python src/create_training_dataset.py -h` for more information on the script usage and available flags.

## Training Stable Diffusion (SD) using DreamBooth trainer

1. To train the SD model using the DreamBooth trainer, first create the environment variables:
```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5" # Model path, do not change
export INSTANCE_DIR="data/generated/train-malignant" # Training instance directory, update with directory generated from step 2
export OUTPUT_DIR="./malignant-model" # Model output directory, change as preferred
```

Then, run the following script and note some of the chosen script parameters:
```bash
accelerate launch src/train_dreambooth.py \ # Use accelerate to optimize resource usage
    --pretrained_model_name_or_path=$MODEL_NAME \ # Model
    --instance_data_dir=$INSTANCE_DIR \ # Training instances
    --output_dir=$OUTPUT_DIR \ # Model output path
    --instance_prompt="melanoma" \
    --resolution=512 \ # Image resolution (height and width)
    --train_batch_size=1 \ # Batch size (no batch training used)
    --gradient_accumulation_steps=1 \ # Number of update steps to do before doing an update pass to the model
    --learning_rate=5e-6 \ # LR
    --lr_scheduler="constant" \ # LR scheduling
    --lr_warmup_steps=0 \ # LR warmup
    --max_train_steps=400
```

In addition, the following default parameters are used:

- Preprocessing:
  - No center crop
- Adam optimizer:
  - `--adam_beta1 0.9`
  - `--adam_beta2 0.999`
  - `--adam_weight_decay 1e-2`
  - `--adam_epsilon 1e-08`


## Image generation
To generate the images using the trained model, run the `src/generate_images.py` script:

```bash
python src/generate_images.py \
    --pretrained_model_name_or_path="malignant-model" \
    --prompt="melanoma" \
    --num_images=5000 \
    --output_folder="data/jpeg/generated"
```

```bash
python src/generate_images.py \
    --pretrained_model_name_or_path="benign-model" \
    --prompt="benign" \
    --num_images="5000" \
    --output_folder="data/jpeg/generated"
```

To create the associated metadata csv file, run the `src/create_generated_csv_file.py` script:
```bash
python src/create_generated_csv_file.py \
    --raw_folder_location="data/jpeg/generated" \
    --csv_location="data/generated.csv"
```

## Data Processing
To standardize the data into a desired format for model training and testing, run the `src/prepreprocess_data.py` script: 

```bash
python src/preprocess_data.py \ # ISIC 2020 Train Set
    --raw_folder_location="data/jpeg/train" \
    --csv_location="data/train.csv" \
    --processed_output_folder="data/processed/train"
```

```bash
python src/preprocess_data.py \ # ISIC 2020 Test Set
    --raw_folder_location="data/jpeg/test" \
    --csv_location="data/test.csv" \
    --processed_output_folder="data/processed/test"
```

```bash
python src/preprocess_data.py \ # Synthetically Generated Set
    --raw_folder_location="data/jpeg/generated" \
    --csv_location="data/generated.csv" \
    --processed_output_folder="data/processed/generated"
```


## Train MobileNetV2 Model
To train the MobileNetV2 model, run the `src/train_mobilenetv2_model.py` script. Commands for each of the scenarios are listed below.

Scenario A: Train only on the ISIC 2020 Training Set
```bash
python src/train_mobilenetv2_model.py \
    --output_folder results/scenarioA \
    --train_csv_location data/train.csv \
    --train_folder_location data/processed/train \
    --test_csv_location data/test.csv \
    --test_folder_location data/processed/test \
    --pretrained_model_name_or_path imagenet
```

Scenario B: Train only on Cancer-Net SCa-Synth
```bash
python src/train_mobilenetv2_model.py \
    --output_folder results/scenarioB \
    --train_csv_location data/generated.csv \
    --train_folder_location data/processed/generated \
    --test_csv_location data/test.csv \
    --test_folder_location data/processed/test \
    --pretrained_model_name_or_path imagenet 
```

Scenario C: Train on Cancer-Net SCa-Synth and Fine-tune with ISIC 2020 Training Set
```bash
python src/train_mobilenetv2_model.py \
    --output_folder results/scenarioC \
    --train_csv_location data/train.csv \
    --train_folder_location data/processed/train \
    --test_csv_location data/test.csv \
    --test_folder_location data/processed/test \
    --pretrained_model_name_or_path results/scenarioB/best_model.weights.h5 
```

