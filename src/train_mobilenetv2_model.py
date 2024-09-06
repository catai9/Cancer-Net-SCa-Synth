# Adapted from https://www.kaggle.com/code/xhlulu/training-mobilenet-v2-in-4-min
# with modifications for specific dataset, binary classification, and class imbalance.
import argparse

import numpy as np
import pandas as pd

from keras import layers, optimizers
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from time import perf_counter


def build_model():
    mobilenet = MobileNetV2(
	weights='imagenet',
	include_top=False,
	input_shape=(224,224,3)
    )
    
    mobilenet.trainable = True
	    
    model = Sequential()
    model.add(mobilenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
	loss='binary_crossentropy',
	optimizer=optimizers.AdamW(learning_rate=0.0001, weight_decay=0.004),
	metrics=['AUC', 'accuracy']
    )	

    return model


parser = argparse.ArgumentParser(description="Input to train mobilenetv2 model script")
parser.add_argument(
    "--output_folder",
    type=str,
    default=None,
    required=True,
    help="Output folder for the model and csv test name.",
)

parser.add_argument(
    "--train_csv_location",
    type=str,
    default=None,
    required=True,
    help="Location to train csv file.",
)

parser.add_argument(
    "--train_folder_location",
    type=str,
    default=None,
    required=True,
    help="Location to train folder.",
)

parser.add_argument(
    "--test_csv_location",
    type=str,
    default=None,
    required=True,
    help="Location to test csv file.",
)

parser.add_argument(
    "--test_folder_location",
    type=str,
    default=None,
    required=True,
    help="Location to test folder.",
)

parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default='imagenet',
    help="Path to pretrained model in local file or omit if use ImageNet as default.",
)

args = parser.parse_args()

start_time = perf_counter()
print("Starting script")

# Load Labels
train_df = pd.read_csv(f'{args.train_csv_location}')
test_df = pd.read_csv(f'{args.test_csv_location}')

# Load Images
train_resized_imgs = []
test_resized_imgs = []

for image_id in train_df['image_name']:
    print(f"Processing image_id: {image_id}")
    img = np.load(f'{args.train_folder_location}/{image_id}.npy')
    train_resized_imgs.append(img)

for image_id in test_df['image_name']:
    print(f"Processing image_id: {image_id}")
    img = np.load(f'{args.test_folder_location}/{image_id}.npy')
    test_resized_imgs.append(img)

x_test = np.stack(test_resized_imgs)

# Split Data
x_train, x_val, y_train, y_val = train_test_split(
    np.stack(train_resized_imgs), 
    train_df['target'], 
    stratify=train_df['target'], 
    test_size=0.4, 
    random_state=2024
)


# Train Model    
model = build_model()

# Load weights if needed
if args.pretrained_model_name_or_path != 'imagenet':
    print("Loading custom weights")
    model.build(input_shape=(None, 224, 224, 3))
    model.load_weights(f'{args.pretrained_model_name_or_path}')

print("Model loaded")
checkpoint = ModelCheckpoint(f'{args.output_folder}/best_model.weights.h5', monitor='val_AUC', save_best_only=True, save_weights_only=True, mode='max', verbose=1)
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    verbose=2,
    callbacks=[checkpoint],
    validation_data=(x_val, y_val)
)

print("Model finished training")

# Load model again
model_new = build_model()
    
# Build the model to initialize all layers
model_new.build(input_shape=(None, 224, 224, 3))
model_new.load_weights(f'{args.output_folder}/best_model.weights.h5')


# Submission
y_test = (model_new.predict(x_test) > 0.5).astype("int32")
print("\nSum:", sum(y_test))

test_df['target'] = y_test
test_df.to_csv(f'{args.output_folder}/submission.csv',index=False)

end_time = perf_counter()
elapsed_time = end_time-start_time

print(f'Total time: {elapsed_time}')
