import argparse
import torch

from diffusers import StableDiffusionPipeline
from PIL import Image

from time import perf_counter


parser = argparse.ArgumentParser(description="Input to generate images script")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model in local file, e.g., '/home/amytai/ura-2024-oustan/malignant-model'.",
)

parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    required=True,
    help="Prompt to use in the image generation, e.g., 'melanoma'",
)

parser.add_argument(
    "--num_images",
    type=int,
    default=100,
    help="Number of images to generate",
)

parser.add_argument("--gpu_id", type=int, default=2, help="GPU device ID")

parser.add_argument(
    "--output_folder",
    type=str,
    default=None,
    required=True,
    help="Path to output folder, e.g., '/home/amytai/ura-2024-oustan/data/jpeg/generated'.",
)

args = parser.parse_args()

start_time = perf_counter()
print("Starting script")

device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16, local_files_only=True, safety_checker = None, requires_safety_checker = False)
pipe = pipe.to(device)
# pipe.safety_checker = lambda images, clip_input: (images, False) # Remove safety check

num_rounds = round(args.num_images / 4)

for rnd in range(num_rounds): 
    images = pipe(args.prompt, num_images_per_prompt=4).images
    
    for idx, image in enumerate(images):
        image.save(f"{args.output_folder}/{args.prompt}-{(idx+rnd*4)}.jpg")
    
end_time = perf_counter()
elapsed_time = end_time-start_time

print(f'Total time: {elapsed_time}')
