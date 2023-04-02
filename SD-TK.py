import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random  
from PIL import Image, ImageTk
import tkinter as tk
import torch
from diffusers import StableDiffusionPipeline


def button_click():
    print("Button clicked!")
    print("Input text: ", input_text.get())

    prompt = input_text.get()
    imageFilename = random.randint(0, 10**8)

    ## Generating random seed ## 
    latents = None
    seed: int = imageFilename
    num_images = 1
    width = 512
    height = 512    

    generator = torch.Generator(device=device)
    generator = generator.manual_seed(seed)
    latents = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            generator = generator,
            device = device
    )

    with torch.autocast(device):
        images = pipe(
            [prompt] * num_images,
            guidance_scale=7.5,
            latents = latents,
        ).images[0].save(str(imageFilename) + ".png")
        

    # Load image file
    image0 = Image.open(str(imageFilename) + ".png")

    # Convert image to Tkinter format
    photo = ImageTk.PhotoImage(image0)
    image_label.configure(image=photo)
    image_label.image = photo

def exit_click():
    print("goodbye...")
    exit()
    
## init model ## 
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

## init TK ##
root = tk.Tk()

# Create input text field
input_text = tk.Entry(root, width=40)
input_text.pack()

# Create button
button = tk.Button(root, text="Run Stable Diffusion", command=button_click)
button.pack()

# Create exit
exitButton = tk.Button(root, text="exit", command=exit_click)
exitButton.pack()

# Create image display 
image_label = tk.Label(root)
image_label.pack(fill="both", expand="yes")

root.mainloop()


