import os
from PIL import Image
import gym_duckietown
from datetime import datetime
from gym_duckietown.simulator import Simulator

# Creating directories for images and logs
img_dir = "images"
test_name = "raw_obs"
save_dir = img_dir + "/" + test_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

env = Simulator(
        seed=666, # random seed
        map_name="loop_empty",
        max_steps=500001, # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=1, # start close to straight
        full_transparency=True,
        distortion=True,
    )

# Wrapping the env

       
images = []
#while True:
for step in range(10):
    action = [0.1,0.1]
    observation, reward, done, misc = env.step(action)
    env.render()
    if done:
        env.reset()
    
    images.append(observation)

env.close()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for img in range(len(images)):
    ftype = ".jpg"    
    pil_img = Image.fromarray(images[img])
    pil_img.save(save_dir+"/"+timestamp+ "_" +str(img+1)+ftype)

print("\nTest ended succesfully. Sample images ready.\n")