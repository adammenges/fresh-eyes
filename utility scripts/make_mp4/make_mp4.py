import imageio
import glob, os, shutil
#from PIL import Image, ImageSequence, ImageFont, ImageDraw
from zipfile import ZipFile

"""
VARIABLES
"""
img_filepath = r"C:\Users\kstei\Desktop\TEMP\walk"
img_extension = ".png"
output_directory = r"C:\Users\kstei\Desktop\TEMP"

output_filename = "OUT"
frames_per_sec = 24

src_files = [os.path.join(img_filepath,file) for file in os.listdir(img_filepath) if file.endswith(img_extension)]
out_filepath = os.path.join(output_directory,'{}.mp4'.format(output_filename))
writer = imageio.get_writer(out_filepath, fps=frames_per_sec)
for im in src_files: writer.append_data(imageio.imread(im))
writer.close()