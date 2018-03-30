import imageio
import glob, os, shutil
from PIL import Image, ImageSequence, ImageFont, ImageDraw
from zipfile import ZipFile

"""
VARIABLES
"""
zip_filepath = os.path.normpath("C:\Users\kyle\Downloads\Opossum Default Almond.zip")

frames_per_sec = 24
show_local_best = True
local_best_range = 72 # range of past iterations to find local best. compare to frames_per_sec

# font = ImageFont.truetype(<font-file>, <font-size>)
font = ImageFont.truetype("fonts/OpenSans-Light.ttf", 12)
text_size = font.getsize("00.0000%")
font_color = (50,50,50) 
fitness_text_format = "{0:.4f}%"

limit_for_debugging = False
remove_temp_directory = True

"""
"""

output_directory = os.path.dirname(zip_filepath)
output_filename = os.path.splitext(os.path.split(zip_filepath)[1])[0] # output_filename mimics zip file name
print (output_filename)

#tmp_path = os.path.join(os.path.split(os.path.realpath(__file__))[0],"tmp")
#src_path = os.path.join(os.path.split(os.path.realpath(__file__))[0],"src")
tmp_path = os.path.join(output_directory,"tmp")
src_path = os.path.join(output_directory,"src")
#print(tmp_path)
if not os.path.exists(tmp_path): os.makedirs(tmp_path)
if not os.path.exists(src_path): os.makedirs(src_path)

with ZipFile(zip_filepath, 'r') as zf:
    #zip.printdir() # printing all the contents of the zip file
    zf.extractall(src_path)

#src_path = "C:\Users\kyle\Downloads\Opossum Default Lobed 10pt\Opossum Default Lobed 10pt"
#src_path = os.path.normpath(src_path)

src_files = [os.path.join(src_path,file) for file in os.listdir(src_path) if file.endswith(".png")]

try:
    # the first TXT file we come across must be the log file
    log_file = next(os.path.join(src_path, file) for file in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, file)) and file.endswith(".txt"))
    print(log_file)
except StopIteration:
    print("NO LOG FILE FOUND - there are no TXT files in that directory.")
    exit()

fitnesses = []
with open(log_file) as f:
    fitnesses = f.readlines()
fitnesses = [float(x.strip().split(' ')[-1]) for x in fitnesses] 

if (limit_for_debugging):
    fitnesses = fitnesses[:100]
    src_files = src_files[:100]

imgtups = zip(fitnesses,src_files)
    
global_best = imgtups[0]
local_best = imgtups[0]
for n, itup in enumerate(imgtups):
    if itup[0] >= global_best[0]: 
        #print(("found better instance: "+fitness_text_format+" > "+fitness_text_format+"").format(itup[0], global_best[0]))
        global_best = itup
    
    if not show_local_best:
        images = map(Image.open, (itup[1], local_best[1], global_best[1]))
    else:
        if n>0: local_best = sorted(imgtups[max(0,n-local_best_range):n])[-1]
        images = map(Image.open, (itup[1], local_best[1], global_best[1]))
    
    widths, heights = zip(*(i.size for i in images))
    total_width, max_height = sum(widths), max(heights) 

    img = Image.new('RGB', (total_width, max_height + text_size[1]*2), (255,255,255))

    x_offset = 0
    for im in images:
      img.paste(im, (x_offset,0))
      x_offset += im.size[0]
    
    
    if show_local_best:
        xposs = (
            total_width*0.1666 - text_size[0]/2, 
            total_width*0.8333 - text_size[0]/2,
            total_width*0.5 - text_size[0]/2
            )
    else:
        xposs = (
            total_width*0.75 - text_size[0]/2, 
            total_width*0.25 - text_size[0]/2
            )
    
    draw = ImageDraw.Draw(img)
    draw.text((xposs[0], max_height + text_size[1]/2),fitness_text_format.format(itup[0]*100),font_color,font=font)   
    draw.text((xposs[1], max_height + text_size[1]/2),fitness_text_format.format(global_best[0]*100),font_color,font=font) 
    if show_local_best: draw.text((xposs[2], max_height + text_size[1]/2),fitness_text_format.format(global_best[0]*100),font_color,font=font) 
    
    img.save(os.path.join(tmp_path,'{0:0>6}.jpg'.format(n)))
    
    
src_files = [os.path.join(tmp_path,file) for file in os.listdir(tmp_path)]    
writer = imageio.get_writer(os.path.join(output_directory,output_filename+'.mp4'), fps=frames_per_sec)
for im in src_files:
    writer.append_data(imageio.imread(im))
writer.close()

if remove_temp_directory: 
    shutil.rmtree(tmp_path)
    shutil.rmtree(src_path)