import argparse
import datetime
import math
from PIL import Image

def splice_images(filenames, border_angle=0, output_filename=None):
    if not output_filename:
        output_filename = f'{datetime.datetime.now()}-spliced.png'

    print(f'[INFO] Splicing images {filenames} with border angle {border_angle} to {output_filename}.')

    images = [Image.open(filename) for filename in filenames]

    for i, img in enumerate(images):
        if img is None:
            print(f'[ERR!] Fatal: unable to read/open {images[i]} properly.')
            return
        if img.size != images[0].size:
            print(f'[ERR!] Fatal: shape mismatch between images, got {img.shape}, expected {images[0].shape}.')
            return

    # max ang = tan-1 (2w/hn)

    output = Image.new('RGB', images[0].size)
    out = output.load()
    img_datas = [img.load() for img in images]
    
    width = images[0].size[0]
    height = images[0].size[1]
    tan = math.tan(math.radians(border_angle))

    tan_limit = 2 * width / (height * len(images))
    if tan > tan_limit:
        print(f'[ERR!] Fatal: border angle too large, maximal is {math.degrees(math.atan(tan_limit))}.')
        return

    eps = 0.005
    for x in range(width):
        for y in range(height):
            _x = x
            _y = height - y

            i = (_x - tan * _y + (height / 2) * tan) * len(images) / width
            _i = max(0, min(int(i), len(images) - 1))
            
            if abs(i - _i) < eps and _i > 0:
                out[x, y] = (255, 255, 255)
                continue
        
            try:
                out[x, y] = img_datas[_i][x, y]
            except IndexError:
                print(f'[ERR!] Fatal: index error at {x}, {y}, {_i}.')
                return
    

    output.save(output_filename)

    return output
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Output a spliced version of a set of images.')

    # Add arguments
    parser.add_argument('-filenames', metavar='filenames', type=str, nargs='*',
                        help='file(s) to splice')

    args = parser.parse_args()

    splice_images(args.filenames)



# # split it so that the first 1/n columns are from the first image, the next 1/n columns are from the second image, etc.
# combined_image = Image.new('RGB', images[0].size)

# # Paste each image onto the combined image 
# div_width = images[0].size[0] // len(images)
# border_offset = int(div_width * 0.1)  # Adjust this value as needed
# height = images[0].size[1]

# for i, img in enumerate(images):
#     combined_image.paste(img.crop((i * div_width, 0, (i + 1) * div_width, height)), (i * div_width, 0))
#     # combined_image.paste(img, (i * div_width, 0, (i + 1) * div_width, images[0].size[1]))

# combined_image.save(output_filename)

# return combined_image