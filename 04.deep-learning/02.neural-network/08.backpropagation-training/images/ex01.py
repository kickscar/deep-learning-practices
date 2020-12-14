from PIL import Image, ImageOps
import numpy as np

im = ImageOps.invert(Image.open('test.bmp'))
w, h = im.size

pixels = np.zeros((h, w), dtype=np.uint8)

for r in range(0, h):
    for c in range(0, w):
        pixels[r, c] = np.average(im.getpixel((c, r)))

# view image
pil_img = Image.fromarray(pixels)
pil_img.show()

# flatten(matrix to vector)
pixels = pixels.reshape(-1)

# nomalize
pixels = pixels.astype(np.float32)
pixels /= 255.0

print(pixels)
print(pixels.shape, pixels.dtype)



