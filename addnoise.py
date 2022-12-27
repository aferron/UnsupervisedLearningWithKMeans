import random

class AddNoise(object):

  def __call__(self, sample):
    img, _ = sample
    row, col = img.shape
    total_pixels = len(row) * len(col)
    max = total_pixels / 6
    min = total_pixels / 24
     
    number_of_pixels = random.randint(min, max)
    for i in range(number_of_pixels):
       
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
         
    number_of_pixels = random.randint(min, max)
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
         
    return img
 