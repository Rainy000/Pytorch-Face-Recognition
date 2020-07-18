from __future__ import print_function
import os 


## dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


image_root = '/data/sharedata/wangyang_face/recognition/icartoon/images/'

class ImagePipline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(ImagePipline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=image_root)
        self.decode = ops.ImageDecoder(device='cpu', output_type=types.RGB)
    
    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return images, labels




import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_images(image_batch, batch_size):

    cols = 4
    rows = (batch_size + 1)//cols
    fig = plt.figure(figsize=(32,(32//clos)*rows))
    gs = gridspec.GridSpec(rows, cols)
    for j in range(cols*rows):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))




if __name__ == "__main__":

    batch_size = 16 
    pipe = ImagePipline(batch_size, 2, 0)
    pipe.build()
    pipe_out = pipe.run()
    print(pipe_out)
    images, labels = pipe_out
