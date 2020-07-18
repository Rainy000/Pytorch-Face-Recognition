import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image
import os
import os.path
import jpeg4py as jpeg
from augmentation import *
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def search_dir(dir):
    ## transform "~,~user" to /home/user 
    dir = os.path.expanduser(dir)
    sub_dir = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    sub_dir.sort()
    imglist_dict = {}
    for target in sub_dir:
        d = os.path.join(dir,target)
        if not os.path.isdir(d):
            continue
        subdir_imglist = []
        for root, dirname, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    #path = os.path.join(root,fname)
                    subdir_imglist.append(fname)
        imglist_dict[target] = subdir_imglist
    return imglist_dict

                    

def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

###################
def find_classes(dir):

    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def jpeg4py_loader(path):
    with open(path, 'rb')as f:
        img = jpeg.JPEG(f).decode()
        return Image.fromarray(img)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
        #return jpeg4py_loader(path)

def tuple_reader(filelist):
    '''
        return parameter:
            imglist: list type, each element is a tuple (imgpath, label1, label2)
    '''
    assert(os.path.exists(filelist)),"{} file not exists".format(filelist)
    imglist = []
    with open(filelist, 'r')as f:
        for line in f.readlines():
            elem_list = line.strip().split()
            item = (elem_list[0], int(elem_list[1]), int(elem_list[2]))
            imglist.append(item)
    f.close()
    return imglist

def path_reader(filelist):
    '''
        return parameter:
            imglist: list type, each element is a str (image path)
    '''
    assert os.path.exists(filelist), "{} file not exists".format(filelist)
    imglist = []
    with open(filelist, 'r') as f:
        for line in f.readlines():
            elem_list = line.strip().split()
            imglist.append(elem_list[0])
    f.close()
    return imglist

def default_filelist_reader(filelist):
    return path_reader(filelist)

def generate_odd(min=5, max=9):
    num = random.randint(min,max)
    if num%2==1:
        return num
    else:
        return min


class folder_dataloader(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, shuffle=True, transform=None, loader=default_loader):
        classes, class_to_idx = find_classes(root)
        self.imgs = make_dataset(root, class_to_idx)
        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.transform = transform
        self.loader = loader
        self.length = len(self.imgs)
        if shuffle:
            random.shuffle(self.imgs)
        

    def __getitem__(self, index):

        path,target = self.imgs[index]
        img = self.loader(path)
        #blur_transforms = [guassian_blur(mask_size=generate_odd(5,9)),median_blur(mask_size=generate_odd(5,9)),mean_bur(mask_size=generate_odd(5,9)),
        #                      motion_blur(length=random.randint(10,25),angle=random.randint(10,30))]
        #blur_transform = blur_transforms[random.randint(0,len(blur_transforms)-1)]
        #img = blur_transform(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.length

## to do
class imglist_dataloader(data.Dataset):
    """
    """

    def __init__(self, imglist, shuffle=True, transform=None, loader=default_filelist_reader):
        self.loader = loader
        self.imglists = self.loader(imglist)
        self.transform = transform
        self.length = len(self.imgs)
        if shuffle:
            random.shuffle(self.imgs)
        

    def __getitem__(self, index):

        path,target = self.imglists[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.length



class test_dataloader(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, save_path, dir_name, imglist, transform=None, target_transform=None, loader=default_loader):

        self.root = root
        self.save_path = save_path
        self.dir_name = dir_name
        self.imglist = []
        with open(imglist) as f:
            for line in f.readlines():
                line = line.strip().split()
                self.imglist.append(line[0])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        ## save *.lst file
        f = open(os.path.join(self.save_path,self.dir_name+".lst"),'w')
        for line in self.imglist:
            f.write('%s\n'%(line))
        f.write('%s'%("FIN"))
        f.close()

    def __getitem__(self, index):

        path = self.imglist[index // 2]
        target = -1  # todo: just for test
        img = self.loader(os.path.join(self.root, self.dir_name, path))
        if index % 2 == 1:
            img = F.hflip(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imglist) * 2
