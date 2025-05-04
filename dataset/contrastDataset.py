from torchvision import datasets
import random
import torchvision.transforms.functional as TF
import math
import numpy as np

class contrastDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(contrastDataset, self).__init__(root)
        self.crop_transform = transform

    def __getitem__(self, idx):
        img, label = super(contrastDataset, self).__getitem__(idx)

        img1, img2 = random.choice([self.generate_global_local_view, self.generate_adjacent_view])(img)
        
        img1 = self.crop_transform(img1)
        img2 = self.crop_transform(img2)
        
        return img1, img2, label

    def generate_global_local_view(self, img):
        img1 = self.random_crop(img, adjacent = False)['img']
        img2 = self.random_crop(img1, adjacent = False)['img']

        return img1, img2

    def generate_adjacent_view(self, img):
        img1 = self.random_crop(img)
        img2 = self.random_crop(img, exclude_area=img1['crop_area'])

        return img1['img'], img2['img']

    def random_crop(self, img, adjacent = True, exclude_area=None):
        width, height = img.size
        
        min_side = min(width, height)
        if adjacent:
            crop_size = random.randint(int(min_side/2.1), int(min_side/1.5))
        else: 
            crop_size = random.randint(int(min_side/2.1), int(min_side/1.1))

        while True:
            left = random.randint(0, width - crop_size)
            top = random.randint(0, height - crop_size)
            right = left + crop_size
            bottom = top + crop_size

            # if exclude_area:
            #     if (exclude_area[0] < right and exclude_area[2] > left and
            #         exclude_area[1] < bottom and exclude_area[3] > top):
            #         continue
             
            # return crop img
            cropped_img = TF.crop(img, top, left, crop_size, crop_size)
            return {
                'img': cropped_img,
                'crop_area': (left, top, right, bottom)
            }

class maskDataset(contrastDataset):
    def __init__(self, root, transform, patch_size = 16):
        super(maskDataset, self).__init__(root, transform)
        self.psz = patch_size
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), (0.3, 1/0.3)))

    def __getitem__(self, idx):
        output = super(maskDataset, self).__getitem__(idx)
        masks = []
        
        for img in [output[0], output[1]]:
            
            H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            # H, W = 14, 14
            high = 0.3 * H * W

            mask = np.zeros((H, W), dtype=bool)
            mask_count = 0
            while mask_count < high:
                max_mask_patches = high - mask_count

                delta = 0
                for attempt in range(10):
                    low = (min(H, W) // 3) ** 2 
                    target_area = random.uniform(low, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < W and h < H:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)

                        num_masked = mask[top: top + h, left: left + w].sum()
                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if mask[i, j] == 0:
                                        mask[i, j] = 1
                                        delta += 1

                    if delta > 0:
                        break

                if delta == 0:
                    break
                else:
                    mask_count += delta
            
            masks.append(mask)

        return output + (masks,)