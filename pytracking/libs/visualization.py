import math
import matplotlib.pyplot as plt
import torch

def visualize_patches(im_patches, title=None, labels = True):
    num_img = im_patches.size()[0]

    if num_img == 1:
        fig, ax = plt.subplots()
        img = im_patches[0]
        img = torch.moveaxis((img-img.min())/(img.max()-img.min()), 0, 2)
        ax.set_axis_off()
        fig.imshow(img)
        fig.tight_layout()
        plt.show()
        return None

    num_row = math.floor(math.sqrt(num_img))
    num_cols = math.ceil(num_img/num_row)
    fig,axs = plt.subplots(num_row, num_cols)




    for row_num in range(num_row):
        for col_num in range(num_cols):
            index_num = (row_num)*num_cols + col_num
            axs[row_num, col_num].set_axis_off()
            if index_num < num_img:
                img = im_patches[index_num]
                #rescale and change to plottable format
                img = torch.moveaxis((img-img.min())/(img.max()-img.min()), 0, 2)
                #plot each image
                axs[row_num, col_num].imshow(img)
                if labels:
                    axs[row_num, col_num].set_title(f'img {index_num}')
    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    plt.show()
    return None

def draw_to_plot(fig, ax, num_row, num_cols, im_patches):
    num_img = im_patches.size()[0]
    for row_num in range(num_row):
        for col_num in range(num_cols):
            index_num = (row_num)*num_cols + col_num
            ax[row_num, col_num].set_axis_off()
            if index_num < num_img:
                img = im_patches[index_num]
                #rescale and change to plottable format
                img = torch.moveaxis((img-img.min())/(img.max()-img.min()), 0, 2)
                #plot each image
                ax[row_num, col_num].imshow(img)
                plt.draw()
                plt.pause(.000001)

def update_at_index(fig, ax, num_row, num_col, im_patches, replace_ind):
    #import pdb; pdb.set_trace()
    num_img = im_patches.size()[0]
    row_num = math.floor(replace_ind/num_col)
    col_num = replace_ind - row_num*num_col
    print(f"for {replace_ind} we have row {row_num} and col {col_num}")
    img = im_patches[replace_ind]
    img = torch.moveaxis((img-img.min())/(img.max()-img.min()), 0, 2)
    ax[row_num, col_num].imshow(img)
    plt.draw()
    plt.pause(.000001)

def update_at_index_from_img(fig, ax, num_row, num_col, image, replace_ind):
    num_img = 35
    row_num = math.ceil(num_img/num_row)
    col_num = replace_ind - row_num*num_col
    img = image
    img = torch.moveaxis((img-img.min())/(img.max()-img.min()), 0, 2)
    ax[row_num, col_num].imshow(img)
    plt.draw()
    plt.pause(.000001)
