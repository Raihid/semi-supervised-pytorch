import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from shutil import rmtree
from tqdm import trange
from PIL import Image

def sample_from_classes(prefix, model, im_shape, z_dim, classes_num):
    im_h, im_w, im_c = im_shape
    canvas = np.empty((im_h * classes_num, im_w * 10, im_c))

    z_samples = torch.randn(classes_num, 10, z_dim).cuda()
    for row_idx in range(classes_num):

        one_hot_class = [0.] * classes_num
        one_hot_class[row_idx] = 1.
        one_hot_class = [one_hot_class] * 10
        one_hot_class = torch.tensor(one_hot_class, dtype=torch.float).cuda()
        
        generated = model.sample(
            z_samples[row_idx],
            one_hot_class
        ).cpu().detach().numpy()
        
        

        for col_idx, sample in enumerate(generated):
            sample[sample < 0] = 0
            sample[sample > 1] = 1
            sample = sample.reshape(im_c, im_h, im_w)
            
            if len(im_shape) == 3:
                sample = sample.transpose(1, 2, 0)

            canvas[row_idx * im_h:(row_idx + 1) * im_h,
                   col_idx * im_w:(col_idx + 1) * im_w] = sample

    plt.xticks([])
    plt.yticks([])

    # fig.tight_layout(pad=0)
    plt.imshow(canvas.squeeze(), cmap="gray")
    plt.savefig("samples/" + prefix + "_sampling.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    

def interpolation(prefix, model, dataset, im_shape):
    
    out_z=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,1]]
    
    n_iterations = 8
    nx = n_iterations + 2
    ny = 10

    im_h, im_w, im_c = im_shape

    canvas = np.empty((im_h * ny, im_w * (nx), im_c))

    for y_idx in range(ny):
        x_from, y_from = dataset[out_z[y_idx][0]]
        x_to, y_to = dataset[out_z[y_idx][1]]
        x_sample = torch.stack([x_from, x_to]).cuda()
        
        _, z, _ = model.encode(x_sample)
        z = z.cpu().detach().numpy()
        
        #self.model.encode(x_sample)
        z_from = z[0]
        z_to = z[1]

        z_inter = np.array(z_from)
        y_inter = np.array(y_from)
        for iteration in np.arange(1, n_iterations):
            z_step = z_from + (z_to - z_from) * iteration / n_iterations
            y_step = y_from + (y_to - y_from) * iteration / n_iterations
            
            z_inter = np.vstack([z_inter, z_step])
            y_inter = np.vstack([y_inter, y_step])

        z_inter = torch.tensor(z_inter, dtype=torch.float).cuda()
        y_inter = torch.tensor(y_inter, dtype=torch.float).cuda()
        reconstructed = model.sample(z_inter, y_inter).cpu().detach().numpy()

        reconstructed = reconstructed.reshape((reconstructed.shape[0], -1))
        reconstructed = np.concatenate(
            (x_from.reshape(1, -1), reconstructed, x_to.reshape(1, -1)), axis=0)
        
        for x_idx, sample in enumerate(reconstructed):

            sample = sample.reshape(im_c, im_h, im_w)
            if len(im_shape) == 3:
                sample = sample.transpose(1, 2, 0)

            # d_plot = 1 / (1 + np.exp(-d_plot))
            sample[sample<0] = 0
            sample[sample>1] = 1

            canvas[(ny - y_idx - 1) * im_h:(ny - y_idx) * im_h,
                   x_idx * im_w:(x_idx + 1) * im_w] = sample.reshape(im_h, im_w, im_c)


    canvas = canvas.squeeze() # (28, 28, 1) => (28, 28)
    
    plt.xticks([])
    plt.yticks([])

    plt.imshow(canvas.squeeze(), cmap="gray")
    
    filename = "samples/" + prefix + "_interpolation.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def cyclic_interpolation(prefix, model, dataset, im_shape, classes_num, labels_names):
    im_h, im_w, im_c = im_shape
    
    samples_num = 10
    interpolation_steps = 10
    padding = 6

    interpolating_samples = list(range(samples_num))
    data_x = torch.stack([dataset[idx][0] for idx in interpolating_samples]).cuda()
    data_y = torch.stack([dataset[idx][1] for idx in interpolating_samples]).cuda()

    probs = model.classify(data_x).cpu().detach().numpy()
    _, z, _ = model.encode(data_x)

    for idx in interpolating_samples:
        fig = plt.figure(figsize=(interpolation_steps, samples_num))

        label = np.argmax(data_y[idx].cpu())
        canvas = np.ones(((im_h + padding) * classes_num, im_w * interpolation_steps, im_c))

        for direction in range(classes_num):

            y_inputs = []
            for step_size in np.linspace(0., 1., num=interpolation_steps):
                y = [0] * classes_num
                y[direction] = step_size
                y[label] += 1 - step_size
                y_inputs += [y]
                
            sample_probs = np.array(y_inputs)
            y_inputs = torch.tensor(y_inputs, dtype=torch.float)
                
              
            outputs = model.sample(z[idx].unsqueeze(0).repeat(10, 1), y_inputs.cuda())
            outputs = outputs.cpu().detach().numpy()
            

            for output_idx, output in enumerate(outputs):
                start_h = (im_h + padding) * direction
                
                caption = "{} {}%".format(
                    labels_names[sample_probs[output_idx].argmax()],
                    round(float(sample_probs[output_idx].max() * 100), 2))
                plt.text(im_w * output_idx, start_h - 2, caption, fontsize=6)
                
                output = output.reshape(im_c, im_h, im_w)
                if len(im_shape) == 3:
                    output = output.transpose(1, 2, 0)


                canvas[start_h:start_h + im_h,
                       im_w * output_idx:im_w * (output_idx + 1)] = output

        plt.xticks([])
        plt.yticks([])
        plt.axes().set_aspect('equal')
        plt.axis("off")
        plt.imshow(canvas.squeeze(), origin="upper", cmap="gray")

        # caption = "Label: {}    Class given by the model: {} with prob {}".format(
        #     real_label, pred_label, round(float(probs[idx].max()), 2))
        # plt.text(0, 0, caption)
        
        filename = "samples/" + prefix + "_cyclic_interpolation_{}.png".format(idx)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()
        #     inter_type = "direct" if direct else "cyclic"
        #     filename = "results/{}/{}_interpolation_{}_{}.png".format(
        #         model.name, inter_type, str(epoch).zfill(3), idx)
        # plt.close(fig)

def save_samples(prefix, model, im_shape, n_samples, classes_num, z_dim):
    samples_dir = "samples/{}_samples/".format(prefix)
    if os.path.isdir(samples_dir):
        rmtree(samples_dir)
    os.makedirs(samples_dir)


    im_h, im_w, im_c = im_shape

    samples_per_class = np.random.multinomial(
        n_samples, [1 / classes_num] * classes_num)

    for class_idx in trange(classes_num):
        y = [0] * classes_num
        y[class_idx] = 1. 

        y = torch.tensor([y], dtype=torch.float).cuda()
        y = y.repeat(samples_per_class[class_idx], 1)
        z = torch.randn(samples_per_class[class_idx], z_dim).cuda()

        outputs = model.sample(z, y).cpu().detach().numpy()
        outputs = outputs.reshape([-1, im_c, im_h, im_w])
        outputs = outputs.transpose(0, 2, 3, 1)

        outputs = (outputs * 255).astype("uint8").reshape([-1] + im_shape)

        for idx, pixels in enumerate(outputs):
            filename = "{}/c{}_{}.png".format(
                samples_dir, class_idx, str(idx).zfill(5))
            if im_c == 1:
                img = Image.fromarray(pixels.squeeze(), "L")
            else:
                img = Image.fromarray(pixels, "RGB")
            img.save(filename)

