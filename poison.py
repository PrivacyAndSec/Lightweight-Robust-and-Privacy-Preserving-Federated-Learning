import copy
import random

def get_poison(images, labels, arg, edge_id, round_id, adversarial_index, malicious):
    poison_count = 0
    new_images = images
    new_labels = labels
    if malicious == True:
        random.seed(edge_id*100 + 2 *round_id)
        if arg.attack == "backdoor":
            for index in range(0, len(images)):
                new_labels[index] = arg.target
                new_images[index] = add_pixel_pattern(images[index], adversarial_index)
                poison_count += 1

        elif arg.attack == "target":
            for index in range(0, len(images)):
                if new_labels[index] == arg.source:
                    new_labels[index] = arg.target
                    new_images[index] = images[index]
                    poison_count += 1
        elif arg.attack == "untarget":

            for index in range(0, len(images)):
                new_labels[index] = random.randint(0,9)
                new_images[index] = images[index]
                poison_count += 1

    return new_images, new_labels, poison_count


def add_pixel_pattern(ori_image, adversarial_index):
    poison_patterns_template = [
        [[0, 0], [0, 1], [0, 2], [0, 3]],
        [[0, 6], [0, 7], [0, 8], [0, 9]],
        [[3, 0], [3, 1], [3, 2], [3, 3]],
        [[3, 6], [3, 7], [3, 8], [3, 9]],
    ]

    image = copy.deepcopy(ori_image)
    if adversarial_index != -1:
        poison_patterns = poison_patterns_template[adversarial_index % 4]
    else:
        poison_patterns = []
        for pattern in poison_patterns_template:
            poison_patterns.extend(pattern)

    channel_num = image.shape[0]  # [C, H, W]
    for pattern_idx in range(len(poison_patterns)):
        pos = poison_patterns[pattern_idx]
        for channel_idx in range(channel_num):
            image[channel_idx][pos[0]][pos[1]] = 1

    return image