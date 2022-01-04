import numpy as np

config = {
    "debug": False
}


def calc_res(shape):
    """
    Implementation based on
    https://github.com/eps696/stylegan2ada
    """
    base0 = 2 ** int(np.log2(shape[0]))
    base1 = 2 ** int(np.log2(shape[1]))
    base = min(base0, base1)
    min_res = min(shape[0], shape[1])

    def int_log2(xs, base):
        return [x * 2 ** (2 - int(np.log2(base))) % 1 == 0 for x in xs]

    if min_res != base or max(*shape) / min(*shape) >= 2:
        if np.log2(base) < 10 and all(int_log2(shape, base * 2)):
            base = base * 2

    return base  # , [shape[0]/base, shape[1]/base]


def calc_init_res(shape, resolution=None, conv_base_index=2):
    """
    Implementation based on
    https://github.com/eps696/stylegan2ada
    """
    if len(shape) == 1:
        shape = [shape[0], shape[0], 1]
    elif len(shape) == 2:
        shape = [*shape, 1]
    size = shape[:2] if shape[2] < min(*shape[:2]) else shape[1:]  # fewer colors than pixels
    if resolution is None:
        resolution = calc_res(size)
    res_log2 = int(np.log2(resolution))
    init_res = [int(s * 2 ** (conv_base_index - res_log2)) for s in size]
    return init_res, resolution, res_log2


def calc_acceptable_res_combination(conv_base_index=2):
    """
    Calculate acceptable resolution of training images for modded implementation of SG2ADA
    """
    acceptable_res = []
    for i in range(1, 257):  # 1 to 256
        for j in range(5, 9):  # 5 to 8
            if i % (2 ** j) == 0 and i not in acceptable_res:
                acceptable_res.append(i)
                break
    from itertools import product

    acceptable_res_combination = []
    for pair in product(acceptable_res, acceptable_res):
        init_res, _, res_log2 = calc_init_res(pair, conv_base_index=conv_base_index)
        if init_res[0] * 2 ** (res_log2 - conv_base_index) == pair[0] and init_res[1] * 2 ** (
                res_log2 - conv_base_index) == pair[1]:
            if config["debug"]:
                print("Target res", pair, "is ok!", f"From init res {init_res}",
                      f"2x up-sampling for {res_log2 - conv_base_index} times.")
            acceptable_res_combination.append(pair)
        else:
            if config["debug"]:
                print("Target res", pair, "is ng!")

    if config["debug"]:
        print("Result:", acceptable_res_combination)
    return acceptable_res_combination


def find_min_acceptable_res_combination(shape, acceptable_res_combination=None, conv_base_index=2):
    if acceptable_res_combination is None:
        acceptable_res_combination = calc_acceptable_res_combination(conv_base_index)

    for r1, r2 in acceptable_res_combination:
        if r1 < shape[0] or r2 < shape[1]:
            continue
        return r1, r2
    return None


if __name__ == "__main__":
    config["debug"] = True
    conv_base_index = 2
    shape = (94, 151)
    base_shape = find_min_acceptable_res_combination(shape, conv_base_index=conv_base_index)
    print(base_shape, calc_init_res(base_shape, conv_base_index=conv_base_index))
