import torch

def random_slelect_from_tesor(input_tensor):
    """
    generate random index by torch.randperm
    """
    output_tensor = input_tensor[torch.randperm(len(input_tensor))[:10], :]
    return output_tensor

def show_sce_heatmap(sce_weights):
    """
    show matrix heatmap
    """

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.rcParams['figure.dpi'] = 80  # 图形分辨率
    pd.options.display.notebook_repr_html = False  # 表格显示

    joint_list = [
        '1 - base of the spine',
        '2 - middle of the spine',
        '3 - neck',
        '4 - head',
        '5 - left shoulder',
        '6 - left elbow',
        '7 - left wrist',
        '8 - left hand',
        '9 - right shoulder',
        '10 - right elbow',
        '11 - right wrist',
        '12 - right hand',
        '13 - left hip',
        '14 - left knee',
        '15 - left ankle',
        '16 - left foot',
        '17 - right hip',
        '18 - right knee',
        '19 - right ankle',
        '20 - right foot',
        '21 - spine',
        '22 - tip of the left hand',
        '23 - left thumb',
        '24 - tip of the right hand',
        '25 - right thumb',
    ]

    np_arr = sce_weights.cpu().detach().numpy()
    mat = pd.DataFrame(np_arr, columns=joint_list)
    sns.heatmap(mat)
    plt.show()


def main():
    weihts_vis = torch.randn(64, 25).sigmoid()
    show_sce_heatmap(weihts_vis, channel_num=30)  # batch 1 c,v weights

if __name__ == "__main__":
    main()