import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def show_sce_heatmap(sce_weights, channel_num = 0):
    """
    show sce heatmap
    """

    # select batch 1 and channel joint dim
    sce_weights = sce_weights.squeeze(2)[:1].squeeze(0)  # 1, channel, joints

    # channel random select
    sce_weights_randn = sce_weights[torch.randperm(len(sce_weights))[:channel_num], :]

    # column joint for ntu-rgb-d 60 dataset
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

    # pytorch tensor to numpy
    np_arr = sce_weights_randn.cpu().detach().numpy()

    # make heatmap
    plt.rcParams['figure.dpi'] = 180  # 图形分辨率
    plt.rc('font', family='Times New Roman', size=10)
    pd.options.display.notebook_repr_html = False  # 表格显示
    mat = pd.DataFrame(np_arr, columns=joint_list)
    sns.heatmap(mat, vmin=0, vmax=1, center=0.5, cmap="coolwarm")
    plt.show()

def show_tce_heatmap(tce_weights, channel_num = 0, frame_num = 0):
    """
    show tce heatmap
    """

    # select batch 1 and channel joint dim
    tce_weights = tce_weights.squeeze(3)[:1].squeeze(0)  # 1, channel, joints

    # random select channel and frame
    channel_random_index = torch.randperm(len(tce_weights))[:channel_num].tolist()
    frame_random_index = torch.randperm(len(tce_weights[0][:]))[:frame_num].tolist()
    frame_random_index.sort()

    tce_weights_randn = tce_weights[channel_random_index, :]
    tce_weights_randn = tce_weights_randn[:, frame_random_index]

    # pytorch tensor to numpy
    np_arr = tce_weights_randn.cpu().detach().numpy()

    # make heatmap
    plt.rcParams['figure.dpi'] = 300  # 图形分辨率
    plt.rc('font', family='Times New Roman', size=8)
    pd.options.display.notebook_repr_html = False  # 表格显示
    mat = pd.DataFrame(np_arr, columns=frame_random_index)
    sns.heatmap(mat, vmin=0, vmax=1, center=0.5, cmap="coolwarm")
    # plt.savefig("./vis/tce_layer.png", bbox_inches = 'tight')
    plt.show()

def main():
    # weihts_vis = torch.randn(3, 64, 1, 25).sigmoid()
    # show_sce_heatmap(weihts_vis, channel_num=20)  # batch 1 c,v weights

    weihts_vis = torch.randn(3, 64, 300, 1).sigmoid()
    show_tce_heatmap(weihts_vis, channel_num=20, frame_num=50)  # batch 1 c,v weights

if __name__ == "__main__":
    main()