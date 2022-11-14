import argparse
import pickle
import numpy as np
from tqdm import tqdm

def ensemble(ds, items):
    if 'ntu120' in ds:
        num_class=120
        if 'xsub' in ds:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in ds:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in ds:
        num_class=60
        if 'xsub' in ds:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in ds:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'UCLA' in arg.dataset:
        num_class=10
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    else:
        raise NotImplementedError

    ckpt_dirs, alphas = list(zip(*items))

    ckpts = []
    for ckpt_dir in ckpt_dirs:
        with open(ckpt_dir, 'rb') as f:
            ckpts.append(list(pickle.load(f).items()))

    right_num = total_num = right_num_5 = 0
    
    classnum = np.zeros(num_class)
    classacc = np.zeros(num_class)
    for i in tqdm(range(len(label))):
        l = label[i]
        r = np.zeros(num_class)
        for alpha, ckpt in zip(alphas, ckpts):
            _, r11 = ckpt[i]
            r += r11 * alpha

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
        
        classnum[int(l)] += 1
        classacc[int(l)] += int(r != int(l))
    
    classacc = 100 * classacc / classnum
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
#                         required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')

    parser.add_argument('--ckpts', nargs='+',
                        help='Directory containing "epoch{i}_test_score.pkl" for eval results')
    parser.add_argument('--joint-dir',
                        default=None,
                        help='Directory containing "epoch{i}_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        default=None,
                        help='Directory containing "epoch{i}_test_score.pkl" for bone eval results')
    parser.add_argument('--jbf-dir', 
                        default=None,
                        help='Directory containing "epoch{i}_test_score.pkl" for jbf eval results')
    parser.add_argument('--velocity-dir',
                        default=None,
                        help='Directory containing "epoch{i}_test_score.pkl" for velocity eval results')

    arg = parser.parse_args()

    
    item = []
    if arg.joint_dir is not None:
        item.append([arg.joint_dir, 1.0])
    if arg.bone_dir is not None:
        item.append([arg.bone_dir, 1.5])
    if arg.jbf_dir is not None:
        item.append([arg.jbf_dir, 2.0])
    if arg.velocity_dir is not None:
        item.append([arg.velocity_dir, 1.0])

    ensemble(arg.dataset, item)
