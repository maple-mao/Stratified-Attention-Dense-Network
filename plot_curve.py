import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

# path0 = './ARDN_X2_train_log.txt'

parser = argparse.ArgumentParser()

parser.add_argument('--txt_path', type=str, default='./ARDN_X2_train_log.txt,./DN_X2_train_log.txt',
                    help='输入的txt文本路径')
parser.add_argument('--color', type=str, default='red,blue')
parser.add_argument('--label', type=str, default='x2_with ADM,x2_no ADM')
parser.add_argument('--y_min', type=float, default='32.0')
parser.add_argument('--y_max', type=float, default='38.0')
parser.add_argument('--yt', type=float, default='7')
parser.add_argument('--save_name', type=str, default='train_log')
parser.add_argument('--dpi', type=int, default='600')

args = parser.parse_args()


def read_txt(path):
    X, Y = [], []
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines[1:]:
        content = line.split()
        content0 = content[0].split('-')[0]
        X.append(float(content0))
        Y.append(float(content[-1]))
    f.close()

    return X, Y


def plot(args):
    plt.figure(figsize=(7, 5), dpi=args.dpi)
    ax = plt.gca()
    ax.tick_params(direction='in', top=True, right=True)

    color = args.color.split(',')
    label = args.label.split(',')
    j = 0
    for i in args.txt_path.split(','):
        X, Y = read_txt(i)
        ax.plot(X, Y, color=color[j], label=label, linewidth=0.5, linestyle='-')
        j += 1

    X3 = np.linspace(0, 10000, 10000)
    Y3 = [36.66] * 10000
    ax.plot(X3, Y3, color='black', label='SRCNN', linewidth=0.5, linestyle='--')

    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 14,
             }

    xt = np.linspace(0, 10000, 6)
    yt = np.linspace(args.y_min, args.y_max, args.yt)
    plt.xticks(xt, fontproperties='Times New Roman', fontsize=14)
    plt.yticks(yt, fontproperties='Times New Roman', fontsize=14)

    plt.xlabel('Iterations', font1)
    plt.ylabel('PSNR(dB)', font1)
    plt.xlim(0, 10000)
    plt.ylim(args.y_min, args.y_max)
    plt.legend(loc='lower right', prop=font1)
    plt.savefig('%s.tif' % args.save_name, dpi=args.dpi, bbox_inches="tight", format='tif')
    print('图片已保存' + os.getcwd() + '\%s.png' % args.save_name)



plot(args)
