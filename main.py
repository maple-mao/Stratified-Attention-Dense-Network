import tensorflow as tf
from ARDN import ARDN_net
from mode import *
import argparse

parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ('true')

## Model specification
parser.add_argument("--channel", type = int, default = 3)
parser.add_argument("--scale", type = int, default = 2)
parser.add_argument("--n_feats", type = int, default = 64)
parser.add_argument("--n_ARDB", type = int, default = 8) ##
parser.add_argument("--kernel_size", type = int, default = 3)
parser.add_argument("--ratio", type = int, default = 16)

parser.add_argument("--n_ARDG", type = int, default = 8)##

## Data specification 
parser.add_argument("--train_GT_path", type = str, default = "./DataSet/DIV2K_train_HR")
parser.add_argument("--train_LR_path", type = str, default = "./DataSet/DIV2K_train_LR_bicubic/X2")
parser.add_argument("--test_GT_path", type = str, default = "./DataSet/benchmark/Set5/HR")
parser.add_argument("--test_LR_path", type = str, default = "./DataSet/benchmark/Set5/LR_bicubic/X2")
parser.add_argument("--patch_size", type = int, default = 48)
parser.add_argument("--result_path", type = str, default = "./result")
parser.add_argument("--model_path", type = str, default = "./model")   # ./model/X2
parser.add_argument("--in_memory", type = str2bool, default = True)   # 如果训练数据足够小，设置为True


## Optimization
parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--max_step", type = int, default = 1 * 1e4)   # 1e6
parser.add_argument("--learning_rate", type = float, default = 1e-4)
parser.add_argument("--decay_step", type = int, default = 2 * 1e3)   # 2 * 1e5
parser.add_argument("--decay_rate", type = float, default = 0.5)
parser.add_argument("--test_with_train", type = str2bool, default = True)
parser.add_argument("--save_test_result", type = str2bool, default = True)

## Training or test specification
parser.add_argument("--mode", type = str, default = "train")
parser.add_argument("--fine_tuning", type = str2bool, default = False)
parser.add_argument("--load_tail_part", type = str2bool, default = True)
parser.add_argument("--log_freq", type = int, default = 1)   # 1e3
parser.add_argument("--model_save_freq", type = int, default = 50)
parser.add_argument("--pre_trained_model", type = str, default = "./model/ARDN_X2_64_8_8-")
parser.add_argument("--self_ensemble", type = str2bool, default = False)
parser.add_argument("--chop_forward", type = str2bool, default = False)
parser.add_argument("--chop_shave", type = int, default = 10)
parser.add_argument("--chop_size", type = int, default = 4 * 1e4)
parser.add_argument("--test_batch", type = int, default = 1)
parser.add_argument("--test_set", type = str, default = 'Set5')


args = parser.parse_args()

model = ARDN_net(args)
model.build_graph()

print("Build model!")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.85
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())

if args.mode == 'train':
    train(args, model, sess)
    
elif args.mode == 'test':
    test(args, model, sess)
        
elif args.mode == 'test_only':
    test_only(args, model, sess)


# 计算参数量
def count1():
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        print("总参数量：", total_parameters)
count1()
