import argparse
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn import cluster

def build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Compress the input image \using clustering')
    parser.add_argument("--input-file",dest="input_file",required=True,help="Input image")
    parser.add_argument("--num-bits",dest="num_bits",required=False,type=int,help="Number of bits used to repressent each pixel")
    return parser

def compress_image(img,num_clusters):
    # 将输入的图片转换成(样本量,特征量)数组，以运行k-means聚类算法
    X = img.reshape((-1,1))       
    kmeans = cluster.KMeans(n_clusters=num_clusters,n_init=4,random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_
    # 为每个数据配置离它最近的中心点，并转变为图片的形状
    input_image_compressed = np.choose(labels,centroids).reshape(img.shape)
    return input_image_compressed 

def plot_image(img,title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img,cmap=plt.cm.get_cmap('gray'),vmin=vmin,vmax=vmax)

if __name__=='__main__':
    args = build_arg_parser().parser_args()
    input_file = args.input_file
    num_bits= args.num_bits
    if not 1<=num_bits<=8:
        raise TypeError('Number of bits should be between 1 and 8')
    num_clusters = no.power(2,num_bits)
    # 打印压缩率
    compression_rate = round(100*(8.0-args.num_bits)/8.0,2)
    print("\nThe size of the image will be reduced by a factor of",8.0/args.num_bits)
    print("\nCompression rate = "+str(compression_rate)+"%")
    input_image = misc.imread(input_file,True).astype(np.uint8)
    plot_image(input_image,'Original image')