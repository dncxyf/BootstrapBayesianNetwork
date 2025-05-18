#导入需要的包
import os
import numpy as np
import pandas as pd
import bnlearn as bn
from sklearn.preprocessing import KBinsDiscretizer

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

from pyvis.network import Network

#定义bnlearn函数
def sample_bnlearn(data_name,sheet_name,n_bin = 3,sample_n=200,sample_frac = 0.9):
    #导入数据
    #os.chdir('data/input')
    dir = 'data/input/'
    data = pd.read_excel(dir+data_name+'.xlsx',sheet_name=sheet_name).dropna(how='any')
    data_n = data.shape[1]
    #离散化
    discretizer = KBinsDiscretizer(n_bins=n_bin, encode='ordinal', strategy='quantile')
    for column in data.columns:
        data[column] = discretizer.fit_transform(data[[column]])
    #抽样,对每次抽样构建贝叶斯，计算adjmat，储存
    mat = []
    for i in range(sample_n):
        sample_x = data.sample(frac=sample_frac)
        mat_x = bn.structure_learning.fit(sample_x, methodtype='hc', scoretype='bic')
        mat.append(mat_x['adjmat'])
    #储存写入h5
    with pd.HDFStore('data_maj.h5') as store:
        for i in range(sample_n):
            name = str(i)
            store[name] = mat[i]
    #初始化累加矩阵
    mat_default = mat[0].copy(deep=True)
    mat_default.iloc[:,:] = 0
    #累加所有贝叶斯结果
    structure0 = mat_default.copy(deep=True)#有向
    structure1 = mat_default.copy(deep=True)#无向
    for i in range(sample_n):
        for x in range(data_n):
            for y in range(data_n):
                if mat[i].iloc[x,y] == True:
                    structure0.iloc[x,y] += 1
    for i in range(data_n):
        for j in range(i,data_n):
            structure1.iloc[i,j] = structure0.iloc[i,j] + structure0.iloc[j,i]
    #写入excel
    df=pd.DataFrame()
    df.to_excel('data/output/bn_'+data_name+'_'+sheet_name+'.xlsx')
    with pd.ExcelWriter('data/output/bn_'+data_name+'_'+sheet_name+'.xlsx', mode='a', if_sheet_exists='overlay') as writer:
        structure0.to_excel(writer, sheet_name='有向')
        structure1.to_excel(writer, sheet_name='无向')
    #50%fig
    g = Network()
    net_id = [i+1 for i in range(data_n)]
    node_list = [i for i in data.columns]
    node_color = ['#BBFFFF' for i in range(data_n-1)]
    node_color.append('#FF0000')
    g.add_nodes(net_id,label=node_list,color=node_color)
    for x in range(data_n):
        for y in range(data_n):
            if structure1.iloc[x,y] >= sample_n/2:
                g.add_edge(x+1,y+1)
    g.save_graph('data/output/fig/bn_fig_'+data_name+'_'+sheet_name+'.html')
    return data,structure0,structure1

def bn_fig(data_name,sheet_name,structure,sample_n=200,frac = 0.5):
    #os.chdir('data')
    dir = 'data/input/'
    data = pd.read_excel(dir+data_name+'.xlsx',sheet_name=sheet_name)
    data_n = data.shape[1]
    g = Network()
    net_id = [i+1 for i in range(data_n)]
    node_list = [i for i in data.columns]
    node_color = ['#BBFFFF' for i in range(data_n-1)]
    node_color.append('#FF0000')
    g.add_nodes(net_id,label=node_list,color=node_color)
    for x in range(data_n):
        for y in range(data_n):
            if structure.iloc[x,y] >= sample_n*frac:
                g.add_edge(x+1,y+1)
    g.save_graph('data/output/fig/bn_fig_'+data_name+'_'+sheet_name+'.html')