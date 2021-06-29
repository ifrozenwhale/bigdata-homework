from enum import Enum
import random
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import io


class State(Enum):
    I = 'Infective'
    S = 'Susceptible'
    R = 'Removal'


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


class SirModel:

    def __init__(self, N, p, k=5):
        self.iter = 0
        self.N = N
        self.p = p
        self.network = nx.random_graphs.watts_strogatz_graph(N, k, p)
        self.infective_nodes = []
        self.removal_nodes = []

        self.S_list = []
        self.I_list = []
        self.R_list = []
        self.pos = None
        self.colors = {"Removal": '#228B22', "Infective": '#B22222', "Susceptible": '#808080'}
        # 将初始的加入
        self.init_source(5)

    def show(self, show=True):
        plt.cla()
        infective_nums = len(self.infective_nodes)
        removal_nums = len(self.removal_nodes)
        states = nx.get_node_attributes(self.network, 'state')
        node_colors = [self.colors[states[i].value] for i in range(self.N)]
        nx.draw_networkx_nodes(self.network, node_color=node_colors, pos=self.pos)
        nx.draw_networkx_edges(self.network, pos=self.pos, alpha=0.4, edge_color='#000000', style='dashed')

        plt.pause(0.005)
        if show: plt.show()

    def update_SIR_nums(self, s, i, r):
        self.S_list.append(s)
        self.I_list.append(i)
        self.R_list.append(r)

    def init_source(self, n, S_rate=1):
        """
        度中心性最高的5个节点作为感染源，初始化其状态为I。
        易感者比例
        """

        for i in range(int(self.N * S_rate)):  # 所有人默认为易感染
            self.network.nodes[i]['state'] = State.S

        degree_cent = nx.algorithms.centrality.degree_centrality(self.network)
        degree_cent = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
        for i in range(n):
            node = degree_cent[i][0]
            self.network.nodes[node]['state'] = State.I
            self.infective_nodes.append(node)
        self.update_SIR_nums(int(self.N * S_rate), n, 0)
        self.pos = nx.random_layout(self.network)

    def spread(self, alpha=0.7, beta=0.4, max_iter=None, show=False):
        """
        假设康复的概率为alpha，被感染的概率为beta ，病毒传播一次过程即，对每个节点生成一个随机概率p，小于康复概率alpha
        则此节点状态由I转为R；对I状态的每个邻居节点生成一个随机概率，小于感染概率beta
        则此节点状态由S转为I。通过不断迭代，得到最终状态，所有人状态都为R。将每轮迭代后的网络可视化，
        可得到病毒传播的详细过程。
        """
        if show:
            plt.ion()
        cur_iter = 0
        img_data_list = []
        while len(self.infective_nodes) > 0:
            cur_iter += 1
            if max_iter and cur_iter > max_iter:
                break

            new_S_num = self.S_list[-1]
            new_I_num = self.I_list[-1]
            new_R_num = self.R_list[-1]
            for i in self.infective_nodes:

                if random.random() < alpha and self.network.nodes[i]['state'] == State.I:
                    self.removal_nodes.append(i)
                    self.infective_nodes.remove(i)

                    self.network.nodes[i]['state'] = State.R
                    new_R_num += 1
                    new_I_num -= 1

            tmp_infective_nodes = nx.Graph()
            tmp_infective_nodes.add_nodes_from(self.infective_nodes)
            new_infective_nodes = []
            for node in tmp_infective_nodes.nodes:
                for neighbor in self.network.neighbors(node):
                    if random.random() < beta and self.network.nodes[neighbor]['state'] == State.S:
                        self.network.nodes[neighbor]['state'] = State.I
                        new_I_num += 1
                        new_S_num -= 1
                        new_infective_nodes.append(neighbor)
            for node in new_infective_nodes:
                if node not in self.infective_nodes:
                    self.infective_nodes.append(node)
            self.update_SIR_nums(new_S_num, new_I_num, new_R_num)

            if show:
                self.show(show=False)
                imgdata = io.BytesIO()
                plt.savefig(imgdata, format='png')
                # imgdata.seek(0)
                img_data_list.append(imgdata)
            # print(img_data_list)
        if show:
            plt.ioff()
            self.show(show=True)
            img_list = []
            for img in img_data_list:
                img_list.append(img.getvalue())
            gif_name = 'results/test.gif'
            duration = 0.2
            print(len(img_list))
            create_gif(img_list, gif_name, duration)

        self.iter = cur_iter

    def show_SIR_detail(self,p=0.2):
        x = list(range(1, len(self.S_list) + 1))
        plt.plot(x, self.S_list, self.colors[State.S.value], label=State.S.value)
        plt.plot(x, self.I_list, self.colors[State.I.value], label=State.I.value)
        plt.plot(x, self.R_list, self.colors[State.R.value], label=State.R.value)
        plt.legend()
        plt.xlabel('iter')
        plt.ylabel('count')
        plt.title(f'paramater p={round(p,4)}')
        plt.savefig(f"./results/sir_curve_{round(p,4)}.png",dpi=400)
        plt.close()

        # plt.show()

import numpy as np

def test():
    p_arr = np.arange(0.005, 0.4, 0.02)
    for p in p_arr:
        sir_p = SirModel(300, p)
        sir_p.spread(alpha=0.4, beta=0.6, show=False)
        sir_p.show_SIR_detail(p=p)


if __name__ == '__main__':
    sir = SirModel(150, 0.2)
    sir.spread(alpha=0.4, beta=0.6, show=True)
    sir.show_SIR_detail(p=0.2)

    # test for different k
    test()
