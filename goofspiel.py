import os.path

import matplotlib.pyplot as plt
import numpy as np
import pickle
import arithmetics
from graph import act_t, Node, NodePtr

np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

class Goofspiel(object):
    def __init__(self):
        # non-iterables
        self.CARD_NUM: int = 4
        self.upcard: list[int] = [1, 1, 1, 1]
        self.weight: dict[str, list[float]] = {"A": [5.00, 1.33, 2.71, 1.80], "B": [4.12, 6.28, 3.33, 1.92]}
        self.root: NodePtr = NodePtr(0)
        self.bfs: list[list[NodePtr]] = []

        # instants
        self.A: dict[NodePtr, np.ndarray] = {}  # self.A[info_set][i, j]
        self.B: dict[NodePtr, np.ndarray] = {}  # self.B[info_set][i, j]

        # records
        self.reg: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # reg[player][info_set][act, t]
        self.sig: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # sig[player][info_set][act, t]
        self.opt: dict[str, dict[NodePtr, np.ndarray]] = {"A": {}, "B": {}}  # gain[player][info_set][act, t]
        self.cfv: dict[NodePtr, np.ndarray] = {}  # cfv[node][player, t]
        self.exp: dict[NodePtr, np.ndarray] = {}  # exp[node][player, t]

    def tree_init(self):
        self.root = Node.new()
        self.root.o.parent = NodePtr(0)
        self.root.o.access = np.array([0, 0])
        self.root.o.actA = np.arange(self.CARD_NUM) + 1
        self.root.o.actB = np.arange(self.CARD_NUM) + 1

        self.bfs = []
        parent = [self.root]
        child = []
        while True:
            self.bfs.append(parent)
            for n in parent:
                n.o.child = np.resize(n.o.child, [len(n.o.actA), len(n.o.actB)])
                for i, a in enumerate(n.o.actA):
                    for j, b in enumerate(n.o.actB):
                        ch = Node.new()
                        n.o.child[i, j] = ch
                        ch.o.parent = n
                        ch.o.access = np.array([a, b])
                        ch.o.actA = n.o.actA[n.o.actA != a]
                        ch.o.actB = n.o.actB[n.o.actB != b]
                        child.append(ch)
            if len(child) == 0:
                break
            else:
                parent = child
                child = []

    def sigma_init(self, T: int = 200):
        for generation in self.bfs[0:-2]:
            for n in generation:
                self.reg["A"][n] = np.empty([len(n.o.actA), T], dtype=float)
                self.reg["B"][n] = np.empty([len(n.o.actB), T], dtype=float)
                self.sig["A"][n] = np.empty([len(n.o.actA), T], dtype=float)
                self.sig["B"][n] = np.empty([len(n.o.actB), T], dtype=float)
        for generation in self.bfs[0:-2]:
            for n in generation:
                self.reg["A"][n][:, 0] = arithmetics.rand_f(1, len(n.o.actA) - 1)[0] * 5
                self.reg["B"][n][:, 0] = arithmetics.rand_f(1, len(n.o.actB) - 1)[0] * 5
                self.sig["A"][n][:, 0] = self.reg["A"][n][:, 0] / self.reg["A"][n][:, 0].sum()
                self.sig["B"][n][:, 0] = self.reg["B"][n][:, 0] / self.reg["B"][n][:, 0].sum()

    def cfv_init(self, T: int = 200):
        for generation in self.bfs:
            for n in generation:
                self.cfv[n] = np.empty([2, T], dtype=float)
        for generation in self.bfs[0:-2]:
            for n in generation:
                self.A[n] = np.empty([len(n.o.actA), len(n.o.actB)], dtype=float)
                self.B[n] = np.empty([len(n.o.actA), len(n.o.actB)], dtype=float)
                self.opt["A"][n] = np.empty([len(n.o.actA), T], dtype=float)
                self.opt["B"][n] = np.empty([len(n.o.actB), T], dtype=float)
                self.exp[n] = np.empty([2, T], dtype=float)

        # leaves:
        for n in self.bfs[-1]:
            h = n.o.h()
            u, v = 0, 0
            for k, acts in enumerate(h):
                a, b = acts
                if a > b:
                    u += self.upcard[k] * self.weight["A"][k]
                    v -= self.upcard[k] * self.weight["B"][k]
                elif a < b:
                    u -= self.upcard[k] * self.weight["A"][k]
                    v += self.upcard[k] * self.weight["B"][k]
            self.cfv[n][:, 0] = np.array([u, v], dtype=float)
        for n in self.bfs[-2]:
            self.cfv[n][:, 0] = self.cfv[n.o.get_child(0, 0)][:, 0].copy()

    def cfv_refresh(self, t: int):
        for generation in self.bfs[-1:-3:-1]:
            for n in generation:
                self.cfv[n][:, t] = self.cfv[n][:, 0]
        for generation in self.bfs[-3::-1]:
            for n in generation:
                for i, a in enumerate(n.o.actA):
                    for j, b in enumerate(n.o.actB):
                        self.A[n][i, j], self.B[n][i, j] = self.cfv[n.o.get_child(i, j)][:, t]
                self.opt["A"][n][:, t] = self.A[n] @ self.sig["B"][n][:, t]
                self.opt["B"][n][:, t] = self.B[n].T @ self.sig["A"][n][:, t]
                self.cfv[n][0, t] = self.opt["A"][n][:, t].T @ self.sig["A"][n][:, t]
                self.cfv[n][1, t] = self.opt["B"][n][:, t].T @ self.sig["B"][n][:, t]
                self.exp[n][0, t] = self.opt["A"][n][:, t].max() - self.cfv[n][0, t]
                self.exp[n][1, t] = self.opt["B"][n][:, t].max() - self.cfv[n][1, t]
    # New idea: Ditch iterables, rely solely on records! Because this can tackle overwrites! 2022-1121-2330
    # Problem: leaves, trivial nodes? -> Anyway, don't care about compression!

    def sigma_update(self, t: int):
        for generation in self.bfs[0:-2]:
            for n in generation:
                x = self.sig["A"][n][:, t - 1]
                y = self.sig["B"][n][:, t - 1]
                u = self.opt["A"][n][:, t - 1]
                v = self.opt["B"][n][:, t - 1]

                r = {"A": u - u.T @ x, "B": v - v.T @ y}
                r["A"][r["A"] < 0] *= 0.05
                r["B"][r["B"] < 0] *= 0.05
                self.reg["A"][n][:, t] = self.reg["A"][n][:, t - 1] + r["A"]
                self.reg["B"][n][:, t] = self.reg["B"][n][:, t - 1] + r["B"]
                self.reg["A"][n][self.reg["A"][n][:, t] < 0, t] = 0
                self.reg["B"][n][self.reg["B"][n][:, t] < 0, t] = 0
                self.sig["A"][n][:, t] = self.reg["A"][n][:, t] / self.reg["A"][n][:, t].sum()
                self.sig["B"][n][:, t] = self.reg["B"][n][:, t] / self.reg["B"][n][:, t].sum()


def split_to_stackers(A: np.ndarray):
    M, N = A.shape
    y = []
    for i in range(M):
        y.append(A[i, :])
    return y


# noinspection PyUnreachableCode
def init():
    if True:
        np.random.seed(133484)
        game = Goofspiel()
        game.tree_init()

        T = 401
        game.sigma_init(T)
        game.cfv_init(T)
        game.cfv_refresh(0)
        for t in range(1, T):
            if t % 200 == 0:
                print(t)
            game.sigma_update(t)
            game.cfv_refresh(t)

    # reg
    if True:
        for l in game.bfs[0:-3]:
            for n in l:
                h = str(n.o.h())
                fig = plt.figure(h, [8, 4], 96)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                colors = [COLORS[a % 10] for a in n.o.actA]
                ax.stackplot(np.arange(T), *split_to_stackers(game.reg["A"][n]), labels=n.o.actA, colors=colors, alpha=0.5)
                ax.set_title(r'$R^A$' + h)
                ax.grid()
                ax.legend()
                fig.savefig(os.path.join("reg_A", h + ".png"))
                plt.close(fig)

                fig = plt.figure(h, [8, 4], 96)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                colors = [COLORS[b % 10] for b in n.o.actB]
                ax.stackplot(np.arange(T), *split_to_stackers(game.reg["B"][n]), labels=n.o.actB, colors=colors, alpha=0.5)
                ax.set_title(r'$R^B$' + h)
                ax.grid()
                ax.legend()
                fig.savefig(os.path.join("reg_B", h + ".png"))
                plt.close(fig)

    # sig
    if True:
        for l in game.bfs[0:-3]:
            for n in l:
                h = str(n.o.h())
                fig = plt.figure(h, [8, 4], 96)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                colors = [COLORS[a % 10] for a in n.o.actA]
                ax.stackplot(np.arange(T), *split_to_stackers(game.sig["A"][n]), labels=n.o.actA, colors=colors, alpha=0.5)
                ax.set_title(r'$\sigma^A$' + h)
                ax.grid()
                ax.legend()
                fig.savefig(os.path.join("sig_A", h + ".png"))
                plt.close(fig)

                fig = plt.figure(h, [8, 4], 96)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                colors = [COLORS[b % 10] for b in n.o.actB]
                ax.stackplot(np.arange(T), *split_to_stackers(game.sig["B"][n]), labels=n.o.actB, colors=colors, alpha=0.5)
                ax.set_title(r'$\sigma^B$' + h)
                ax.grid()
                ax.legend()
                fig.savefig(os.path.join("sig_B", h + ".png"))
                plt.close(fig)

    # cfv
    if True:
        for l in game.bfs[0:-3]:
            for n in l:
                h = str(n.o.h())
                fig = plt.figure(h, [8, 4], 96)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                ax.plot(np.arange(T), game.cfv[n][0], label="mixed")
                for i, a in enumerate(n.o.actA):
                    ax.plot(np.arange(T), game.opt["A"][n][i, :], label=a, alpha=0.3, color="C"+str(a))
                ax.set_title(r'$cfv^A$' + h)
                ax.grid()
                ax.legend()
                fig.savefig(os.path.join("cfv_A", h + ".png"))
                plt.close(fig)

                fig = plt.figure(h, [8, 4], 96)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                ax.plot(np.arange(T), game.cfv[n][1], label="mixed")
                for j, b in enumerate(n.o.actB):
                    ax.plot(np.arange(T), game.opt["B"][n][j, :], label=b, alpha=0.3, color="C"+str(b))
                ax.set_title(r'$cfv^B$' + h)
                ax.grid()
                ax.legend()
                fig.savefig(os.path.join("cfv_B", h + ".png"))
                plt.close(fig)

    # ep-value
    if True:
        for l in game.bfs[0:-3]:
            for n in l:
                h = str(n.o.h())
                fig: plt.Figure = plt.figure(h, [8, 4], 96)
                fig.add_axes([0.1, 0.1, 0.8, 0.8])
                fig.axes[0].loglog(np.arange(T), game.exp[n].sum(axis=0), label=r'$ep$')
                fig.axes[0].loglog(np.arange(T), game.exp[n][0, :], label=r'$ep_A$', alpha=0.3)
                fig.axes[0].loglog(np.arange(T), game.exp[n][1, :], label=r'$ep_B$', alpha=0.3)
                fig.axes[0].set_title(r'$ep$' + h)

                ax: plt.Axes
                for ax in fig.axes:
                    ax.set_ylim(0.01, 10)
                    ax.grid()
                    ax.legend()
                fig.savefig(os.path.join("ep", h + ".png"))
                plt.close(fig)

if __name__ == "__main__":
    init()
