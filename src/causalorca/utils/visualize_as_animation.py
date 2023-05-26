import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from causalorca.utils.util import ensure_dir


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def animate_scene(title, scene, scene_causality, peds, figname='animation.gif', lim=6.0, only_causal=True):
    figname_folder = os.path.dirname(figname)
    ensure_dir(figname_folder)

    num_traj = scene.shape[0]
    num_frame = scene.shape[2]
    # print('animate', scene.shape)
    cm_subsection = np.linspace(0.0, 1.0, num_traj)
    colors = [matplotlib.cm.jet(x) for x in cm_subsection]

    plt.close()
    fig = plt.figure(figsize=(8, 6))

    xlists = []
    ylists = []
    xlists_non_causal = []
    ylists_non_causal = []
    l = []
    legend = []

    for i in range(num_traj):
        xlists.append([])
        ylists.append([])
        xlists_non_causal.append([])
        ylists_non_causal.append([])
        # l.append(plt.plot([], [], color=colors[i]))
        # legend.append('Agent ' + str(i + 1))

    for i in range(2 * num_traj):
        if i >= num_traj:
            l.append(plt.plot([], [], linestyle=':', marker="x", color=colors[i - num_traj], linewidth=2))
        else:
            if i == 0:
                l.append(plt.plot([], [], color=colors[i], marker='v', linewidth=2))
                legend.append('Ego-agent ' + str(i + 1))
            else:
                l.append(plt.plot([], [], color=colors[i], marker='o', linewidth=1.5))
                legend.append('Agent ' + str(i + 1))

    # print(len(l))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    # plt.legend(legend)

    metadata = dict(title='Movie', artist='codinglikemad')
    writer = PillowWriter(fps=6, metadata=metadata)

    # print(len(peds[0][0]))
    with writer.saving(fig, figname, 140):

        # Plot the first line and cursor
        for i, xval in enumerate(peds[0][0]):
            # print(enumerate(peds[0][0]))
            # print(peds[0][0].shape)
            xlists[0].append(xval)
            ylists[0].append(peds[0][1][i])
            # l1, = l[0]
            # l[0][0].set_data(xlists[0], ylists[0])

            for j in range(1, num_traj):
                # print(j)
                if only_causal:
                    # print('hey', peds[0].shape[1], peds[0][:][1])
                    # print(i, j)
                    if scene_causality[0][i][j] == 1:
                        xlists[j].append(peds[j][0][i])
                        ylists[j].append(peds[j][1][i])
                        xlists_non_causal[j].append(None)
                        ylists_non_causal[j].append(None)
                    else:
                        # print(xlists_non_causal)
                        xlists[j].append(None)
                        ylists[j].append(None)
                        xlists_non_causal[j].append(peds[j][0][i])
                        ylists_non_causal[j].append(peds[j][1][i])
                else:
                    xlists[j].append(peds[j][0][i])
                    ylists[j].append(peds[j][1][i])

            for k in range(2 * num_traj):
                if k >= num_traj:
                    l[k][0].set_data(xlists_non_causal[k - num_traj], ylists_non_causal[k - num_traj])
                else:
                    # print(k)
                    l[k][0].set_data(xlists[k], ylists[k])

            writer.grab_frame()

    plt.close()


def animate(scene, causality, figname='animate.gif', lim=6.0,
            title='Motion of agents that have a causal effect on agent 1 (dotted line means non-causal'):
    # print(data.shape)
    scene = torch.tensor(scene).permute(0, 2, 1)
    scene_causality = torch.tensor(causality)
    nb_ped = scene.shape[0]

    peds = []

    for i in range(nb_ped):
        peds.append(scene[i].numpy())

    nb_ped = scene.shape[0]

    animate_scene(title, scene, scene_causality, peds, figname, only_causal=True, lim=lim)
