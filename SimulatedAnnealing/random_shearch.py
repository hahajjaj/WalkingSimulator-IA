

import itertools
import os
import random
import numpy as np
from environment import ConfigurableAnimalEnv

from ia import GeneticAlgoByTime
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from concurrent import futures
def eval(ConfigurableAnimalEnv, pop_size, gen_max, time_max, selection_rate, mutation_rate):
    ai = GeneticAlgoByTime(ConfigurableAnimalEnv, pop_size, gen_max, time_max, selection_rate=selection_rate, selection_cap=selection_rate, mutation_rate=mutation_rate, mutation_cap=mutation_rate)

    ai.train()

    return selection_rate, mutation_rate, ai.info['Best Score'], ai.info

if __name__ == '__main__':

    # from matplotlib.backend_bases import MouseButton
    # import matplotlib.pyplot as plt
    # import numpy as np

    # t = np.arange(0.0, 1.0, 0.01)
    # s = np.sin(2 * np.pi * t)
    # fig, ax = plt.subplots()
    # lines = ax.plot(t, s)


    # def on_move(event):
    #     if event.inaxes:
    #         cont, ind = lines[0].contains(event)
    #         # print(f'data coords {event.xdata} {event.ydata},',
    #         #     f'pixel coords {event.x} {event.y}')


    # def on_click(event):
    #     if event.button is MouseButton.LEFT:
    #         print('disconnecting callback')
    #         plt.disconnect(binding_id)


    # binding_id = plt.connect('motion_notify_event', on_move)
    # plt.connect('button_press_event', on_click)

    # plt.show()
    # exit()

    num_param = 2 # **2

    pop_size = 4
    gen_max = 5
    time_max = 100

    selection_rates = np.random.rand(num_param).tolist()
    mutation_rate = np.random.rand(num_param).tolist()


    parameters = list(itertools.product(selection_rates, mutation_rate))
    # random.shuffle(parameters)

    # results = []

    with futures.ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.75)) as executor:
        res = executor.map(eval, *zip(*[
            (ConfigurableAnimalEnv, pop_size, gen_max, time_max, selection_rate, mutation_rate)
            for selection_rate, mutation_rate in parameters 
        ]))

    res = list(res)
    results = [(sel, mut, score) for sel, mut, score, _ in res]
    infos = [info for _, _, _, info in res]

    # for selection_rate, mutation_rate in parameters:
    #     ai = GeneticAlgoByTime(ConfigurableAnimalEnv, pop_size, gen_max, time_max, selection_rate=selection_rate, selection_cap=selection_rate, mutation_rate=mutation_rate, mutation_cap=mutation_rate)

    #     ai.train()

    #     score = ai.info['Best Score']

        # results.append((selection_rate, mutation_rate, score))
    best_selection_rate, best_mutation_rate, best_score = max(results, key=lambda x: x[2])
    print("Best parameters:", best_selection_rate, best_mutation_rate, "Score:", best_score)



    # X,Y = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    # Z = X * np.exp(-X**2 - Y**2)
    # plt.style.use('_mpl-gallery')
    # print(X.shape, Y.shape, Z.shape)
    # X,Y, Z= X.tolist(), Y.tolist(), Z.tolist()
    # Make data
    X_sel,Y_mut,Z_score = map(list, list(zip(*results)))

    points = np.array([X_sel, Y_mut]).T

    X_grid, Y_grid = np.mgrid[0:1:50j, 0:1:50j]
    # print(X_grid)

    Z_grid = griddata(points, Z_score, (X_grid, Y_grid), method='nearest')

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Selection Rate')
    ax.set_ylabel('Mutation Rate')
    ax.set_zlabel('Score')

    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cm.coolwarm, 
                        linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)





    ax2 = fig.add_subplot(121)

    # X_gen = np.array([np.arange(gen_max) for _ in range(len(parameters))])
    X_gen = list(range(1, gen_max+1))
    scores = np.array([info['Scores'] for info in infos])
    print(scores)
    Y_best_scores: np.ndarray = np.amax(scores, axis=2)
    print(Y_best_scores)
    Y_best_scores = Y_best_scores.reshape((gen_max, len(parameters)))

    lines = ax2.plot(X_gen, Y_best_scores)

    annot = ax2.annotate("", (0,0))
    annot.set_visible(False)

    def on_move(event):
        if event.inaxes:
            for line, (sel, mut, best_score) in zip(lines, results):
                cont, ind = line.contains(event)
                if cont:
                    print("hello")
                    annot.set_text(f"sel: {sel}\nmut:{mut}\nbest score: {best_score}")
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
                
            annot.set_visible(False)
            fig.canvas.draw_idle()

            # print(f'data coords {event.xdata} {event.ydata},',
            #     f'pixel coords {event.x} {event.y}')


    # def on_click(event):
    #     if event.button is MouseButton.LEFT:
    #         print('disconnecting callback')
    #         plt.disconnect(binding_id)


    binding_id = plt.connect('motion_notify_event', on_move)
    # plt.connect('button_press_event', on_click)

    plt.show()


    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_trisurf(X, Y, Z, linewidth=0, antialiased=False)

    # # ax.set(xticklabels=[],
    # #        yticklabels=[],
    # #        zticklabels=[])

    # plt.show()

    # plt.show()




