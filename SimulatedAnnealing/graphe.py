import matplotlib as plt
import SimulatedAnnealing 
import ia





def generate_grapics():

    sa = ia.load_ai("/Users/ehlalouchsafouan/Desktop/projet3/PROJET-BA3/classSA/file")
    plt.plot(sa.iterationsListe, sa.scoresListe)
    #plt.fill_between(info["Generation"], np.mean(np.asarray(info["Scores"]), axis=1) - np.std(np.asarray(info["Scores"]), axis=1), np.mean(np.asarray(info["Scores"]), axis=1) + np.std(np.asarray(info["Scores"]), axis=1), alpha=0.2, label="mean-std")
    #plt.fill_between(info["Generation"], np.amax(np.asarray(info["Scores"]), axis=1), np.median(np.asarray(info["Scores"]), axis=1), alpha=0.2, label="max-median")
    #plt.plot(info["Generation"], np.amax(np.asarray(info["Scores"]), axis=1), label="max fitness")

    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.legend(loc="upper left")
    plt.title("Simulated Annealing - Initial temperature {}".format(sa.t0))
    plt.show()



generate_grapics()