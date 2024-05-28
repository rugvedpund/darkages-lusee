import simutils
from simflows import SkyAnalyzer
import numpy as np
import pickle

##---------------------------------------------------------------------------##
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    import seaborn as sns
    import pandas as pd

    parser = simutils.create_parser()
    args = parser.parse_args()
    config_paths = args.configs
    output_paths = args.outputs

    print("loading sky..")
    sky = SkyAnalyzer().from_args(args)
    # sky = SkyAnalyzer().from_configs(config_paths)
    # sky.save("outputs/sky.pkl")
    # sky = pickle.load(open("outputs/sky.pkl", "rb"))

    print("plot 00R..")
    plt.figure(figsize=(8, 8))
    sky.set_comb("00R")
    sky.doPCA_and_project()
    simutils.plt_scree(sky, ax=plt.gca())

    plt.plot([], [], "C0", label="mean ulsa")
    plt.plot([], [], "C1", label="mean da")
    plt.plot([], [], "C2", label="mean cmb")
    plt.plot([], [], "C3", label="rms ulsa")
    plt.title(f"00R, {output_paths}", fontsize="x-small")
    plt.legend()
    plt.show()

    # print("plot auto..")
    # plt.figure(figsize=(8, 8))
    # alphas = [1, 0.7, 0.5, 0.3]
    # for si, subsample in enumerate([1, 10, 30, 60]):
    #     print(f"subsample: {subsample}")
    #     sky.set_comb("auto")
    #     sky.subsample(subsample)
    #     sky.doPCA_and_project()
    #     simutils.plt_scree(sky, ax=plt.gca(), alpha=alphas[si])
    #     plt.plot([], [], "gray", label=f"{subsample} min", alpha=alphas[si])
    # plt.plot([], [], "C0", label="mean ulsa")
    # plt.plot([], [], "C1", label="mean da")
    # plt.plot([], [], "C2", label="mean cmb")
    # plt.plot([], [], "C3", label="rms ulsa")
    # plt.title(f"auto, {sim_paths}", fontsize="x-small")
    # plt.legend()
    # plt.show()

    # print("plot pair plot..")
    # sky.set_comb("00R")
    # sky.subsample(60)
    # sky.doPCA_and_project()
    # eigmodes = [0, 10, 20, 30, 40]
    # _, ndata = sky.ulsa.norm_pdata.shape
    # d = np.vstack(
    #     [
    #         sky.ulsa.norm_pdata.T,
    #         sky.ulsa.norm_pmean,
    #         sky.da.norm_pmean,
    #         sky.cmb.norm_pmean,
    #     ]
    # )
    # index = ["data"] * ndata + ["mean ulsa", "mean da", "mean cmb"]
    # df = pd.DataFrame(d, index=index).reset_index()
    # kwargs = {
    #     "markers": [".", "d", "^", "v"],
    #     "height": 1,
    #     "vars": eigmodes,
    #     "hue": "index",
    # }
    # sns.pairplot(df, **kwargs)
    # plt.show()

###---------------------------------------------------------------------------##
