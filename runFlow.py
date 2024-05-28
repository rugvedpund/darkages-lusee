import simutils
import simflows as sf
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

##---------------------------------------------------------------------------##

parser = simutils.create_parser()
args = parser.parse_args()
params_path = args.params_yaml
retrain = args.retrain
print(args)

simflow = sf.SimFlow(args)
if retrain: 
    simflow.train()
    simflow.save()

amp = np.logspace(np.log10(0.01),np.log10(1000),1000)
simflow.run(amp)

##---------------------------------------------------------------------------##

fig,ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(simflow.amp,simutils.exp(simflow.da_loglikelihood))
ax[0].axvline(x=1,linestyle="--",color="k",label="Truth")
ax0 = ax[0].twinx()
ax0.plot(simflow.amp,simflow.da_loglikelihood,color="r",label="loglikelihood")
ax0.set_ylabel("Log Likelihood",color="r")
ax0.legend(loc="upper right")

# ax0.set_yscale("symlog")
ax[0].set_xlabel("Amplitude")
ax[0].set_ylabel("Likelihood")
ax[0].set_title("DA likelihood")
ax[0].set_xscale("log")
ax[0].legend(loc="lower left")

ax[1].plot(simflow.amp,simutils.exp(simflow.cmb_loglikelihood))
ax[1].axvline(x=1,linestyle="--",color="k",label="Truth")
ax1 = ax[1].twinx()
ax1.plot(simflow.amp,simflow.cmb_loglikelihood,color="r",label="loglikelihood")
ax1.set_ylabel("Log Likelihood",color="r")
ax1.legend(loc="upper right")
# ax1.set_yscale("symlog")
ax[1].set_xlabel("Amplitude")
ax[1].set_ylabel("Likelihood")
ax[1].set_title("CMB likelihood")
ax[1].set_xscale("log")
ax[1].legend(loc="lower left")

plt.suptitle(simflow.params["flow"]["path"])
plt.tight_layout()
plt.show()




