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

# import ipdb; ipdb.set_trace()
# nf_pdata,nf_pmean, nf_pda, nf_pcmb = simutils.simflow_forward(simflow)
# eigmodes = [0,10,20,30,40]
# simutils.sns_pairplot(nf_pdata.T,nf_pmean,nf_pda,nf_pcmb,eigmodes)
# import ipdb; ipdb.set_trace()
# simutils.sns_pairplot(nf_pdata.T,nf_pmean,nf_pda,nf_pcmb,eigmodes)
# plt.savefig("outputs/pairplot_00R_a0-80_nflow.png")

true_amp = 10000.0
amp = np.logspace(np.log10(true_amp*1e-2),np.log10(true_amp*100),10000)
simflow.run(amp,true_amp)
print("# layers in SINF model: ", simflow.flow.model.ndim)

##---------------------------------------------------------------------------##

fig,ax = plt.subplots(1,2,figsize=(12,6))

ax[0].plot(simflow.amp,simutils.exp(simflow.da_loglikelihood))
ax[1].plot(simflow.amp,simutils.exp(simflow.cmb_loglikelihood))

ax[0].axvline(x=true_amp,linestyle="--",color="k",label="truth")
ax[1].axvline(x=true_amp,linestyle="--",color="k",label="truth")

ax0 = ax[0].twinx()
ax1 = ax[1].twinx()
ax0.plot(simflow.amp,simflow.da_loglikelihood,color="r",label="loglikelihood")
ax1.plot(simflow.amp,simflow.cmb_loglikelihood,color="r",label="loglikelihood")

ax[0].set_ylabel("likelihood")
ax[1].set_ylabel("likelihood")
ax0.set_ylabel("log likelihood",color="r")
ax1.set_ylabel("log likelihood",color="r")

ax0.legend(loc="upper right")
ax1.legend(loc="upper right")

ax0.set_yscale("symlog")
ax1.set_yscale("symlog")

ax[0].set_xlabel("test amplitude")
ax[1].set_xlabel("test amplitude")

ax[0].set_title(f"{true_amp:,.0f}x true DA amplitude")
ax[1].set_title(f"{true_amp:,.0f}x true CMB amplitude")

ax[0].set_xscale("log")
ax[1].set_xscale("log")

ax[0].legend(loc="lower left")
ax[1].legend(loc="lower left")

# ax1.set_ylim(-1e29,-1e27)

plt.suptitle(simflow.params["flow"]["path"])
plt.tight_layout()
plt.show()





