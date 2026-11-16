import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA

import os
import shutil
import matplotlib.pyplot as plt


# add project path to sys to import relative modules
import sys
sys.path.insert(0, '..') # add parent dir

from models.guide import InvSlotAttentionGuide
from models.architectures import Predictor
from main import params
from datasets.data_utils import MyDataset
from utils.loss import compute_matching_cost


import logging
logfile_name = f"eval-{params['jobID']}.log"
logger = logging.getLogger(params["running_type"])
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
main_dir = os.path.abspath(__file__+'/../')

def main():    

    assert params["running_type"] == "eval"



    logger.info(device)
    
    A = torch.tensor([
                    [0., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                ])
    L = len(params["latent_vars"])
    ldim = params["latent_dim"]

    guide = InvSlotAttentionGuide(model_name=params["model_name"],
                                  resolution = params['resolution'],
                                  num_iterations = 3,
                                  latent_dim = params["latent_dim"],
                                  nlatents = len(params["latent_vars"]),
                                  A = A
                                  ).to(device)

    GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['epoch_to_load']}.pth"
    guide.load_state_dict(torch.load(GUIDE_PATH, weights_only=False)["model"])
    guide.eval()
    
    #logger.info(f'\n\nseed {seed}\n\n')
    logger.info(GUIDE_PATH)

    plots_dir = os.path.abspath("set_prediction_plots")
    if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
    else: 
        shutil.rmtree(plots_dir)
        os.mkdir(plots_dir)
    
    for p in guide.parameters():
        p.requires_grad = False

    
    for OOD_EVAL in [False]:

        logger.info(f"\nOOD: {OOD_EVAL}\n")

        dataset_dir = os.path.join(params["dataset_path"], "eval")
        n_scenes = 6
        dataset = MyDataset(dataset_dir, OOD_EVAL)
        dataloader = DataLoader(dataset, n_scenes, shuffle=True, generator=torch.Generator(device='cuda'))

        with torch.no_grad():
            for b, (img, target) in enumerate(dataloader):
                img = img.to(device)
                target = target.to(device)

                logger.info(f"targets: {target[:n_scenes]}")

                B = img.size(0)
                
                mu_z, z_noise, z_scm, img_hat, recon_figures = guide(img, target)
                z0, z1, z2, z3 = torch.split(torch.flatten(z_scm, -2), params["latent_dim"], -1)
                preds = guide.predictor(torch.flatten(mu_z, -2))
                logger.info(f"\ninitial preds: {preds[:n_scenes]}")

                _, aligned_targets = compute_matching_cost(preds, target)
                aligned_targets = torch.flatten(aligned_targets, 0, 1).cpu()

                for C in [0, 1, 2, 3]:
                
                    if C == 0:
                        z0_int = torch.flip(z0, dims=(1,))
                        z_noise, _, _, _ = guide.causal_flow.backward(z_scm)
                        s1_int = guide.causal_flow.s_cond(z0_int) * guide.causal_flow.scale_gain
                        t1_int = guide.causal_flow.t_cond(z0_int) * guide.causal_flow.shift_gain
                        z1_int = torch.exp(s1_int)*z_noise[:, :, 1, :] + t1_int
                        
                        z_int = torch.cat((z0_int.unsqueeze(-2),
                                        z1_int.unsqueeze(-2),
                                        z2.unsqueeze(-2),
                                        z3.unsqueeze(-2)), dim=-2) 

                        logger.info(f"\nscale intervention")
                        


                    elif C == 1:
                        z1_perm = torch.flip(z1, dims=(1,))
                        z_int = torch.cat((z0.unsqueeze(-2),
                                        z1_perm.unsqueeze(-2),
                                        z2.unsqueeze(-2),
                                        z3.unsqueeze(-2)), dim=-2)

                        logger.info(f"\ncolor intervention") 

                    elif C == 2:
                        z2_perm = torch.flip(z2, dims=(1,))
                        z_int = torch.cat((z0.unsqueeze(-2),
                                        z1.unsqueeze(-2),
                                        z2_perm.unsqueeze(-2),
                                        z3.unsqueeze(-2)), dim=-2) 

                        logger.info(f"\nx intervention")

                    elif C == 3:
                        z3_perm = torch.flip(z3, dims=(1,))
                        z_int = torch.cat((z0.unsqueeze(-2),
                                        z1.unsqueeze(-2),
                                        z2.unsqueeze(-2),
                                        z3_perm.unsqueeze(-2)), dim=-2) 

                        logger.info(f"\ny intervention")

                    
                    x_int, _ = guide.scene_decoder(z_int)
                    x_int = torch.clamp(x_int, 0., 1.)
                    _, _, z_causal_int = guide.encoder(x_int)
                    z_causal_int = z_causal_int.reshape(z_causal_int.size(0), z_causal_int.size(1), L, ldim)
                    x_int_cons, _ = guide.scene_decoder(z_causal_int)

                    changes = F.mse_loss(z_causal_int, z_int, reduction="none").mean(dim=[0, 1, 3])
                    logger.info(f"MSE: {changes}")


                    fig, ax = plt.subplots(n_scenes, 4)
                    for i in range(n_scenes):
                        ax[i, 0].imshow(np.clip(np.transpose(img[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                        if i == 0: ax[i, 0].set_title("img")
                        ax[i, 1].imshow(np.clip(np.transpose(img_hat[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                        if i == 0: ax[i, 1].set_title("img_hat")
                        ax[i, 2].imshow(np.clip(np.transpose(x_int[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                        if i == 0: ax[i, 2].set_title(f"perm {C}")
                        ax[i, 3].imshow(np.clip(np.transpose(x_int_cons[i].detach().cpu().numpy(), (1, 2, 0)), 0., 1.))
                        if i == 0: ax[i, 3].set_title(f"img_hat_cons")

                        for j in range(4):
                            ax[i, j].grid(False)
                            ax[i, j].axis('off')

                    plt.savefig(f"{params['check_attn_folder']}/{params['latent_vars'][C]}_intervention.png")
                    #plt.savefig(f"{params['check_attn_folder']}/{params['latent_vars'][C]}_intervention.svg", format="svg", dpi=1000, bbox_inches="tight")
                    plt.close()

                if b == 0: break
     

if __name__ == '__main__':
    main()