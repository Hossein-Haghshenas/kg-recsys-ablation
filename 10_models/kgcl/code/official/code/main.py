from tqdm import tqdm
from contrast import Contrast
from torch import optim
from torch.optim import optimizer, lr_scheduler
import world
import utils
from world import cprint
import torch
import numpy as np
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset, kg_dataset
Recmodel = register.MODELS[world.model_name](world.config, dataset, kg_dataset)
Recmodel = Recmodel.to(world.device)
contrast_model = Contrast(Recmodel).to(world.device)
optimizer = optim.Adam(Recmodel.parameters(), lr=world.config['lr'])
if world.dataset == "MIND":
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.2)
else:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1500, 2500], gamma = 0.2)
bpr = utils.BPRLoss(Recmodel, optimizer)
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = world.config.get("neg_k", 1)

# tensorboardX is optional; the runner always passes --tensorboard=0
if world.tensorboard:
    from tensorboardX import SummaryWriter
    w = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    # for early stop
    # recall@20
    least_loss = 1e5
    best_result = 0.
    stopping_step = 0

    for epoch in tqdm(range(world.TRAIN_epochs), disable=True):
        start = time.time()
        # transR learning
        if epoch%1 == 0:
            if world.train_trans:
                cprint("[Trans]")
                trans_loss = Procedure.TransR_train(Recmodel, optimizer)
                print(f"trans Loss: {trans_loss:.3f}")

        
        # joint learning part
        if not world.pretrain_kgc:
            cprint("[Drop]")
            if world.kgc_joint:
                contrast_views = contrast_model.get_views()
            else:
                contrast_views = contrast_model.get_views("ui")
            cprint("[Joint Learning]")
            if world.kgc_joint or world.uicontrast!="NO":
                output_information = Procedure.BPR_train_contrast(dataset, Recmodel, bpr, contrast_model, contrast_views, epoch, optimizer, neg_k=Neg_k,w=w)
            else:
                # LightGCN
                output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            

            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            if epoch<world.test_start_epoch:
                if epoch %5 == 0:
                    cprint("[VALID]")
                    Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], eval_dict=dataset.validDict)
            else:
                if epoch % world.test_verbose == 0:
                    cprint("[VALID]")
                    result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], eval_dict=dataset.validDict)
                    recall = np.asarray(result["recall"]).reshape(-1)
                    ks = list(world.topks)
                    k_index = ks.index(20) if 20 in ks else len(ks) - 1
                    recall20 = float(recall[k_index])
                    print(f"VALID_SCORE {recall20:.4f}")
                    if recall20 > best_result:
                        stopping_step = 0
                        best_result = recall20
                        print("find a better model")
                        torch.save(Recmodel.state_dict(), weight_file)
                    else:
                        stopping_step += 1
                        if stopping_step >= world.early_stop_cnt:
                            print(f"early stop triggerd at epoch {epoch}")
                            break
        
        scheduler.step()
    print(f"BEST_VALID {best_result:.6f}")
finally:
    if world.tensorboard:
        w.close()