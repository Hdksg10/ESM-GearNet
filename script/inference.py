import os
import sys
import math
import pprint
import random

import numpy as np

import torch
from torch.optim import lr_scheduler

from torchdrug import core, models, tasks, datasets, utils, data
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import model, cdconv, gvp, dataset, task, protbert

from torch.utils import data as torch_data
import pandas as pd

def train_and_validate(cfg, solver, scheduler):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 50)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        metric = solver.evaluate("valid")
        # solver.evaluate("test")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver


def test(cfg, solver):
    solver.evaluate("valid")
    return solver.evaluate("test")

@torch.no_grad()
def predict(cfg, solver):
    solver.evaluate("valid")
    test_set = getattr(solver, "%s_set" % "test")
    sampler = torch_data.DistributedSampler(test_set, solver.world_size, solver.rank)
    dataloader = data.DataLoader(test_set, solver.batch_size, sampler=sampler, num_workers=solver.num_worker)
    model = solver.model
    model.split = "test"
    model.eval()
    preds = []
    targets = []
    mutations = []
    for batch in dataloader:
        if solver.device.type == "cuda":
            batch = utils.cuda(batch, device=solver.device)
        for mutation in batch["mutation"]:
            mutations.append(mutation)
        pred, target = model.predict_and_target(batch)
        preds.append(pred)
        targets.append(target)
    # print(mutations)
    # print(len(mutations))
    pred = utils.cat(preds)
    target = utils.cat(targets)
    
    # print(len(pred))
    if solver.world_size > 1:
        pred = comm.cat(pred)
        target = comm.cat(target)
        mutation = comm.cat(mutation)
    return pred, mutations

def save_file(result, file):
    mutation = result["mutation"]
    prediction = result["prediction"]
    print(len(mutation))
    print(len(prediction))
    assert len(mutation) == len(prediction)
    
    index = range(len(mutation))
    
    # 将字典数组转换为数据框
    df = pd.DataFrame(result, index=index)

    # 将数据框保存为 CSV 文件
    df.to_csv(file, index=False)

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    seed = args.seed
    output = args.output
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver, scheduler = util.build_downstream_solver(cfg, dataset)

    # train_and_validate(cfg, solver, scheduler)
    # test(cfg, solver)
    pred, mutation = predict(cfg, solver)
    pred = pred.squeeze()
    pred = pred.detach().cpu().numpy()
    # print(pred.shape)
    result = {
        "mutation": mutation,
        "prediction": pred
    
    }
    # save_file = "/root/GearNet/ESM-GearNet/proteinmul/ac.csv"
    # print(pred)
    
    save_file(result, output)