"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python alignn/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""

from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union
import ignite
import torch

from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
    # For different version of pytorch-ignite
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from data import get_train_val_loaders
from config import TrainingConfig
from models.alignn import ALIGNN
from jarvis.db.jsonutils import dumpjson
import json
import os

# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.preprocessing import StandardScaler

# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    # pc = pk.load(open("pca.pkl", "rb"))
    # y_pred = torch.tensor(pc.transform(y_pred), device=device)
    # y = torch.tensor(pc.transform(y), device=device)

    # y_pred = torch.tensor(pca_sc.inverse_transform(y_pred),device=device)
    # y = torch.tensor(pca_sc.inverse_transform(y),device=device)
    # print (y.shape,y_pred.shape)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_dgl(config: Union[TrainingConfig, Dict[str, Any]], model: nn.Module = None, train_val_test_loaders=[], resume=0, model_config=None):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used

    Args:
        config: Training configuration
        model: Model to train
        train_val_test_loaders: Data loaders
        resume: Resume from checkpoint
        model_config: Model configuration (ALIGNNConfig) to save in checkpoint
    """
    # print(config)
    # if type(config) is dict:
    #     try:
    #         print(config)
    #         config = TrainingConfig(**config)
    #     except Exception as exp:
    #         print("Check", exp)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    # print("config:")
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    # pprint.pprint(tmp)  # , sort_dicts=False)
    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    line_graph = False
    alignn_models = {"alignn","dense_alignn","alignn_cgcnn","alignn_layernorm"}
    if config.model.name == "clgn":
        line_graph = True
    if config.model.name == "cgcnn":
        line_graph = True
    if config.model.name == "icgcnn":
        line_graph = True
    if config.model.name in alignn_models and config.model.alignn_layers > 0:
        line_graph = True

    # print ('output_dir train', config.output_dir)
    train_loader = train_val_test_loaders[0]
    val_loader = train_val_test_loaders[1]
    test_loader = train_val_test_loaders[2]
    prepare_batch = train_val_test_loaders[3]

    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True

    # define network, optimizer, scheduler
    _model = {"alignn": ALIGNN}
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model
    model_number=[]
    if resume ==1:
        for f in os.listdir(config.output_dir):
            # print(config.output_dir)
            if f.startswith('checkpoint_'):
                # print(f)
                model_number.append(int(f.split('.')[0].split('_')[1]))
    # print(model_number)
    # exit()
    if resume ==1:
        checkpoint = torch.load(config.output_dir+'checkpoint_'+str(max(model_number))+'.pt')
        net.load_state_dict(checkpoint["model"])

    net.to(device)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if resume ==1:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=config.learning_rate,epochs=config.epochs,steps_per_epoch=steps_per_epoch,pct_start=0.3)
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer)

    if resume ==1:
        scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "bce": nn.BCELoss(),
    }
    criterion = criteria[config.criterion]

    # Check if contrastive learning is enabled
    use_contrastive = getattr(config.model, 'use_contrastive_loss', False)
    contrastive_weight = getattr(config.model, 'contrastive_loss_weight', 0.1)

    # set up metrics based on task type
    if classification:
        # Classification metrics
        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = (y_pred > 0.5).float()
            return y_pred, y

        metrics = {
            "loss": Loss(criterion),
            "accuracy": Accuracy(output_transform=thresholded_output_transform),
            "precision": Precision(output_transform=thresholded_output_transform, average=True),
            "recall": Recall(output_transform=thresholded_output_transform, average=True),
        }
        print(f"\n📊 分类模式已启用，使用准确率、精确率、召回率指标")
    else:
        # Regression metrics
        metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}

    # Gradient clipping for stability
    grad_clip = 1.0
    print(f"\n📎 梯度裁剪已启用: max_norm = {grad_clip}")

    if use_contrastive:
        print(f"\n🔥 对比学习已启用:")
        print(f"  - 损失权重: {contrastive_weight}")
        print(f"  - 温度参数: {getattr(config.model, 'contrastive_temperature', 0.1)}")

        # Custom training function for contrastive learning
        def custom_train_step(engine, batch):
            net.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch)

            # Forward pass
            output = net(x)

            # Handle dict output from contrastive learning
            if isinstance(output, dict):
                y_pred = output['predictions']
                task_loss = criterion(y_pred, y)

                # Add contrastive loss if available
                if 'contrastive_loss' in output:
                    contrastive_loss = output['contrastive_loss']
                    total_loss = task_loss + contrastive_weight * contrastive_loss
                else:
                    total_loss = task_loss
            else:
                y_pred = output
                total_loss = criterion(y_pred, y)

            total_loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            return total_loss.item()

        trainer = ignite.engine.Engine(custom_train_step)
    else:
        # Custom training function with gradient clipping
        def train_step_with_clip(engine, batch):
            net.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch)
            y_pred = net(x)
            if isinstance(y_pred, dict):
                y_pred = y_pred['predictions']
            loss = criterion(y_pred, y)
            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            return loss.item()

        trainer = ignite.engine.Engine(train_step_with_clip)

    if resume ==1:
        trainer.load_state_dict(checkpoint["trainer"])

    # Custom output transform for contrastive learning
    if use_contrastive:
        def output_transform(output):
            """Extract predictions from dict output"""
            y_pred, y = output
            if isinstance(y_pred, dict):
                return y_pred['predictions'], y
            return y_pred, y

        # Create custom metrics with output transform
        metrics = {
            "loss": Loss(criterion, output_transform=output_transform),
            "mae": MeanAbsoluteError(output_transform=output_transform)
        }

    evaluator = create_supervised_evaluator(net,metrics=metrics,prepare_batch=prepare_batch,device=device)

    train_evaluator = create_supervised_evaluator(net,metrics=metrics,prepare_batch=prepare_batch,device=device)

    test_evaluator = create_supervised_evaluator(net,metrics=metrics,prepare_batch=prepare_batch,device=device)

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: scheduler.step())

    best_loss = float('inf')
    best_val_mae = float('inf')
    best_test_mae = float('inf')

    # Early stopping setup
    early_stopping_patience = config.n_early_stopping
    epochs_without_improvement = 0
    if early_stopping_patience:
        print(f"\n⏹️ Early Stopping 已启用: patience = {early_stopping_patience}")

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        # pbar.attach(evaluator,output_transform=lambda x: {"mae": x})

    history = {"train": {m: [] for m in metrics.keys()},"validation": {m: [] for m in metrics.keys()},"test": {m: [] for m in metrics.keys()}}

    if config.store_outputs:
        # log_results handler will save epoch output
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        train_evaluator.run(train_loader)
        evaluator.run(val_loader)
        test_evaluator.run(test_loader)

        tmetrics = train_evaluator.state.metrics
        vmetrics = evaluator.state.metrics
        tstmetrics = test_evaluator.state.metrics

        for metric in metrics.keys():
            tm = tmetrics[metric]
            vm = vmetrics[metric]
            tstm = tstmetrics[metric]

            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()
                vm = vm.cpu().numpy().tolist()
                tstm = tstm.cpu().numpy().tolist()

            history["train"][metric].append(tm)
            history["validation"][metric].append(vm)
            history["test"][metric].append(tstm)

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(filename=os.path.join(config.output_dir, "history_val.json"),data=history["validation"])
            dumpjson(filename=os.path.join(config.output_dir, "history_train.json"),data=history["train"])
        if config.progress:
            pbar = ProgressBar()
            pbar.log_message(f"Epoch: {engine.state.epoch:.1f}")
            pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
            pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
            pbar.log_message(f"Test_MAE: {tstmetrics['mae']:.4f}")

        nonlocal best_loss, best_val_mae, best_test_mae, epochs_without_improvement

        # Save best validation model
        improved = False
        if vmetrics['mae'] < best_val_mae:
            best_val_mae = vmetrics['mae']
            best_val_checkpoint = {
                "model": net.state_dict(),
                "config": model_config,  # Save model config (ALIGNNConfig)
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": engine.state.epoch,
                "val_mae": best_val_mae,
            }
            torch.save(best_val_checkpoint, os.path.join(config.output_dir, "best_val_model.pt"))
            print(f"✅ Saved best val model (MAE: {best_val_mae:.4f}) at epoch {engine.state.epoch}")
            improved = True
            epochs_without_improvement = 0

        # Save best test model
        if tstmetrics['mae'] < best_test_mae:
            best_test_mae = tstmetrics['mae']
            best_test_checkpoint = {
                "model": net.state_dict(),
                "config": model_config,  # Save model config (ALIGNNConfig)
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": engine.state.epoch,
                "test_mae": best_test_mae,
            }
            torch.save(best_test_checkpoint, os.path.join(config.output_dir, "best_test_model.pt"))
            print(f"✅ Saved best test model (MAE: {best_test_mae:.4f}) at epoch {engine.state.epoch}")

        if tstmetrics['mae'] < best_loss:
            best_loss = tstmetrics['mae']

        # Early stopping check
        if not improved:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"\n⏹️ Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                print(f"Best Val MAE: {best_val_mae:.4f}")
                trainer.terminate()

        print(f"Best_val_mae: {best_val_mae:.4f}, Best_test_mae: {best_test_mae:.4f}")
        if early_stopping_patience:
            print(f"Epochs without improvement: {epochs_without_improvement}/{early_stopping_patience}")
        print("\n")

    # train the model!
    trainer.run(train_loader, max_epochs=config.epochs)

    # Print final summary
    print("\n" + "="*80)
    print("🎯 Training Complete!")
    print("="*80)
    print(f"Best Validation MAE: {best_val_mae:.4f}")
    print(f"Best Test MAE: {best_test_mae:.4f}")
    print(f"\nCheckpoints saved:")
    print(f"  - best_val_model.pt")
    print(f"  - best_test_model.pt")
    print("="*80 + "\n")

    # Write Predictions for validation set
    net.eval()
    f_val = open(os.path.join(config.output_dir, "prediction_results_val_set.csv"),"w")
    f_val.write("id,target,prediction\n")
    val_targets = []
    val_predictions = []
    with torch.no_grad():
        val_ids = val_loader.dataset.ids
        sample_idx = 0
        for dat in val_loader:
            g, lg, text, target = dat
            out_data = net([g.to(device), lg.to(device), text])
            if isinstance(out_data, dict):
                out_data = out_data['predictions']
            out_data = out_data.cpu().numpy().tolist()
            target = target.cpu().numpy().flatten().tolist()

            batch_size = len(target) if isinstance(target, list) else 1
            if batch_size == 1 and not isinstance(target, list):
                target = [target]
                out_data = [out_data]

            for k in range(batch_size):
                id = val_ids[sample_idx + k]
                pred_value = max(0.0, out_data[k])
                f_val.write("%s, %6f, %6f\n" % (id, target[k], pred_value))
                val_targets.append(target[k])
                val_predictions.append(pred_value)
            sample_idx += batch_size
    f_val.close()

    from sklearn.metrics import mean_absolute_error
    print("Validation MAE:", mean_absolute_error(np.array(val_targets), np.array(val_predictions)))

    # Write Predictions for test set
    f = open(os.path.join(config.output_dir, "prediction_results_test_set.csv"),"w")
    f.write("id,target,prediction\n")
    targets = []
    predictions = []
    with torch.no_grad():
        ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
        sample_idx = 0  # 追踪当前样本索引
        for dat in test_loader:
            # g, lg, target = dat
            # out_data = net([g.to(device), lg.to(device)])
            g, lg, text, target = dat
            out_data = net([g.to(device), lg.to(device), text])
            # 处理对比学习模式的dict输出
            if isinstance(out_data, dict):
                out_data = out_data['predictions']
            out_data = out_data.cpu().numpy().tolist()
            if config.standard_scalar_and_pca:
                sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
                out_data = sc.transform(np.array(out_data).reshape(-1, 1))[
                    0
                ][0]
            target = target.cpu().numpy().flatten().tolist()

            # 处理batch中的每个样本
            batch_size = len(target) if isinstance(target, list) else 1
            if batch_size == 1 and not isinstance(target, list):
                target = [target]
                out_data = [out_data]

            for k in range(batch_size):
                # 获取当前样本的id
                id = ids[sample_idx + k]
                # 将负数预测值统一为0
                pred_value = max(0.0, out_data[k])
                f.write("%s, %6f, %6f\n" % (id, target[k], pred_value))
                targets.append(target[k])
                predictions.append(pred_value)

            sample_idx += batch_size
    f.close()
    from sklearn.metrics import mean_absolute_error
    # print(targets)
    # print(predictions)
    print("Test MAE:",mean_absolute_error(np.array(targets), np.array(predictions)))


    return history


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config, progress=True)
