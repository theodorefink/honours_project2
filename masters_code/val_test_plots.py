import os
import numpy as np

import torch.nn as nn

import sklearn
from pathlib import Path

from matplotlib import pyplot as plt

from models.instance_based.attenmil import attenmil
from models.instance_based.automil import automil
from models.instance_based.lnpmil import lnpmil
from models.instance_based.mixmil import mixmil

from models.embedding_based.abmil import ABMIL
from models.embedding_based.ac_mil import AC_MIL
from models.embedding_based.amd_mil import AMD_MIL
from models.embedding_based.clam_sb import CLAM_SB_MIL
from models.embedding_based.transmil import TransMIL

from models.litmodels import LitEmbeddingBasedModel, LitInstanceBasedModel
from dataset import WSI_Dataset

plt.rc('legend', fontsize=8)


def calc_conf_mats(df_dict, thresh):
    conf_mats = []
    for i in range(1,6):
        tp = np.sum((df_dict[f'M F{i}'] == 1.) & (df_dict[f'Fold {i}'] >= thresh))
        tn = np.sum((df_dict[f'M F{i}'] == 0.) & (df_dict[f'Fold {i}'] < thresh))
        fp = np.sum((df_dict[f'M F{i}'] == 0.) & (df_dict[f'Fold {i}'] >= thresh))
        fn = np.sum((df_dict[f'M F{i}'] == 1.) & (df_dict[f'Fold {i}'] < thresh))

        conf_mats.append((tp, tn, fp, fn))
    return conf_mats


def assemble_res_dict(ckpt_root, res, name):

    res_dict = {
            'M F1': np.array([val.item() for val in res[0][1]]),
            'Fold 1': np.array(res[0][0]),
            'M F2': np.array([val.item() for val in res[1][1]]),
            'Fold 2': np.array(res[1][0]),
            'M F3': np.array([val.item() for val in res[2][1]]),
            'Fold 3': np.array(res[2][0]),
            'M F4': np.array([val.item() for val in res[3][1]]),
            'Fold 4': np.array(res[3][0]),
            'M F5': np.array([val.item() for val in res[4][1]]),
            'Fold 5': np.array(res[4][0]),
        }

    return res_dict


def write_conf_mats(ckpt_root, name, mat005, mat010, test=False):
    if Path(f'{ckpt_root}/preds/val/{name}.txt').exists():
        return
    with open(f"{ckpt_root}/preds/{'test/' if test else 'val/'}{name}.txt", 'w') as f:
        f.write("Threshold at 0.05\n")
        for i in range(5):
            f.write(f'Fold {i+1}: ')
            f.write(f'TP: {mat005[i][0]}, FN: {mat005[i][3]}, FP: {mat005[i][2]}, TN: {mat005[i][1]}')
            f.write('\n')
        f.write('\n')
        f.write("Threshold at 0.10 \n")
        for i in range(5):
            f.write(f'Fold {i+1}: ')
            f.write(f'TP: {mat010[i][0]}, FN: {mat010[i][3]}, FP: {mat010[i][2]}, TN: {mat010[i][1]}')
            f.write('\n')
        f.write('\n')


def calc_metrics(mat):
    prec = mat[0] / (mat[0] + mat[2]) if mat[0] + mat[2] > 0 else 0
    rec = mat[0] / (mat[0] + mat[3]) if mat[0] + mat[3] > 0 else 0
    spec = mat[1] / (mat[1] + mat[2]) if mat[1] + mat[2] > 0 else 0
    acc = (mat[0] + mat[1])/sum(mat)
    return prec, rec, spec, acc


def write_metrics(ckpt_root, name, mat005, mat010, test=False):

    metrics_005 = np.array([calc_metrics(mat005[i]) for i in range(5)])
    metrics_010 = np.array([calc_metrics(mat010[i]) for i in range(5)])

    with open(f"{ckpt_root}/preds/{'test/' if test else 'val/'}/{name}.txt", 'w') as f:
        f.write("Threshold at 0.05\n\n")
        for i in range(5):
            f.write(f'Fold {i+1}:\n')
            prec, rec, spec, acc = metrics_005[i]
            f.write(f"Precision: {prec}\n")
            f.write(f"Recall: {rec}\n")
            f.write(f"Specificity: {spec}\n")
            f.write(f"Accuracy: {acc}\n")
            f.write('\n\n')

        f.write("Mean:\n")
        f.write(f"Precision: {np.mean(metrics_005[:,0])}\n")
        f.write(f"Recall: {np.mean(metrics_005[:,1])}\n")
        f.write(f"Specificity: {np.mean(metrics_005[:,2])}\n")
        f.write(f"Accuracy: {np.mean(metrics_005[:,3])}\n")
        f.write('\n\n')

        f.write("Threshold at 0.10\n\n")
        for i in range(5):
            f.write(f'Fold {i+1}:\n')
            prec, rec, spec, acc = metrics_010[i]
            f.write(f"Precision: {prec}\n")
            f.write(f"Recall: {rec}\n")
            f.write(f"Specificity: {spec}\n")
            f.write(f"Accuracy: {acc}\n")
            f.write('\n\n')
        
        f.write("Mean:\n")
        f.write(f"Precision: {np.mean(metrics_010[:,0])}\n")
        f.write(f"Recall: {np.mean(metrics_010[:,1])}\n")
        f.write(f"Specificity: {np.mean(metrics_010[:,2])}\n")
        f.write(f"Accuracy: {np.mean(metrics_010[:,3])}\n")
        f.write('\n\n')


def evals(ckpt_root):
    names = ['AutoMIL', 'AttenMIL', 'MixMIL', 'LNPMIL', 'ABMIL', 'AC-MIL', 'AMD-MIL', 'TransMIL', 'CLAM']

    ckpt_models = {}

    # I wrote a .txt file with the paths of the ckpts
    
    with open(f"{ckpt_root}/ckpts.txt", 'r') as f:
        lines = f.readlines()

    for i, name in enumerate(names):        # for each name
        curr = ckpt_models.get(name, [])
        for j in range(5):                  # 5 ckpts per name because 5 folds
            curr.append(lines[5*i+j].strip())
        ckpt_models[name] = curr

    feature_dim = 512
    model_aurocs = {}
    model_rocs = {}
    model_conf_mats_005 = {}
    model_conf_mats_010 = {}

    for name in names:
        if name == "AutoMIL":
            model = automil(input_size=feature_dim, output_size=1).cuda()
        if name == "AttenMIL":
            model = attenmil(input_size=feature_dim, output_size=1).cuda()
        if name == "LNPMIL":
            model = lnpmil(input_size=feature_dim, output_size=1).cuda()
        if name == "MixMIL":
            model = mixmil(input_size=feature_dim, output_size=1).cuda()
        if name == "ABMIL":
            model = ABMIL(input_size=feature_dim)
        if name == "AC-MIL":
            model = AC_MIL(in_dim=feature_dim, hidden_dim=int(feature_dim/4), num_classes=1).cuda()
        if name == "AMD-MIL":
            model = AMD_MIL(num_classes=1, in_dim=feature_dim, embed_dim=int(feature_dim/4), dropout=0., act = nn.ReLU()).cuda()
        if name == "TransMIL":
            model = TransMIL(input_size=feature_dim, n_classes=1).cuda()
        if name == "CLAM":
            model = CLAM_SB_MIL(in_dim=feature_dim).cuda()

        res = []


        for i, ckpt_path in enumerate(ckpt_models[name]):           #for each name, for each ckpt i.e. fold
            dataset = WSI_Dataset(f"path/to/valset/for/each/fold")
            
            if name in ['AutoMIL', 'AttenMIL', 'LNPMIL', 'MixMIL']:
                litmodel = LitInstanceBasedModel.load_from_checkpoint(ckpt_path, model=model)
                litmodel.eval()
                
                
                fold_res = []
                fold_labels = []
                
                for (vec, label, _) in dataset:
                    fold_res.append(litmodel.model(vec.cuda().float())[0].cpu().detach().numpy())
                    fold_labels.append(label)
                
                res.append((fold_res, fold_labels))
            
            else:
                litmodel = LitEmbeddingBasedModel.load_from_checkpoint(ckpt_path, model=model)
                litmodel.eval()
                
                fold_res = []
                fold_labels = []
                
                for (vec, label,_) in dataset:
                    fold_res.append(litmodel.model(vec.unsqueeze(0).cuda().float())["logits"].cpu().detach().numpy())
                    fold_labels.append(label)
                
                res.append((fold_res, fold_labels))

        res_dict = assemble_res_dict(ckpt_root, res, name)
        
        model_conf_mats_005[name] = calc_conf_mats(res_dict, 0.05)
        model_conf_mats_010[name] = calc_conf_mats(res_dict, 0.10)
        
        write_conf_mats(ckpt_root, name, model_conf_mats_005[name], model_conf_mats_010[name])
        write_metrics(ckpt_root, name, model_conf_mats_005[name], model_conf_mats_010[name])

        aurocs = []
        roc_curves = []
        for fold in res:
            fold_res, fold_labels = fold
            auroc = sklearn.metrics.roc_auc_score(fold_labels, fold_res)
            aurocs.append(auroc)
            roc_curves.append(sklearn.metrics.roc_curve(fold_labels, fold_res))


        # interpolation for the mean ROC-curve
        mean_fpr = np.linspace(0,1,100)
        interp_tprs = []

        for i in range(5):
            interp_tpr = np.interp(mean_fpr, roc_curves[i][0], roc_curves[i][1])
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1
        mean_auroc = sklearn.metrics.auc(mean_fpr, mean_tpr)

        aurocs.append(mean_auroc)

        roc_curves.append((mean_fpr, mean_tpr))

        model_aurocs[name] = aurocs
        model_rocs[name] = roc_curves

    return model_aurocs, model_rocs


def test_evals(ckpt_root):
    names = ['AutoMIL', 'AttenMIL', 'MixMIL', 'LNPMIL', 'ABMIL', 'AC-MIL', 'AMD-MIL', 'TransMIL', 'CLAM']

    ckpt_models = {}

    # i wrote a test_models.txt file with the ckpt paths of the best fold repeated 5 times because
    # I didn't want to rewrite the logic to get the ckpt paths, and only /plots/test/best.png matters

    with open(f"{ckpt_root}/test_models.txt", 'r') as f:
        lines = f.readlines()

    for i, name in enumerate(names):
        curr = ckpt_models.get(name, [])
        for j in range(5):
            curr.append(lines[5*i+j].strip())
        ckpt_models[name] = curr

    feature_dim = 512
    model_aurocs = {}
    model_rocs = {}
    model_conf_mats_005 = {}
    model_conf_mats_010 = {}

    dataset = WSI_Dataset(f'path/to/test/set')

    for name in names:
        if name == "AutoMIL":
            model = automil(input_size=feature_dim, output_size=1).cuda()
        if name == "AttenMIL":
            model = attenmil(input_size=feature_dim, output_size=1).cuda()
        if name == "LNPMIL":
            model = lnpmil(input_size=feature_dim, output_size=1).cuda()
        if name == "MixMIL":
            model = mixmil(input_size=feature_dim, output_size=1).cuda()
        if name == "ABMIL":
            model = ABMIL(input_size=feature_dim)
        if name == "AC-MIL":
            model = AC_MIL(in_dim=feature_dim, hidden_dim=int(feature_dim/4), num_classes=1).cuda()
        if name == "AMD-MIL":
            model = AMD_MIL(num_classes=1, in_dim=feature_dim, embed_dim=int(feature_dim/4), dropout=0., act = nn.ReLU()).cuda()
        if name == "TransMIL":
            model = TransMIL(input_size=feature_dim, n_classes=1).cuda()
        if name == "CLAM":
            model = CLAM_SB_MIL(in_dim=feature_dim).cuda()

        res = []

        for i, ckpt_path in enumerate(ckpt_models[name]):
            if name in ['AutoMIL', 'AttenMIL', 'LNPMIL', 'MixMIL']:
                litmodel = LitInstanceBasedModel.load_from_checkpoint(ckpt_path, model=model)
                litmodel.eval()
                
                fold_res = []
                fold_labels = []
                
                for (vec, label, coords) in dataset:
                    fold_res.append(litmodel.model(vec.cuda().float())[0].cpu().detach().numpy())
                    fold_labels.append(label)
                
                res.append((fold_res, fold_labels))
            
            else:
                litmodel = LitEmbeddingBasedModel.load_from_checkpoint(ckpt_path, model=model)
                litmodel.eval()
                
                fold_res = []
                fold_labels = []
                
                for (vec, label, coords) in dataset:
                    fold_res.append(litmodel.model(vec.unsqueeze(0).cuda().float())["logits"].cpu().detach().numpy())
                    fold_labels.append(label)
                
                res.append((fold_res, fold_labels))

        df = assemble_res_dict(ckpt_root, res, name)
        
        model_conf_mats_005[name] = calc_conf_mats(df, 0.05)
        model_conf_mats_010[name] = calc_conf_mats(df, 0.10)

        write_conf_mats(ckpt_root, name, model_conf_mats_005[name], model_conf_mats_010[name], test=True)
        write_metrics(ckpt_root, name, model_conf_mats_005[name], model_conf_mats_010[name], test=True)

        aurocs = []
        roc_curves = []
        for fold in res:
            fold_res, fold_labels = fold
            auroc = sklearn.metrics.roc_auc_score(fold_labels, fold_res)
            aurocs.append(auroc)
            roc_curves.append(sklearn.metrics.roc_curve(fold_labels, fold_res))


        # interpolation for the mean ROC-curve
        mean_fpr = np.linspace(0,1,100)
        interp_tprs = []

        for i in range(5):
            interp_tpr = np.interp(mean_fpr, roc_curves[i][0], roc_curves[i][1])
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1
        mean_auroc = sklearn.metrics.auc(mean_fpr, mean_tpr)

        aurocs.append(mean_auroc)

        roc_curves.append((mean_fpr, mean_tpr))

        model_aurocs[name] = aurocs
        model_rocs[name] = roc_curves

    with open(f"{ckpt_root}/preds/test/ROCs.txt", 'a') as f:
        for name in names:
            f.write(str(model_rocs[name][0]))

    return model_aurocs, model_rocs


def plot(ckpt_root, test=False):
    print(f'Making plots for {ckpt_root}.')
    os.makedirs(f"{ckpt_root}/plots/{"test" if test else "val"}", exist_ok=True)
    os.makedirs(f"{ckpt_root}/preds/{"test" if test else "val"}", exist_ok=True)

    if test:
        aurocs, rocs = test_evals(ckpt_root)
    else:
        aurocs, rocs = evals(ckpt_root)

    names = ['AutoMIL', 'AttenMIL', 'MixMIL', 'LNPMIL', 'ABMIL', 'AC-MIL', 'AMD-MIL', 'TransMIL', 'CLAM']


    # plot each fold ROC-Curve for one model at a time, and the mean ROC-Curve
    for name in names:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        for j, fold in enumerate(rocs[name]):
            if j == 5:
                axes.plot(fold[0], fold[1], label=f'Mean: {aurocs[name][j]:.2f}')
            else:
                axes.plot(fold[0], fold[1], label=f'Fold {j+1}: {aurocs[name][j]:.2f}')
        axes.set_title(f'ROC-curves for {name}, mean AUROC={aurocs[name][-1]:.2f}')
        axes.set_xlabel("False Positive Rate (FPR)")
        axes.set_ylabel("True Positive Rate (TPR)")
        axes.legend()
        axes.set_box_aspect(1)
        

        plt.tight_layout()
        plt.savefig(f"{ckpt_root}/plots/{"test" if test else "val"}/{name}.png")


    # plot the mean ROC-Curves for each model
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    for name in names:
        axes.plot(rocs[name][-1][0], rocs[name][-1][1], label=f'{name}: {aurocs[name][5]:.2f}')

    axes.set_title('Mean ROC Curves')
    axes.set_xlabel("False Positive Rate (FPR)")
    axes.set_ylabel("True Positive Rate (TPR)")
    axes.legend()
    axes.set_box_aspect(1)

    plt.tight_layout()

    plt.savefig(f"{ckpt_root}/plots/{"test" if test else "val"}/all.png")

    # plot the best ROC-Curves for each model
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    for name in names:
        best = np.argmax(aurocs[name])
        axes.plot(rocs[name][best][0], rocs[name][best][1], label=f'{name}: {aurocs[name][best]:.2f}')

    axes.set_title('Best-Fold ROC Curves')
    axes.set_xlabel("False Positive Rate (FPR)")
    axes.set_ylabel("True Positive Rate (TPR)")
    axes.legend()
    axes.set_box_aspect(1)

    plt.tight_layout()

    plt.savefig(f"{ckpt_root}/plots/{"test" if test else "val"}/best.png")

    print(f'Finished and Saved plots for {ckpt_root}')
    print('\n')

    plt.close('all')



ckpt_conch_512 = Path("/local/oli24/masters_repo/proper_02_16_dim_512")

plot(ckpt_conch_512)
# plot(ckpt_conch_512, test=True)


