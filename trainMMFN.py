import os
import argparse

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from transformers import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from MMFN import MultiModal
import myweibo_dataset as weibo_data
import gossipcop_dataset as gossip_data


logging.set_verbosity_warning()
logging.set_verbosity_error()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
ACTIVE_CLIPMODEL = gossip_data.clipmodel


def parse_args():
    parser = argparse.ArgumentParser(description="Train MMFN")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=2, help="Patience for early stopping")
    parser.add_argument("--exp_name", type=str, default="mmfn_base", help="Experiment name for saving results")
    parser.add_argument("--alpha", type=float, default=0.08, help="Contrastive loss weight")
    parser.add_argument("--use_entity_enrich", action="store_true", default=True,
                        help="Enable entity enrichment path: jieba/tagme + wikipedia first sentence + entity BERT")
    parser.add_argument("--dataset", type=str, default="weibo", choices=["weibo", "gossip"],
                        help="Dataset to train/evaluate on")
    parser.add_argument("--hard_sample_downweight", action="store_true", default=False,
                        help="Downweight repeatedly misclassified training samples instead of deleting them")
    parser.add_argument("--hard_sample_warmup_epochs", type=int, default=2,
                        help="Warmup epochs before enabling hard-sample downweight")
    parser.add_argument("--hard_sample_decay", type=float, default=0.9,
                        help="Multiplicative decay for misclassified sample weights")
    parser.add_argument("--hard_sample_min_weight", type=float, default=0.5,
                        help="Lower bound for hard-sample weights")

    return parser.parse_args()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def build_dataloader(args):
    global ACTIVE_CLIPMODEL
    os.environ["USE_ENTITY_ENRICH"] = "1" if args.use_entity_enrich else "0"
    batch_size = args.batch_size

    if args.dataset == "gossip":
        os.environ["GOSSIP_TRAIN_CSV"] = "train_gossipcop.clean.csv"
        os.environ["GOSSIP_TEST_CSV"] = "test_gossipcop.clean.csv"
        train_set = gossip_data.gossipcop_dataset(is_train=True)
        validate_set = gossip_data.gossipcop_dataset(is_train=False)
        collate = gossip_data.collate_fn
        dataset_name = "gossip"
        ACTIVE_CLIPMODEL = gossip_data.clipmodel
    else:
        train_set = weibo_data.weibo_dataset(is_train=True)
        validate_set = weibo_data.weibo_dataset(is_train=False)
        collate = weibo_data.collate_fn
        dataset_name = "weibo"
        ACTIVE_CLIPMODEL = weibo_data.clipmodel

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collate,
        shuffle=True
    )
    test_loader = DataLoader(
        validate_set,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collate,
        shuffle=True
    )
    return train_loader, test_loader, dataset_name


def unpack_batch(batch):
    if len(batch) == 11:
        (input_ids, attention_mask, token_type_ids,
         entity_input_ids, entity_attention_mask, entity_token_type_ids,
         image, imageclip, textclip, label, sample_indices) = batch
        return (input_ids, attention_mask, token_type_ids,
                entity_input_ids, entity_attention_mask, entity_token_type_ids,
                image, imageclip, textclip, label, sample_indices)
    if len(batch) == 10:
        (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label,
         entity_input_ids, entity_attention_mask, entity_token_type_ids) = batch
        return (input_ids, attention_mask, token_type_ids,
                entity_input_ids, entity_attention_mask, entity_token_type_ids,
                image, imageclip, textclip, label, None)
    if len(batch) == 8:
        input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label, sample_indices = batch
        return (input_ids, attention_mask, token_type_ids,
                None, None, None,
                image, imageclip, textclip, label, sample_indices)
    raise ValueError(f"Unexpected batch format with length={len(batch)}")


def test(rumor_module, test_loader, dataset_name, alpha=0.08):
    rumor_module.eval()
    loss_f_rumor = torch.nn.CrossEntropyLoss(reduction='none')

    rumor_count = 0
    loss_total_sum = 0
    rumor_label_all = []
    rumor_pre_label_all = []

    print("[Eval] start validation...", flush=True)
    eval_bar = tqdm(test_loader, desc="eval", smoothing=0, mininterval=1.0)

    with torch.no_grad():
        for _, batch in enumerate(eval_bar):
            (input_ids, attention_mask, token_type_ids,
             entity_input_ids, entity_attention_mask, entity_token_type_ids,
             image, imageclip, textclip, label, sample_indices) = unpack_batch(batch)

            input_ids = to_var(input_ids)
            attention_mask = to_var(attention_mask)
            token_type_ids = to_var(token_type_ids)
            image = to_var(image)
            imageclip = to_var(imageclip)
            textclip = to_var(textclip)
            label = to_var(label)
            if entity_input_ids is not None:
                entity_input_ids = to_var(entity_input_ids)
                entity_attention_mask = to_var(entity_attention_mask)
                entity_token_type_ids = to_var(entity_token_type_ids)

            image_clip = ACTIVE_CLIPMODEL.encode_image(imageclip)
            text_clip = ACTIVE_CLIPMODEL.encode_text(textclip)

            pre_rumor, loss_con = rumor_module(
                input_ids, attention_mask, token_type_ids,
                image, text_clip, image_clip,
                labels=None,
                dataset_name=dataset_name,
                entity_input_ids=entity_input_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
            )
            loss_ce = loss_f_rumor(pre_rumor, label).mean()
            loss_total = loss_ce + alpha * loss_con
            pre_label_rumor = pre_rumor.argmax(1)

            loss_total_sum += loss_total.item() * input_ids.shape[0]
            rumor_count += input_ids.shape[0]

            rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
            rumor_label_all.append(label.detach().cpu().numpy())

    loss_rumor_test = loss_total_sum / rumor_count
    rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
    rumor_label_all = np.concatenate(rumor_label_all, 0)

    acc_rumor_test = accuracy_score(rumor_label_all, rumor_pre_label_all)
    precision_rumor_test = precision_score(rumor_label_all, rumor_pre_label_all, average=None)
    recall_rumor_test = recall_score(rumor_label_all, rumor_pre_label_all, average=None)
    f1_rumor_test = f1_score(rumor_label_all, rumor_pre_label_all, average=None)
    conf_rumor = confusion_matrix(rumor_label_all, rumor_pre_label_all)

    classification_report_rumor = classification_report(
        rumor_label_all, rumor_pre_label_all,
        target_names=["realnews", "fakenews"], digits=4
    )

    print("Overall Accuracy:", acc_rumor_test, flush=True)
    print("Precision per class:", precision_rumor_test, flush=True)
    print("Recall per class:", recall_rumor_test, flush=True)
    print("F1 Score per class:", f1_rumor_test, flush=True)
    print("Confusion Matrix:\n", conf_rumor, flush=True)
    print("Classification Report:\n", classification_report_rumor, flush=True)

    return acc_rumor_test, precision_rumor_test, recall_rumor_test, f1_rumor_test, loss_rumor_test, conf_rumor


def train(args):
    patience = args.patience
    best_loss = np.inf
    patience_counter = 0
    alpha = args.alpha

    train_loader, test_loader, dataset_name = build_dataloader(args)

    print("✓ 使用原始MMFN版本")
    rumor_module = MultiModal()

    rumor_module.to(device)
    loss_f_rumor = torch.nn.CrossEntropyLoss(reduction='none')

    sample_weights = torch.ones(len(train_loader.dataset), dtype=torch.float32)

    base_params = list(map(id, rumor_module.bert.parameters()))
    base_params += list(map(id, rumor_module.entity_bert.parameters()))
    base_params += list(map(id, rumor_module.swin.parameters()))

    optim_task = torch.optim.Adam([
        {
            "params": filter(lambda p: p.requires_grad and id(p) not in base_params, rumor_module.parameters()),
            "lr": args.lr,
            "weight_decay": 5e-4
        },
        {"params": rumor_module.bert.parameters(), "lr": args.lr * 1e-2, "weight_decay": 1e-4},
        {"params": rumor_module.entity_bert.parameters(), "lr": args.lr * 1e-2, "weight_decay": 1e-4},
        {"params": rumor_module.swin.parameters(), "lr": args.lr * 1e-2, "weight_decay": 1e-4}
    ], lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_task,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6
    )

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
        rumor_module.train()

        corrects_pre_rumor = 0
        total_loss_sum = 0
        rumor_count = 0

        tk0 = tqdm(train_loader, desc="train", smoothing=0, mininterval=1.0)
        for _, batch in enumerate(tk0):
            (input_ids, attention_mask, token_type_ids,
             entity_input_ids, entity_attention_mask, entity_token_type_ids,
             image, imageclip, textclip, label, sample_indices) = unpack_batch(batch)

            input_ids = to_var(input_ids)
            attention_mask = to_var(attention_mask)
            token_type_ids = to_var(token_type_ids)
            image = to_var(image)
            imageclip = to_var(imageclip)
            textclip = to_var(textclip)
            label = to_var(label)
            if entity_input_ids is not None:
                entity_input_ids = to_var(entity_input_ids)
                entity_attention_mask = to_var(entity_attention_mask)
                entity_token_type_ids = to_var(entity_token_type_ids)

            with torch.no_grad():
                image_clip = ACTIVE_CLIPMODEL.encode_image(imageclip)
                text_clip = ACTIVE_CLIPMODEL.encode_text(textclip)

            pre_rumor, loss_con = rumor_module(
                input_ids, attention_mask, token_type_ids,
                image, text_clip, image_clip,
                labels=label,
                dataset_name=dataset_name,
                entity_input_ids=entity_input_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
            )

            loss_ce_vec = loss_f_rumor(pre_rumor, label)
            if args.hard_sample_downweight:
                batch_weights = sample_weights[sample_indices].to(loss_ce_vec.device)
                loss_ce = (loss_ce_vec * batch_weights).mean()
            else:
                loss_ce = loss_ce_vec.mean()
            loss_total = loss_ce + alpha * loss_con

            optim_task.zero_grad()
            loss_total.backward()
            optim_task.step()

            pre_label_rumor = pre_rumor.argmax(1)
            corrects_pre_rumor += pre_label_rumor.eq(label.view_as(pre_label_rumor)).sum().item()
            total_loss_sum += loss_total.item() * input_ids.shape[0]
            rumor_count += input_ids.shape[0]

            if args.hard_sample_downweight and (epoch + 1) > args.hard_sample_warmup_epochs:
                with torch.no_grad():
                    mis_mask = pre_label_rumor.ne(label).detach().cpu()
                    mis_indices = sample_indices[mis_mask]
                    if mis_indices.numel() > 0:
                        sample_weights[mis_indices] = torch.clamp(
                            sample_weights[mis_indices] * args.hard_sample_decay,
                            min=args.hard_sample_min_weight
                        )

        loss_rumor_train = total_loss_sum / rumor_count
        acc_rumor_train = corrects_pre_rumor / rumor_count

        print("[Train] epoch finished, entering validation...", flush=True)

        acc_rumor_test, precision_rumor_test, recall_rumor_test, f1_rumor_test, loss_rumor_test, conf_rumor = test(
            rumor_module, test_loader, dataset_name, alpha=alpha
        )

        print("-----------rumor detection----------------", flush=True)
        print(
            "EPOCH = %d || acc_rumor_train = %.3f || acc_rumor_test = %.3f || loss_rumor_train = %.3f || loss_rumor_test = %.3f"
            % (epoch + 1, acc_rumor_train, acc_rumor_test, loss_rumor_train, loss_rumor_test),
            flush=True
        )
        print("-----------rumor_confusion_matrix---------", flush=True)
        print(conf_rumor, flush=True)

        scheduler.step(loss_rumor_test)

        if loss_rumor_test < best_loss:
            best_loss = loss_rumor_test
            patience_counter = 0
            torch.save(rumor_module.state_dict(), args.best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered, but ignored as requested.", flush=True)
                # break

    rumor_module.load_state_dict(torch.load(args.best_model_path))
    return rumor_module, test_loader, dataset_name


if __name__ == "__main__":
    args = parse_args()

    print("=== MMFN 训练脚本 ===")

    exp_dir = os.path.join("ckpt", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.best_model_path = os.path.join(exp_dir, "best_model.pth")

    model, test_loader, dataset_name = train(args)
    test(model, test_loader, dataset_name, alpha=args.alpha)

    print("=== 训练完成 ===")
