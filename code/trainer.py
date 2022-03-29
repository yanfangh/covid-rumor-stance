from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader
import math
from torch import nn
from evaluator import *
import torch
import os


logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Multi-class Focal loss implementation
    Copy from https://github.com/z814081807/DeepNER/blob/master/src/utils/model_utils.py
    """
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss


def build_loss_fct(loss_type, class_weight, device):
    if class_weight is not None:
        class_weight = torch.tensor(class_weight).to(device)
    if loss_type=='ce':
        loss_fct = nn.CrossEntropyLoss(weight=class_weight)
    elif loss_type=='focal':
        loss_fct = FocalLoss(weight=class_weight)
    return loss_fct


def build_optimizer_and_scheduler(args, model, t_total):

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    return optimizer, scheduler

def train(args, model, train_dataset, label_list, evaluator=None, class_weight=None):

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    label2idx = dict((label, i) for i, label in enumerate(label_list))

    if args.warmup_ratio is not None:
        args.warmup_steps = math.ceil(len(train_loader) * args.num_train_epochs * args.warmup_ratio)

    t_total = len(train_loader) * args.num_train_epochs
    loss_fct = build_loss_fct(args.loss_type, class_weight, device=args.device)
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Training batch size = {args.train_batch_size}")
    logger.info(f"  Loss type = {args.loss_type}, weight = {class_weight}")
    logger.info(f"  Total optimization steps = {t_total}")
    logger.info(f"  seed = {args.seed}")
    logger.info(f"  lr = {int(args.learning_rate/1e-5)}")

    global_steps = 0
    total_loss = 0
    for epoch in range(args.num_train_epochs):
        logger.info(f'Epoch: {epoch}')
        model.train()
        iterable = tqdm(train_loader)
        for i, batch in enumerate(iterable):

            b_labels = batch['labels']
            b_label_ids = torch.tensor([label2idx[label] for label in b_labels]).to(args.device)
            del batch['labels']

            outputs = model(**batch)

            loss = loss_fct(outputs, b_label_ids)

            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps

            # Perform a backward pass to calculate gradients
            loss.backward()

            total_loss += loss.item()

            if (i+1) % args.gradient_accumulation_steps == 0:
                # Clip the norm of the gradients to 1.0 to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                model.zero_grad()

            global_steps += 1

            # if global_steps%args.checkpoint_save_steps==0:
            #     torch.save(model.state_dict(), f'{args.ckpt_dir}/{global_steps}.pt')

            iterable.set_description(f'Loss: {loss.item() : 0.4f}')

        ckpt_path = os.path.join(args.ckpt_dir, str(args.seed))
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save(model.state_dict(), f'{ckpt_path}/checkpoint_{epoch}.pt')

        avg_train_loss = total_loss / global_steps
        logger.info("Average train loss: %.4f" %(avg_train_loss))

        if args.eval_during_training:
            evaluator(args, model, on=['overall'])

    # if global_steps%args.checkpoint_save_steps!=0:
    #     torch.save(model.state_dict(), f'{args.ckpt_dir}/{global_steps}.pt')











