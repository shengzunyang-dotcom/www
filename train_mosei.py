#from lib2to3.pgen2 import token
#from PIL import Image
import torch
# import wandb
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from global_configs_mosei import *

from transformers.models.clip.tokenization_clip import CLIPTokenizer
import argparse
from utils.utils import *
import pickle
from data.dataset import *
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.electra.tokenization_electra import ElectraTokenizer
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from modules.GLOT_mosei import *
from contextlib import redirect_stdout


parser = argparse.ArgumentParser()
parser.add_argument("--cuda_no", type=str, default=os.environ["CUDA_VISIBLE_DEVICES"])
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default=DATASETS)
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=32)#BATCH_SIZE
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=EPOCHS)
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")
parser.add_argument("--best_acc", type=float, default=0.1)
parser.add_argument("--wandb_name", type=str, default='none')
parser.add_argument("--domain_type", type=int, default=1)
parser.add_argument("--freeze", type=str, default='freeze')
parser.add_argument("--unimodal", type=str, default='text')
parser.add_argument("--layer", type=int, default=1)

parser.add_argument("--warm_up", type=int, default=5)

parser.add_argument("--test", type=int, default=0)


parser.add_argument("--t_dim", type=int, default=4096)
parser.add_argument("--v_dim", type=int, default=512)
parser.add_argument("--a_dim", type=int, default=1024)

parser.add_argument("--dg_label_dim", type=int, default=2)
parser.add_argument("--ds_label_dim", type=int, default=1)
parser.add_argument("--dsbert",type=str,default="T5")

parser.add_argument("--t_len",type=int,default=100)
parser.add_argument("--v_len",type=int,default=100)
parser.add_argument("--a_len",type=int,default=100)
parser.add_argument("--loss_rate",type=int,default=1)
parser.add_argument("--layers",type=int,default=1)
parser.add_argument("--attn_mask",action='store_false',
                    help='use attention mask for Transformer (default: true)')

parser.add_argument("--d_l", type=int, default=128)
parser.add_argument("--gran_t", type=int, default=10)
parser.add_argument("--gran_a", type=int, default=10)
parser.add_argument("--gran_v", type=int, default=10)
parser.add_argument("--TEXT_DIM", type=int, default=768)
parser.add_argument("--ACOUSTIC_DIM", type=int, default=1024)
parser.add_argument("--VISUAL_DIM", type=int, default=512)
parser.add_argument("--attn_dropout", type=float, default=0.5)
parser.add_argument("--num_heads", type=int, default=16) 
parser.add_argument("--relu_dropout", type=float, default=0.3)
parser.add_argument("--res_dropout", type=float, default=0.3)
parser.add_argument("--embed_dropout", type=float, default=0.2)
parser.add_argument("--p", type=int, default=2)
parser.add_argument("--entreg", type=float, default=.1)
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--classifier_dropout",type=float,default=0.0)
parser.add_argument("--num_classes",type=int,default=7)
parser.add_argument("--topk",type=int,default=10)
parser.add_argument("--sample_num",type=int,default=32)
parser.add_argument("--alpha",type=float,default=0.01)
parser.add_argument("--beta",type=float,default=0.001)

args = parser.parse_args()

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def get_loss_func():
    dg_loss_fct = CrossEntropyLoss()
    if args.domain_type == 1 or args.domain_type == 2:
        ds_loss_fct = MSELoss()
    else:
        ds_loss_fct = CrossEntropyLoss()
    return dg_loss_fct, ds_loss_fct

def prepare_training(train_dataloader):
    model = GLOTModel(args)

    # if torch.cuda.device_count() > 1:
    #     device_map = {f"cuda:{i}": f"cuda:{i}" for i in range(torch.cuda.device_count())}
    # else:
    #     device_map = {"cuda:0": "cuda:0"}

    # # 迁移未分配组件（网页4的模型并行策略）
    # for name, module in model.named_children():
    #     if not hasattr(module, 'device_map'):  # 非分片模块
    #         module.to(device_map['cuda:0'])

    model = model.to(DEVICE)
    # model = nn.DataParallel(model, device_ids=[0,1,2,3])
    
    # optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)#1e-5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCHS)
    return model, optimizer, scheduler

def compute_accurracy(preds, y_test,use_zero=False):
    preds = np.array(preds)
    y_test = np.array(y_test)
    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    # test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    # test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    # acc5 = multiclass_acc(test_preds_a5, test_truth_a5)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 ])
    preds = preds[non_zeros]
    y_test = y_test[non_zeros]
    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]
    preds = preds >= 0
    y_test = y_test >= 0
    f_score = f1_score(y_test, preds, average="weighted")
    recall = recall_score(y_test,preds,average="weighted")
    pre = precision_score(y_test,preds,average="weighted")
    acc = accuracy_score(y_test, preds)
    #return acc, mae, corr, f_score, acc5, acc7
    return acc, f_score, recall, pre ,mae,corr,acc7


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch,valid):
    step = 0
    tr_loss = 0
    to_l_loss = 0
    to_g_loss = 0
    to_proxy_loss = 0
    to_IB_loss = 0
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        sentence, input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, source_label, label_id, segment = batch
        input_ids, attention_mask, text_mask, visual, visual_mask, audio, audio_mask, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), text_mask.to(DEVICE),visual.to(DEVICE), visual_mask.to(DEVICE), audio.to(DEVICE), audio_mask.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
        # src_label = source_label.long()
        # label_id = label_id.long()
        step += 1
        optimizer.zero_grad()
        outputs,l_loss,g_loss,proxy_loss,IB_loss = model(input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch = epoch)
        # outputs,u_loss = model(input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch)
        # outputs ,l_loss = model(input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch)
        dg_loss_fct, ds_loss_fct = get_loss_func()        
        # (text_logits, visual_logits, audio_logits) = outputs
        # text_loss = dg_loss_fct(text_logits, label_id.view(-1))
        # visual_loss = dg_loss_fct(visual_logits, label_id.view(-1))
        # audio_loss = dg_loss_fct(audio_logits, label_id.view(-1))
        # total_loss = text_loss + audio_loss + visual_loss
        total_loss = ds_loss_fct(outputs,label_id.view(-1))+l_loss+g_loss+proxy_loss+IB_loss
        # total_loss = ds_loss_fct(outputs,label_id.view(-1))
        # total_loss = ds_loss_fct(outputs,label_id.view(-1))+u_loss
        # total_loss = dg_loss_fct(outputs,label_id.view(-1))
        # to_l_loss += l_loss.item()
        # to_g_loss += g_loss.item()
        # to_u_loss += u_loss.item()
        to_l_loss += l_loss
        to_g_loss += g_loss
        to_proxy_loss += proxy_loss
        to_IB_loss += IB_loss

        total_loss.backward()
        tr_loss += total_loss.item()
        optimizer.step()
        scheduler.step()
    tr_loss /= step
    to_l_loss/=step
    to_g_loss/=step
    to_proxy_loss/=step
    to_IB_loss/=step
    return tr_loss ,to_l_loss,to_g_loss,to_proxy_loss,to_IB_loss

def eval_epoch(model, dev_dataloader, optimizer, domain=0, epoch=0,valid = False):
    model.eval()
    if epoch == 50:
        save_path = './checkpoint/RAG_model_1.pth'
        torch.save(model.state_dict(), save_path)
        model = GLOTModel(args)
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        model.to(DEVICE)

    step = 0
    dev_loss = 0
    y_test = []
    text_preds = []
    visual_preds = []
    audio_preds = []
    preds = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            sentence, input_ids, attention_mask, text_mask,visual, visual_len, visual_mask, audio, audio_len, audio_mask, source_label, label_id, segment = batch
            input_ids, attention_mask, text_mask, visual, visual_mask, audio, audio_mask, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), text_mask.to(DEVICE), visual.to(DEVICE), visual_mask.to(DEVICE), audio.to(DEVICE), audio_mask.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
            # source_label = source_label.long()
            # label_id = label_id.long()
            outputs ,l_loss,g_loss,IB_loss= model(input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, epoch=epoch)
            # outputs = model(input_ids, attention_mask, text_mask,visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch)
            # outputs ,l_loss = model(input_ids, attention_mask, text_mask,visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch)
            dg_loss_fct, ds_loss_fct = get_loss_func()
            # outputs = outputs[0]
            # (text_logits, visual_logits, audio_logits) = outputs
            # text_loss = dg_loss_fct(text_logits, label_id.view(-1))
            # visual_loss = dg_loss_fct(visual_logits, label_id.view(-1))
            # audio_loss = dg_loss_fct(audio_logits, label_id.view(-1))
            # _, text_predicted = torch.max(text_logits.data, dim=1)
            # _, visual_predicted = torch.max(visual_logits.data, dim=1)
            # _, audio_predicted = torch.max(audio_logits.data, dim=1)
            loss = ds_loss_fct(outputs,label_id.view(-1))+l_loss+g_loss+IB_loss
            # loss = ds_loss_fct(outputs,label_id.view(-1))
            # loss = ds_loss_fct(outputs,label_id.view(-1))+l_loss
            # loss = dg_loss_fct(outputs,label_id.view(-1))
            # _,predicted = torch.max(outputs.data,dim = 1)
            y_test_val = label_id.view(-1).cpu().detach().tolist()

            # text_preds_val = text_predicted.cpu().detach().tolist()
            # text_preds += text_preds_val
            # visual_preds_val = visual_predicted.cpu().detach().tolist()
            # visual_preds += visual_preds_val
            # audio_preds_val = audio_predicted.cpu().detach().tolist()
            # audio_preds += audio_preds_val
            #preds_val = predicted.cpu().detach().tolist()
            preds_val = outputs.cpu().detach().tolist()
            preds += preds_val
            y_test += y_test_val
            
            
            total_loss = loss

        dev_loss += total_loss.item()
        # text_acc = accuracy_score(y_test, text_preds)
        # visual_acc = accuracy_score(y_test, visual_preds)
        # audio_acc = accuracy_score(y_test, audio_preds)
        # if valid == True:
        #     print('v_preds:',preds)
        #     print("v_y_test:",y_test)
        # else:
        #     print('t_preds:',preds)
        #     print("t_y_test:",y_test)
        acc, f_score, recall, pre , mae, corr,acc7= compute_accurracy(preds,y_test)
        return dev_loss, acc, f_score, recall, pre , mae, corr,acc7




def get_dataset():
    tokenizer = AutoTokenizer.from_pretrained(T5_PRETRAIN_PATH)
    # tokenizer = BertTokenizer.from_pretrained(PRETRAIN_PATH)
    # tokenizer = CLIPTokenizer.from_pretrained(PRETRAIN_PATH)
    mosi_path = 'merge/mosi_vgg_hubert.pkl'
    mosei_path = 'merge/mosei_vgg_hubert.pkl'
    meld_path = 'merge/meld_vgg_hubert.pkl'

    mosi_data_path = os.path.join(PATH, mosi_path)
    mosei_data_path = os.path.join(PATH, mosei_path)
    meld_data_path = os.path.join(PATH, meld_path)

    # with open(mosi_data_path, "rb") as handle:
    #     mosi_data = pickle.load(handle)

    with open(mosei_data_path, "rb") as handle:
        mosei_data = pickle.load(handle)

    # with open(meld_data_path, "rb") as handle:
    #     meld_data = pickle.load(handle)



    # train_dataset = mosi_data['train']
    # #print(train_dataset[0][0][2].shape)
    # dev_dataset = mosi_data['dev']
    # test_dataset = mosi_data['test']

    train_dataset = mosei_data['train']
    #print(train_dataset[0][0][2].shape)
    dev_dataset = mosei_data['dev']#除10余8
    test_dataset = mosei_data['test']#除10余1

    print(len(train_dataset)+len(dev_dataset)+len(test_dataset))

    # train_dataset = mosi_data['train'] + mosei_data['train'] + meld_data['train']
    # dev_dataset = mosi_data['dev'] + mosei_data['dev'] + meld_data['dev']
    # test_dataset = mosi_data['test'] + mosei_data['test'] + meld_data['test']


    train_data = MultimodalDataset(train_dataset, tokenizer)
    dev_data = MultimodalDataset(dev_dataset, tokenizer)
    test_data = MultimodalDataset(test_dataset, tokenizer)


    return train_data, dev_data, test_data




def get_dataloader(
        train_data, 
        dev_data, 
        test_data, 
        ):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size, collate_fn=padding_collate_fn,drop_last=True)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.train_batch_size, collate_fn=padding_collate_fn,drop_last=True)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.train_batch_size, collate_fn=padding_collate_fn,drop_last=True)
    # dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    # test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    num_train_optimization_steps = 0
    return train_dataloader, dev_dataloader, test_dataloader

def train(
    model,
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
):
    valid_losses, test_losses = [], []
    valid_text_best_accs, valid_visual_best_accs, valid_audio_best_accs = [], [], []
    test_text_best_accs, test_visual_best_accs, test_audio_best_accs = [], [], []

    acc_list = []
    f1_list = []
    mae_list = []
    corr_list = []

    for epoch in range(int(args.n_epochs)):
        train_loss,l_loss,g_loss,proxy_loss,IB_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch,valid = False)
        valid_loss, v_acc, v_f_score, v_recall, v_pre , v_mae, v_corr ,v_acc7= eval_epoch(model, dev_dataloader, optimizer, epoch=epoch,valid = True)
        test_loss, t_acc, t_f_score, t_recall, t_pre , t_mae, t_corr,t_acc7= eval_epoch(model, test_dataloader, optimizer, epoch=epoch,valid=True)

        with open('/data/yangshengzun/work2/log/log_mosei724.txt', 'a') as f:
            with redirect_stdout(f):
                print(
                    f"""epoch:{epoch}, train_loss:{train_loss}, l_loss:{l_loss},g_loss:{g_loss},proxy_loss:{proxy_loss},IB_loss:{IB_loss},valid_loss:{valid_loss},
    valid_acc:{v_acc},valid_f1:{v_f_score}, valid_recall:{v_recall}, valid_pre:{v_pre} ,valid_mae:{v_mae},valid_corr:{v_corr},valid_acc7:{v_acc7},
    test_acc:{t_acc}, test_f1:{t_f_score},test_recall:{t_recall},test_pre:{t_pre},test_mae:{t_mae},test_corr:{t_corr},test_acc7:{t_acc7}"""
                )
        print(
            f"epoch:{epoch}, train_loss:{train_loss}, l_loss:{l_loss},g_loss:{g_loss},proxy_loss:{proxy_loss},IB_loss:{IB_loss},valid_loss:{valid_loss},\
            valid_acc:{v_acc},valid_f1:{v_f_score}, valid_recall:{v_recall}, valid_pre:{v_pre} ,valid_mae:{v_mae},valid_corr:{v_corr},valid_acc7:{v_acc7}\
            test_acc:{t_acc}, test_f1:{t_f_score},test_recall:{t_recall},test_pre:{t_pre},test_mae:{t_mae},test_corr:{t_corr},test_acc7:{t_acc7}"
        )
        # tqdm.write(f"epoch:{epoch}, train_loss:{train_loss}, l_loss:{l_loss},g_loss:{g_loss},u_loss:{u_loss}valid_loss:{valid_loss},\
        #     valid_acc:{v_acc},valid_f1:{v_f_score}, valid_recall:{v_recall}, valid_pre:{v_pre} ,valid_mae:{v_mae},valid_corr:{v_corr},valid_acc7:{v_acc7}\
        #     test_acc:{t_acc}, test_f1:{t_f_score},test_recall:{t_recall},test_pre:{t_pre},test_mae:{t_mae},test_corr:{t_corr},test_acc7:{t_acc7}")
        valid_losses.append(valid_loss)

        acc_list.append(t_acc)
        f1_list.append(t_f_score)
        mae_list.append(t_mae)
        corr_list.append(t_corr)

        # valid_text_best_accs.append(valid_text_acc)
        # valid_visual_best_accs.append(valid_visual_acc)
        # valid_audio_best_accs.append(valid_audio_acc)


        # test_text_best_accs.append(test_text_acc)
        # test_visual_best_accs.append(test_visual_acc)
        # test_audio_best_accs.append(test_audio_acc)


        # wandb.log(
        #     (
        #         {
        #             "train_loss": train_loss,
        #             "best_valid_loss": min(valid_losses),
        #             "valid_loss": valid_loss,

        #             # 'valid_text_acc': valid_text_acc,
        #             # "valid_visual_acc": valid_visual_acc,
        #             # "valid_audio_acc": valid_audio_acc,
        #             # 'valid_text_best_acc': max(valid_text_best_accs),
        #             # "valid_visual_best_acc": max(valid_visual_best_accs),
        #             # "valid_audio_best_acc": max(valid_audio_best_accs),

        #             # 'text_text_acc': test_text_acc,
        #             # "test_visual_acc": test_visual_acc,
        #             # "test_audio_acc": test_audio_acc,
        #             # 'test_text_best_acc': max(test_text_best_accs),
        #             # "test_visual_best_acc": max(test_visual_best_accs),
        #             # "test_audio_best_acc": max(test_audio_best_accs),
    
        #         }
        #     )
        # )
    best_index = acc_list.index(max(acc_list))
    
    # 返回对应的指标
    return (acc_list[best_index], 
            f1_list[best_index], 
            mae_list[best_index], 
            corr_list[best_index])



def main():
    #wandb.init(project="Pretrained-DG", name=args.wandb_name)
    # wandb.init(project="ysz-project1", name=args.wandb_name)
    # wandb.config.update(args)
    # seed = [42,1234,3407,7890,2345,2,3,4,56,7]
    # seed = random.sample(range(1, 4000), 40)
    # seed = seed + [3404,3408,3403,3402,3405,3406,3409,3410,3417,3407]
    # seed = [3404]
    result = []
    # print("plot_dim:",args.prompt_plot)
    # for i in seed:
        # args.seed = 3407
        # args.seed = i
        # args.seed = 3407
    set_random_seed(args.seed)
        # full setting
    train_data, dev_data, test_data = get_dataset()
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(train_data, dev_data, test_data)
    model, optimizer, scheduler = prepare_training(train_dataloader)
    result_i = train(
                model=model,
                train_dataloader=train_dataloader,
                dev_dataloader=dev_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                scheduler=scheduler
                )
    result.append(result_i)
    model.cpu()  # 迁移模型参数到CPU内存
    del model, optimizer, scheduler  # 解除对象引用
    with open('/data/yangshengzun/work2/log/log_mosei724.txt', 'a') as f:
            with redirect_stdout(f):
                for i in range(len(result)):
                    print("learing_rate:",args.learning_rate,"seed:",args.seed,"acc:",result[i][0],"f1:",result[i][1],"mae:",result[i][2],"corr:",result[i][3])
                    print("\n")
    for i in range(len(result)):
        print("learing_rate:",args.learning_rate,"seed:",args.seed,"acc:",result[i][0],"f1:",result[i][1],"mae:",result[i][2],"corr:",result[i][3])



if __name__ == "__main__":
    main()
