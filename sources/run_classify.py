from sources.early_stopping import Early_Stopping, Early_Stopping_multi
from sources.create_dataset import Create_Dataset
from sources.utils import init_logger, set_seed
from sources.preprocessor import Preprocessor
from sources.mongo_processor import Mongo
from sources.model import BERT
from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import logging
from tqdm import tqdm
import numpy as np
import logging
import config
import wandb
import torch
import os



mongo = Mongo(mongo_uri=config.MONGO_URI,
              db_name = config.DB_NAME,
              collection = config.COLLECTION)
mongo_test = Mongo(mongo_uri=config.MONGO_URI,
              db_name = config.DB_NAME,
              collection = config.COLLECTION_TEST)
init_logger()
logger = logging.getLogger(__name__)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def classify(do_train, args):
    logger.info(do_train)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

    if do_train:
        datas = mongo.find_item()
        sentences, encoed_labels, labels, labels_str = Preprocessor(datas).pre_process()

        x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.001, random_state=42, shuffle=True)

        logger.info('x_train_len:{}'.format(len(x_train)))
        logger.info('x_test_len:{}'.format(len(x_test)))

        train_dataset = Create_Dataset(datas=x_train,
                                       labels=y_train,
                                       max_len=config.MODEL_CONFIG['max_len'],
                                       tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=config.MODEL_CONFIG['batch_size'], shuffle=True, num_workers=0)

        val_dataset = Create_Dataset(datas=x_test,
                                     labels=y_test,
                                     max_len=config.MODEL_CONFIG['max_len'],
                                     tokenizer=tokenizer)
        val_dataloader = DataLoader(val_dataset, batch_size=config.MODEL_CONFIG['batch_size'], shuffle=True, num_workers=0)
        train(train_dataloader, val_dataloader, args, do_train=True)
 
    else:
        datas = mongo_test.find_item()
        sentences, encoed_labels, labels, labels_str = Preprocessor(datas).pre_process()

        test_dataset = Create_Dataset(datas=sentences,
                                      labels=labels,
                                      max_len=config.MODEL_CONFIG['max_len'],
                                      tokenizer=tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=config.MODEL_CONFIG['batch_size'], shuffle=False,
                                  num_workers=1)
        evlauate(test_dataloader, args, do_train=False)

def train(train_dataloader, val_dataloader, args, do_train):
    if torch.cuda.is_available():
        if args.target_gpu == 'm':
            model = KoElectra_m()
            device = torch.device("cuda:0")
            logger.info("We willl use {} GPUs".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model, device_ids=[0,1,2], dim=0)
        else:
            model = BERT(args)
            device = torch.device('cuda:' + str(args.target_gpu))
            logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.info('We will use the GPU:{}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')
    
    model.to(device)

    t_total = len(train_dataloader) // config.MODEL_CONFIG['gradient_accumulation_steps'] * config.MODEL_CONFIG['epochs'] # gpu 메모리 효율적으로 하기위함.

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", config.MODEL_CONFIG['epochs'])
    logger.info("  Total train batch size = %d", config.MODEL_CONFIG['batch_size'])
    logger.info("  Gradient Accumulation steps = %d", config.MODEL_CONFIG['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", config.MODEL_CONFIG['logging_steps'])
    logger.info("  Save steps = %d", config.MODEL_CONFIG['save_steps'])

    wandb.watch(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.MODEL_CONFIG['learning_rate'],
                      eps=config.MODEL_CONFIG['adam_epsilon'])
    

    epochs = config.MODEL_CONFIG['epochs']

    #linear scheduler
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config.MODEL_CONFIG['warmup_steps'], t_total=t_total)
    scaler = torch.cuda.amp.GradScaler() 

    global_step = 0
    total_correct = 0
    step = 0
    model.zero_grad()
    early_stopping = Early_Stopping(verbose = True)
    early_stopping_m = Early_Stopping_multi(verbose = True)

    for epoch in range(1, epochs + 1):
        results ={}
        train_loss = 0
        val_loss = 0
        model.train()
        logger.info('####################### Epoch {}: Training Start #######################'.format(epoch))

        for batch_idx, data in enumerate(train_dataloader):
            step +=1

            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)

            if args.target_gpu == 'm':
                with torch.cuda.amp.autocast():
                    outputs, pred, real, correct, _ = model(ids,labels, device)
                total_correct += correct.sum().item()
                loss = outputs.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, pred, real, correct, _ = model(ids,labels)
                total_correct += correct.sum().item()
                loss = outputs[0]
                loss.backward() 
                optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            train_acc = total_correct / len(real)
            train_f1 = f1_score(real.tolist(), pred.tolist(), average='weighted')

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scheduler.step()
            model.zero_grad()
            ckpt ="TRAIN_{}epoch_{}step_loss{}_acc{}.pt".format(epoch,step,loss,train_acc)
            
            ck_path = os.path.join(args.ck_path, ckpt)

            logger.info('---------------------- [Train] Epoch {}/{}, Steps {}/{} ----------------------'.format(epoch, epochs, batch_idx, len(train_dataloader)))
            logger.info('                   Accuracy {}, Loss {}, F1 score {}\n'.format(train_acc, loss, train_f1))

            wandb.log({"(TRAIN)loss": loss, "epoch":epoch, "custom_step": step, "(TRAIN)accuracy": train_acc, "(TRAIN)f1": train_f1})

            if config.MODEL_CONFIG['save_steps'] > 0 and step % config.MODEL_CONFIG['save_steps'] == 0:
                if args.target_gpu == 'm': # multi GPU
                    early_stopping_m(loss,model, ck_path)
                else:
                    early_stopping(loss,model, ck_path)

            if config.MODEL_CONFIG['logging_steps'] > 0 and step % len(train_dataloader) == 0:
                if do_train:
                    val_loss, val_acc= validation(args, model, val_dataloader, epoch)
                    val_loss = val_loss / len(val_dataloader)
                    train_loss = train_loss / len(train_dataloader)

                    logger.info('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}\n'.format(epoch,train_loss,val_loss))
                    logger.info('####################### Epoch {}: Validation End #######################'.format(epoch))
            
            total_correct = 0
            if config.MODEL_CONFIG['max_steps'] > 0 and step > config.MODEL_CONFIG['max_steps']:
                break
        if config.MODEL_CONFIG['max_steps'] > 0 and step > config.MODEL_CONFIG['max_steps']:
                break

def validation(args, model, eval_dataset, epoch):
    results = {}
    val_loss = 0
    val_steps = 0
    total_correct = 0
    val_labels, val_outputs = [], []

    logger.info('####################### Epoch {}: Validation Start #######################\n'.format(epoch))
    logger.info("  Num examples = {}".format(len(eval_dataset)))

    if torch.cuda.is_available():
        if args.target_gpu == 'm': # multi GPU
            device = torch.device("cuda:0")
        else:
            device = torch.device('cuda:' + str(args.target_gpu))
    else:
        device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')

    early_stopping = Early_Stopping(verbose = True)
    early_stopping_m = Early_Stopping_multi(verbose = True)

    for batch_idx, data in enumerate(eval_dataset):
        model.eval()
        ckpt = "VAL_{}epoch_{}_step.pt".format(epoch,val_steps)
        ck_path = os.path.join(args.ck_path, ckpt)
        with torch.no_grad():
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            if args.target_gpu == 'm':
                with torch.cuda.amp.autocast():
                    outputs, pred, real, correct, _ = model(ids,labels, device)
                v_loss = outputs.mean()
            else:
                outputs, pred, labels, correct, _ = model(ids,labels)
                v_loss, logits = outputs[:2]

            total_correct += correct.sum().item()
            
            val_loss += v_loss.mean().item()
            val_loss = val_loss + ((1 / (batch_idx + 1)) * (v_loss.item() - val_loss))
            val_acc = total_correct / len(labels)

            val_f1 = f1_score(labels.tolist(), pred.tolist(), average='weighted')
 
            total_correct = 0

            logger.info('---------------------- [Validation] Epoch {}, Steps {}/{} ----------------------'.format(epoch, batch_idx, len(eval_dataset)))
            logger.info('                     Accuracy {}, Loss {}, F1 score {}\n'.format(val_acc, v_loss, val_f1))

            wandb.log({"(VAL)loss": v_loss, "epoch":epoch, "custom_step": batch_idx, "(VAL)accuracy": val_acc, "(VAL)f1": val_f1})

        val_steps += 1

    return val_loss, val_acc

def evlauate(test_dataloader, args, do_train):
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.target_gpu))
        logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logger.info('We will use the GPU:{}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')
    
    model = BERT(args)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.MODEL_CONFIG['learning_rate'],
                      eps=config.MODEL_CONFIG['adam_epsilon'])

    model.load_state_dict(torch.load(args.load_ck,map_location=device))
    model.eval()

    total_correct, eval_loss, acc, f1 = 0,0,0,0
    pred_list, labels_list = [], []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)

            if args.op == 'test':
                outputs, pred, labels, correct, _ = model(ids,labels)
                lt =labels.tolist()
                pt = pred.tolist()
                correct = np.equal(np.array(pt),np.array(lt))
            
            loss, logits = outputs[:2]

            total_correct += correct.sum().item()

            eval_acc = total_correct / len(labels)

            eval_f1 = f1_score(labels.tolist(), pred.tolist(), average='micro')
            f1 += eval_f1
            acc += eval_acc

            for l,p in zip(lt,pt):
                labels_list.append(l)
                pred_list.append(p)

            total_correct = 0

            logger.info('---------------------- [Test] Steps {}/{} ----------------------'.format(batch_idx, len(test_dataloader)))
            logger.info('                     Accuracy {}, F1 score {}'.format(eval_acc, eval_f1))

            wandb.log({"custom_step": batch_idx, "(TEST)accuracy": eval_acc})
        acc = acc/len(test_dataloader)
        f1 = f1/len(test_dataloader)
        logger.info('----------------------> (Average) Accuracy {}, F1 score {}'.format(acc, f1))
        print(classification_report(labels_list, pred_list))



