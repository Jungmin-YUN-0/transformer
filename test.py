import torch
import dill
from tqdm import tqdm
from torchtext.data import Dataset, BucketIterator
from torchtext.data.metrics import bleu_score  # BLEU
from model.Transformer import Transformer
from model.Classification import CF_Transformer
from model.translator import Translator
import argparse
import wandb
import torch.nn.functional as F
from sklearn.metrics import classification_report


class Test():
    def __init__(self, gpu, opt, batch_size, data_pkl, model_save, pred_save, hidden_dim, n_layer, n_head, ff_dim, dropout, data_task):

        self.data_pkl = data_pkl
        self.saved_model = model_save
        self.saved_result = pred_save

        gpu = "cuda:"+gpu
        device = torch.device(gpu if torch.cuda.is_available() else 'cpu')  # cuda(gpu) 사용

        self.HIDDEN_DIM = hidden_dim
        self.ENC_LAYERS = self.DEC_LAYERS = n_layer
        self.ENC_HEADS = self.DEC_HEADS = n_head
        self.ENC_PF_DIM = self.DEC_PF_DIM = ff_dim
        self.ENC_DROPOUT = self.DEC_DROPOUT = dropout
        attn_option = opt
   
        ##########################################################################################################
        data = dill.load(open(self.data_pkl, 'rb'))

        SRC, TRG = data['vocab']['src'], data['vocab']['trg']
        SRC_PAD_IDX = SRC.vocab.stoi["<blank>"]
        if data_task == "MT":
            TRG_PAD_IDX = TRG.vocab.stoi["<blank>"]
            TRG_SOS_IDX = TRG.vocab.stoi["<sos>"]
            TRG_EOS_IDX = TRG.vocab.stoi["<eos>"]

        INPUT_DIM = len(SRC.vocab)
        OUTPUT_DIM = len(TRG.vocab)

        test_data = Dataset(examples=data['test_data'], fields={'src': SRC, 'trg': TRG})
        test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device)

        if data_task == "MT":
            model = Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX, self.HIDDEN_DIM,
                            self.ENC_LAYERS, self.DEC_LAYERS, self.ENC_HEADS, self.DEC_HEADS,
                            self.ENC_PF_DIM, self.DEC_PF_DIM, self.ENC_DROPOUT, self.DEC_DROPOUT, device, attn_option).to(device)
        elif data_task == "CF":
            model = CF_Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, self.HIDDEN_DIM, self.ENC_LAYERS, self.ENC_HEADS, self.ENC_PF_DIM, self.ENC_DROPOUT, device).to(device)

        model.load_state_dict(torch.load(self.saved_model))
        print('[Info] Trained model state loaded.')

        wandb.init(project="transformer", entity="jungminy")
        

        #===============================================================#
        if data_task == 'MT':
            translator = Translator(model=model, beam_size=5, max_seq_len=512, src_pad_idx=SRC_PAD_IDX, trg_pad_idx=TRG_PAD_IDX, trg_bos_idx=TRG_SOS_IDX, trg_eos_idx=TRG_EOS_IDX, device=device).to(device)
            unk_idx = SRC.vocab.stoi[SRC.unk_token]

            pred_trgs = []
            trgs= []
            index=0
            
            print('[Info] Inference ...')
            if attn_option == 'BASE':
                with open(self.saved_result, 'w') as f:
                    for example in tqdm(test_data, desc='  - (Test)', leave=False):
                        #print(' '.join(example.src))
                        src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]

                        src_seq_ext = torch.zeros((1,512), dtype=torch.long) #!#
                        src_seq_ext[:, :len(src_seq)] = torch.LongTensor([src_seq]) #!#
                        pred_seq = translator.translate_sentence(src_seq_ext.to(device)) #!#


                        pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
                        
                        pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
                        #pred_line = pred_line.replace("<sos>", '').replace("<eos>", '')
                        pred_line = pred_line.replace("<sos>", '').replace("<unk>", '').replace("<eos>", '')
                        #print(pred_line)
                        f.write(pred_line.strip() + '\n')

                        ## for BLEU_score
                        pred_trgs.append(pred_line.split(" ")) ##예측            
                        trgs.append([vars(example)['trg']]) ##정답
                        if (index%100)==0:
                            print(f"[{index} / {len(test_data)}")
                            print(f'예측: {pred_line}')
                            print(f"정답: {' '.join(vars(example)['trg'])}")
                        index += 1
                bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25,0.25,0.25,0.25])
                print('[Info] Finished.')
                print(f'BLEU score : {bleu}')
            
            elif attn_option == 'LR':
                with open(self.saved_result, 'w') as f:
                    for example in tqdm(test_data, desc='  - (Test)', leave=False):
                        src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
                        
                        src_seq_ext = torch.zeros((1,512), dtype=torch.long) #!#
                        src_seq_ext[:, :len(src_seq)] = torch.LongTensor([src_seq]) #!#
                        pred_seq = translator.translate_sentence(src_seq_ext.to(device)) #!#
                        #pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))  
                        #print(pred_seq)
                        #print("#")
                        #print("#")
                        #print("#")

                        pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
                        pred_line = pred_line.replace("<sos>", '').replace("<unk>", '').replace("<eos>", '')
                        print(pred_line)
                        f.write(pred_line.strip() + '\n')

                        ## for BLEU_score
                        pred_trgs.append(pred_line.split(" ")) ##예측            
                        trgs.append([vars(example)['trg']]) ##정답
                        if (index%100)==0:
                            print(f"[{index} / {len(test_data)}")
                            print(f'예측: {pred_line}')
                            print(f"정답: {' '.join(vars(example)['trg'])}")
                        index += 1
                bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25,0.25,0.25,0.25])
                print('[Info] Finished.')
                print(f'BLEU score : {bleu}')
            wandb.log({"bleu_score": bleu})

            return
##################################################################################################################
        if data_task == 'CF':
            def get_predictions(model, iterator, device):
                model.eval()

                review_texts = []
                predictions = []
                prediction_probs = []
                real_values = []
                epoch_accuracy = 0

                with torch.no_grad():
                    for batch in tqdm(iterator):
                        input_seq, _ = batch.src
                        target = batch.trg
                        pred = model(input_seq)

                        pred_class = pred.argmax(dim=-1)
                        correct_pred = pred_class.eq(target).sum()
                        accuracy = correct_pred / pred.shape[0]

                        epoch_accuracy += accuracy.item()

                        from sklearn.metrics import f1_score
                        f1_score = f1_score(target.cpu(), pred_class.cpu())
                        
                        
                #predictions = torch.stack(predictions).cpu()
                #prediction_probs = torch.stack(prediction_probs).cpu()
                #real_values = torch.stack(real_values).cpu()
                #return review_texts, predictions, prediction_probs, real_values
                return epoch_accuracy / len(iterator), f1_score

            #pred, predicted_prob = get_predictions(model, test_iterator, device)
            #print(pred, predicted_prob)
            acc, f1_score = get_predictions(model, test_iterator, device)
            print(f"accuracy: {acc}")
            print(f"f1_score: {f1_score}")
            #y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_iterator, device)
            #class_names = ['negative','positive']
            #print(classification_report(y_test, y_pred, target_names=class_names))
##################################################################################################################

if __name__ == "__main__":
    Test()