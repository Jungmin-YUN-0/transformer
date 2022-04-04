# hyperparameter 설정

#INPUT_DIM: len(SRC.vocab)
#OUTPUT_DIM: len(TRG.vocab)
class hparams():
    def __init__(self):
        self.HIDDEN_DIM = 256  #d_model
        #
        self.ENC_LAYERS = 3
        self.DEC_LAYERS = 3
        #
        self.ENC_HEADS = 8
        self.DEC_HEADS = 8
        #
        self.ENC_PF_DIM = 512
        self.DEC_PF_DIM = 512
        #
        self.ENC_DROPOUT = 0.1
        self.DEC_DROPOUT = 0.1
        
        return
