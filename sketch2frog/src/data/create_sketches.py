from src.DexiNed_TF2.run_model import run_DexiNed

def run_dexined(args):
    model = run_DexiNed(args=args)
    if args.model_state=='train':
        model.train()
    elif args.model_state =='test':
        model.test()
    else:
        raise NotImplementedError('Sorry you just can test or train the model, please set in '
                                  'args.model_state=')