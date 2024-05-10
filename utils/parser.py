import argparse

def get_parser(mode: str):
    parser = argparse.ArgumentParser(mode, description=f"Program of {mode}.")
    parser.add_argument('--load', "-l", action='store_true', help="which load pre-train model or not. if not write -l, stored False")
    parser.add_argument('--save', "-s", action='store_true', help="which save trained model or not. if not write -s, stored False")
    parser.add_argument('--n_epoch', "-e", type=int, help="The num of epoch.")
    parser.add_argument('--batch', "-b", type=int, help="The num of batch.")

    parser.add_argument("--datasetname",    default="slakh")
    parser.add_argument("--f_size",         default=1024)
    parser.add_argument("--seconds",        default=10)
    parser.add_argument("--reduce_silence", default="stems")
    parser.add_argument("--mono",           default=True)
    parser.add_argument("--to1d_mode",      default="mean_linear")
    parser.add_argument("--order",          default="timefreq")
    parser.add_argument("--cases",          default=0b11111)
    parser.add_argument("--tanh",           default=False)
    parser.add_argument('--margin', "-m",   default=0.2, help="The num of margin in Margin Ranking Loss.")
    parser.add_argument('--model', type=str, default="unet5_2d_de5", choices=["unet5_1d_de5", "triplet_1d_de5", "triplet_2d_de5", "jnet_128_embnet"], help="Select model.")
    parser.add_argument('--additional', action='store_true', help="Which additional dataset mode is true or not.")

    return parser