import argparse
from app.ml.ingestion import read_jsonl
from app.ml.train import train
parser=argparse.ArgumentParser(); parser.add_argument("--input",default="data/sample/products.jsonl"); parser.add_argument("--output",default="models"); args=parser.parse_args()
df,stats=read_jsonl(args.input); print("ingestion",stats); print("benchmark",train(df,args.output))
