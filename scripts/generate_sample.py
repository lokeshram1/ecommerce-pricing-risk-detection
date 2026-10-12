import json, random
from pathlib import Path
random.seed(17)
brands=["Northstar","Acme","Pine Labs","Orbit"]
categories=["Headphones","Storage","Monitors","Cameras"]
def generate(n=240):
    rows=[]
    for i in range(n):
        cat=categories[i%len(categories)]; base={"Headphones":80,"Storage":45,"Monitors":220,"Cameras":350}[cat]; price=round(base*random.uniform(.65,1.4),2)
        rows.append({"product_id":f"SYN-{i:04d}","title":f"{random.choice(brands)} {cat} model {i}","category":cat,"price":price,"list_price":round(price*random.uniform(1,1.3),2),"brand":random.choice(brands),"description":f"Synthetic development listing for {cat}, model {i}.","features":["wireless" if i%2 else "portable","development sample"],"rating":round(random.uniform(3,5),1),"rating_count":random.randint(0,5000),"is_synthetic":True})
    return rows
if __name__=="__main__":
    path=Path("data/sample/products.jsonl"); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x)+"\n" for x in generate()),encoding="utf8"); print(f"Wrote {path} (all records synthetic)")
