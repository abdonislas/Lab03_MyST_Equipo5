#Aplicamos el vecm
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

ART = "artifacts"


def main():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    cols = chosen["grupo"].split("+")

    px = pd.read_csv(f"{ART}/prices_train.csv", index_col=0, parse_dates=True)
    data = np.log(px[cols]).dropna()

    model = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="co")
    res = model.fit()

    alpha = pd.DataFrame({
        "ticker": cols,
        "alpha": np.round(res.alpha[:, 0], 4),
        "pval": np.round(res.pvalues_alpha[:, 0], 4),
    })
    beta = pd.DataFrame({"ticker": cols, "beta": np.round(res.beta[:, 0], 4)})

    alpha.to_csv(f"{ART}/vecm_alpha.csv", index=False)
    beta.to_csv(f"{ART}/vecm_beta.csv", index=False)

    corrige = alpha.loc[alpha["alpha"].abs().idxmax(), "ticker"]
    print("=== Beta (vector de cointegracion) ===");  print(beta.to_string(index=False))
    print("\n=== Alpha (velocidad de ajuste) ===");    print(alpha.to_string(index=False))
    print(f"\n>>> Corrige principalmente el desequilibrio: {corrige} "
          f"(alpha mas grande en magnitud y significativo si pval<0.05)")


if __name__ == "__main__":
    main()