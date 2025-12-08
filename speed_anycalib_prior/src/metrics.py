import numpy as np
import matplotlib.pyplot as plt

def evaluate_split(df_stats, df_gt, split_name="train"):


    df_eval_mean = df_stats.merge(df_gt, on="gt_id", how="inner")

    if df_eval_mean.empty:
        print(f"[{split_name}] No hay vehículos para evaluar.")
        return

    # Errores usando la MEDIA
    y_true = df_eval_mean["speed_gt_kmh"].values
    y_pred = df_eval_mean["speed_kmh_mean"].values

    err = y_pred - y_true
    abs_err = np.abs(err)

    mae = abs_err.mean()
    rmse = np.sqrt((err**2).mean())
    med_ae = np.median(abs_err)

    # MAPE 
    mask_pos = y_true > 0
    if mask_pos.any():
        mape = (np.abs(err[mask_pos] / y_true[mask_pos]).mean()) * 100.0
    else:
        mape = np.nan

    # R^2
    denom = ((y_true - y_true.mean())**2).sum()
    if denom > 0:
        r2 = 1.0 - ((err**2).sum() / denom)
    else:
        r2 = np.nan



    print(f"\nMétricas usando la MEDIA por auto [{split_name}]:")
    print(f"  MAE   (km/h): {mae:.2f}")
    print(f"  MedAE (km/h): {med_ae:.2f}")
    print(f"  RMSE  (km/h): {rmse:.2f}")
    print(f"  MAPE    (%): {mape:.2f}")
    print(f"  R²         : {r2:.3f}")
    print(f"  Vehículos evaluados: {len(df_eval_mean)}")



    # --------- Scatter: media vs ground truth ---------
    plt.figure(figsize=(8, 6))

    plt.scatter(
        df_eval_mean["speed_gt_kmh"],
        df_eval_mean["speed_kmh_mean"],
        alpha=0.7,
        edgecolors="k",
        label="Vehículos"
    )

    # Línea ideal y = x
    min_v = min(df_eval_mean["speed_gt_kmh"].min(), df_eval_mean["speed_kmh_mean"].min()) - 5
    max_v = max(df_eval_mean["speed_gt_kmh"].max(), df_eval_mean["speed_kmh_mean"].max()) + 5
    plt.plot([min_v, max_v], [min_v, max_v], "r--", label="y = x (ideal)")

    plt.xlabel("Velocidad Ground Truth (km/h)")
    plt.ylabel("Velocidad estimada (MEDIA, km/h)")
    plt.title(f"Velocidad estimada (media) vs Ground Truth [{split_name}]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()