import numpy as np
import matplotlib.pyplot as plt
'''# ===============================
# Dati MAE e RMSE per ogni studio
# ===============================

methods = ["PCA+LR", "RF", "SVR", "iPPGtoBP (our)"]

# Organizzazione: SBP, DBP, MAP
mae_sbp = [17.67, 11.62, 16.37, 7.17]
rmse_sbp = [21.84, 14.97, 20.35, 9.05]

mae_dbp = [17.22, 11.41, 10.64, 5.34]
rmse_dbp = [21.20, 14.92, 13.72, 6.44]

mae_map = [0, 11.36, 0, 4.92]    # 0 per valori non riportati
rmse_map = [0, 14.79, 0, 6.25]

x = np.arange(len(methods))  # posizione gruppi
width = 0.25                 # larghezza barre

# Offset per SBP, DBP, MAP
offsets = {"SBP": -width, "DBP": 0, "MAP": width}
colors = {"SBP": "tab:blue", "DBP": "tab:orange", "MAP": "tab:green"}

# ===============================
# Figura unica con 2 sottografi
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)

# --- MAE ---
axes[0].bar(x + offsets["SBP"], mae_sbp, width, label="SBP", color=colors["SBP"], alpha=0.8)
axes[0].bar(x + offsets["DBP"], mae_dbp, width, label="DBP", color=colors["DBP"], alpha=0.8)
axes[0].bar(x + offsets["MAP"], mae_map, width, label="MAP", color=colors["MAP"], alpha=0.8)
axes[0].set_title("MAE (mmHg)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, rotation=20)
axes[0].set_ylabel("Errore (mmHg)")
axes[0].legend()

# --- RMSE ---
axes[1].bar(x + offsets["SBP"], rmse_sbp, width, label="SBP", color=colors["SBP"], alpha=0.8)
axes[1].bar(x + offsets["DBP"], rmse_dbp, width, label="DBP", color=colors["DBP"], alpha=0.8)
axes[1].bar(x + offsets["MAP"], rmse_map, width, label="MAP", color=colors["MAP"], alpha=0.8)
axes[1].set_title("RMSE (mmHg)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods, rotation=20)
axes[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()'''

