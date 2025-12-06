import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_radar_chart(df, title, save_path, top_k=6, include_proposed=True):
    """
    Plots a radar chart for IQA comparison.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    metrics = ["SROCC", "PLCC", "KROCC"]

    if include_proposed and "Proposed" in df["Method"].values:
        proposed_row = df[df["Method"] == "Proposed"]
        df = df[df["Method"] != "Proposed"]

        df["Mean"] = df[metrics].mean(axis=1)
        df = df.sort_values("Mean", ascending=False).head(top_k)
        df = pd.concat([df, proposed_row], ignore_index=True)  # Ensure index reset
    else:
        df["Mean"] = df[metrics].mean(axis=1)
        df = df.sort_values("Mean", ascending=False).head(top_k)

    labels = df["Method"].tolist()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    plotted_lines = []

    for i, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        lw = 2.5 if row["Method"] == "Proposed" else 2
        line, = ax.plot(angles, values, label=row["Method"], linewidth=lw)
        ax.fill(angles, values, alpha=0.1 if row["Method"] != "Proposed" else 0.2)
        plotted_lines.append(line)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)

    # ✅ Set Y-axis from 0.2 to 0.8
    ax.set_ylim(0.2, 0.8)

    plt.title(title, size=20)

    # ✅ Custom legend in correct order
    handles = plotted_lines
    labels = df["Method"].tolist()
    ax.legend(handles=handles,
              labels=labels,
              loc="upper right",
              bbox_to_anchor=(1.3, 1.1),
              fontsize=9,
              frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved radar chart: {save_path}")


# ------------------- Data with Proposed Included ------------------------

uwiqa_data = {
    "Method": ["BRISQUE", "SSEQ", "GM-LOG", "DIQaM-NR", "HyperIQA", "UCIQE", "FDUM", "CCF", "UEIQM", "UIQM", "UCIQE_2", "CSN", "Proposed"],
    "SROCC": [0.5689, 0.4892, 0.5831, 0.6991, 0.674, 0.5147, 0.6746, 0.452, 0.5809, 0.6467, 0.6271, 0.6878, 0.726],
    "PLCC":  [0.5614, 0.4688, 0.5629, 0.6745, 0.6931, 0.5122, 0.6683, 0.4488, 0.5925, 0.6297, 0.4862, 0.6791, 0.754],
    "KROCC": [0.4207, 0.3539, 0.4271, 0.5436, 0.5294, 0.3858, 0.522, 0.3306, 0.444, 0.4917, 0.6261, 0.5383, 0.583]
}
uwiqa_df = pd.DataFrame(uwiqa_data)

uid_data = {
    "Method": ["BRISQUE", "SSEQ", "GM-LOG", "DIQaM-NR", "UCIQE", "FDUM", "CCF", "UEIQM", "UIQM", "UCIQE_2", "NUIQ", "CSN", "Proposed"],
    "SROCC": [0.4794, 0.6194, 0.6169, 0.6813, 0.5892, 0.6406, 0.4577, 0.5854, 0.7198, 0.615, 0.7168, 0.6606, 0.768],
    "PLCC":  [0.4689, 0.6006,  0.6011, 0.6841, 0.6335, 0.6464, 0.5371, 0.5625, 0.7211, 0.6474, 0.7266, 0.6652, 0.773],
    "KROCC": [0.3192, 0.6011,  0.4284, 0.5012,  0.434, 0.4589, 0.3314, 0.3984, 0.5252, 0.4503, 0.5293, 0.4748, 0.58]
}
uid_df = pd.DataFrame(uid_data)

saud_data = {
    "Method": ["BRISQUE", "SSEQ", "GM-LOG", "UCIQE", "FDUM", "CCF", "UIQM", "UCIQE_2", "NUIQ", "CSN", "Proposed"],
    "SROCC": [0.5018, 0.3059, 0.3828, 0.3719, 0.2613, 0.3614, 0.5050, 0.3719, 0.7480, 0.7437, 0.763],
    "PLCC":  [0.5276, 0.4146, 0.4084, 0.3674, 0.2815, 0.3872, 0.4939, 0.3674, 0.7413, 0.7467, 0.769],
    "KROCC": [0.3533, 0.2119, 0.2644, 0.2597, 0.1785, 0.1508, 0.3536, 0.2597, 0.5471, 0.5529, 0.571]
}
saud_df = pd.DataFrame(saud_data)

# ---------------------- Generate Radar Charts ---------------------------

plot_radar_chart(uwiqa_df, "UWIQA Dataset – IQA Performance", "radar_uwiqa_with_proposed.png")
plot_radar_chart(uid_df, "UID2021 Dataset – IQA Performance", "radar_uid2021_with_proposed.png")
plot_radar_chart(saud_df, "SAUD Dataset – IQA Performance", "radar_saud_with_proposed.png")
