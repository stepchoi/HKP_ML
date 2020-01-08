def eda_hist(df, name='temp'):
    from matplotlib import pyplot as plt
    import math

    fig = plt.figure(figsize=(50, 50), dpi=80)
    plt.rcParams.update({'font.size': 6})
    k = 1
    for col in df.columns.to_list():
        n = math.ceil(len(df.columns) ** 0.5)
        axis1 = fig.add_subplot(n, n, k)
        axis1.hist(df.loc[df[col].notnull(), col], density=True, bins=50)
        axis1.set_title(col, fontsize=60)
        print(col, k)
        k += 1
    fig.tight_layout()
    fig.savefig(name + '.png')
